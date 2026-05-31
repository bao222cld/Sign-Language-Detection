"""
predict_video.py — Run inference on a video and save annotated output.

Draws the MediaPipe skeleton and overlays the predicted gesture label
with a confidence bar on every frame.

Usage:
    python predict_video.py --input path/to/video.mp4 --output result.mp4
"""

import argparse
import json
import os

import cv2
import numpy as np
from mediapipe import solutions as mp_solutions

from config import SAVE_DIR, SEQLEN, RAW_FEAT_DIM
from keypoints import extract_keypoints_from_frame, normalize_sequence
from model import load_model


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _confidence_color(p: float):
    """Blue → green gradient based on confidence."""
    p = float(np.clip(p, 0, 1))
    if p < 0.5:
        return (255, int(255 * p / 0.5), 0)
    return (int(255 * (1 - (p - 0.5) / 0.5)), 255, 0)


def draw_skeleton(frame, results):
    mp_draw   = mp_solutions.drawing_utils
    mp_style  = mp_solutions.drawing_styles
    mp_pose   = mp_solutions.pose
    mp_hands  = mp_solutions.hands

    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_style.get_default_pose_landmarks_style(),
        )
    for hand_lms, spec_lm, spec_cn in [
        (results.left_hand_landmarks,
         mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
         mp_draw.DrawingSpec(color=(0, 128, 255), thickness=2)),
        (results.right_hand_landmarks,
         mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
         mp_draw.DrawingSpec(color=(255, 128, 0), thickness=2)),
    ]:
        if hand_lms:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS, spec_lm, spec_cn)


def draw_label_box(frame, text: str, conf: float, pos=(30, 80)):
    x, y   = pos
    color  = _confidence_color(conf)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 15, y - 55), (x + 650, y + 35), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    cv2.putText(frame, text, (x + 4, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 5, cv2.LINE_AA)

    bar_x1, bar_y1 = x, y + 20
    bar_x2, bar_y2 = x + 500, bar_y1 + 20
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (40, 40, 40), -1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + int(conf * 500), bar_y2), color, -1)


# ── Core inference pipeline ───────────────────────────────────────────────────

def predict_video(video_path: str, out_path: str, actions: list, model) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    raw_frames, kpt_seq = [], []

    with mp_solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
    ) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res  = holistic.process(rgb)
            kpt_seq.append(extract_keypoints_from_frame(res))
            raw_frames.append((frame.copy(), res))

    cap.release()

    kpts = np.array(kpt_seq, dtype=np.float32)
    total = len(kpts)

    # Build sequences (focus on the middle 2/3 of the video)
    mid_start = int(total * (1 / 6))
    mid_end   = int(total * (5 / 6))

    seqs, indices = [], []
    for i in range(0, total - SEQLEN + 1):
        end_frame = i + SEQLEN - 1
        if mid_start <= end_frame <= mid_end:
            seq = normalize_sequence(kpts[i : i + SEQLEN])
            seqs.append(seq)
            indices.append(end_frame)

    if not seqs:
        print("[WARN] Video too short for inference.")
        writer.release()
        return

    X      = np.array(seqs, dtype=np.float32)
    preds  = model.predict(X, batch_size=8)
    cls_   = np.argmax(preds, axis=1)
    probs_ = np.max(preds, axis=1)

    frame_cls  = dict(zip(indices, cls_))
    frame_prob = dict(zip(indices, probs_))

    smooth_conf = None
    for i, (frame, res) in enumerate(raw_frames):
        draw_skeleton(frame, res)
        if i in frame_cls:
            smooth_conf = (
                probs_[list(indices).index(i)]
                if smooth_conf is None
                else 0.7 * smooth_conf + 0.3 * frame_prob[i]
            )
            label = f"{actions[frame_cls[i]]}  |  {smooth_conf * 100:.1f}%"
            draw_label_box(frame, label, smooth_conf)
        writer.write(frame)

    writer.release()
    print(f"✅  Output saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign language video inference")
    parser.add_argument("--input",  required=True, help="Path to input video")
    parser.add_argument("--output", default="prediction_output.mp4", help="Path for output video")
    parser.add_argument("--model",  default=os.path.join(SAVE_DIR, "best_model.keras"))
    parser.add_argument("--labels", default=os.path.join(SAVE_DIR, "labels.json"))
    args = parser.parse_args()

    with open(args.labels, encoding="utf-8") as f:
        actions = json.load(f)

    model = load_model(args.model)
    print(f"Model loaded from {args.model}")

    predict_video(args.input, args.output, actions, model)
