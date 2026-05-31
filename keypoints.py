"""
keypoints.py — MediaPipe keypoint extraction and sequence normalization.

Keypoint layout (134 dims per frame):
  [0:8]   Upper-body: landmarks 11,12,23,24  ×  (x, y)
  [8:71]  Left hand:  21 landmarks           ×  (x, y, z)
  [71:134] Right hand: 21 landmarks          ×  (x, y, z)
"""

import numpy as np
import cv2
from mediapipe import solutions as mp_solutions

from config import SEQLEN, RAW_FEAT_DIM

_UPPER_IDX = [11, 12, 23, 24]   # shoulder-L, shoulder-R, hip-L, hip-R


# ── Per-frame extraction ──────────────────────────────────────────────────────

def extract_keypoints_from_frame(results) -> np.ndarray:
    """
    Extract a 134-dim feature vector from a single MediaPipe Holistic result.

    Returns:
        np.ndarray of shape (134,), dtype float32.
    """
    upper = np.zeros((4, 2), dtype=np.float32)
    if results.pose_landmarks:
        for i, idx in enumerate(_UPPER_IDX):
            lm = results.pose_landmarks.landmark[idx]
            upper[i] = [lm.x, lm.y]

    lh = np.zeros((21, 3), dtype=np.float32)
    rh = np.zeros((21, 3), dtype=np.float32)

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            lh[i] = [lm.x, lm.y, lm.z]

    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            rh[i] = [lm.x, lm.y, lm.z]

    return np.concatenate([upper.flatten(), lh.flatten(), rh.flatten()])


# ── Sequence normalization ────────────────────────────────────────────────────

def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Body-relative normalization for one sequence (SEQLEN, 134).

    Shifts the origin to the midpoint between shoulders and hips, then
    scales by the shoulder-to-hip distance so the representation is
    invariant to camera distance and subject height.

    Returns:
        np.ndarray of shape (SEQLEN, 134), dtype float32.
    """
    s = seq.copy()
    upper = s[:, :8]                          # (T, 8)
    lh    = s[:, 8:71].reshape(-1, 21, 3)    # (T, 21, 3)
    rh    = s[:, 71:].reshape(-1, 21, 3)     # (T, 21, 3)

    ls  = upper[:, 0:2]   # left shoulder  (lm 11)
    rs  = upper[:, 2:4]   # right shoulder (lm 12)
    lhc = upper[:, 4:6]   # left hip       (lm 23)
    rhc = upper[:, 6:8]   # right hip      (lm 24)

    shoulders = (ls + rs) / 2
    hips      = (lhc + rhc) / 2
    center    = (shoulders + hips) / 2                                   # (T, 2)
    scale     = np.linalg.norm(shoulders - hips, axis=1, keepdims=True) + 1e-6  # (T, 1)

    lh[:, :, :2] = (lh[:, :, :2] - center[:, None, :]) / scale[:, None, :]
    rh[:, :, :2] = (rh[:, :, :2] - center[:, None, :]) / scale[:, None, :]

    return np.concatenate([upper, lh.reshape(-1, 63), rh.reshape(-1, 63)], axis=1).astype(np.float32)


# ── Video → sequence list ─────────────────────────────────────────────────────

def read_video_keypoints(
    vpath: str,
    seqlen: int = SEQLEN,
    step: int = 1,
    head_sec: float = 0.0,
    tail_sec: float = 0.0,
) -> list:
    """
    Read an MP4 file, extract per-frame keypoints, and return a list of
    sliding-window sequences of shape (seqlen, RAW_FEAT_DIM).

    Args:
        vpath:    Path to the video file.
        seqlen:   Number of frames per window.
        step:     Sliding-window stride (frames).
        head_sec: Seconds to skip at the beginning.
        tail_sec: Seconds to skip at the end.

    Returns:
        List of np.ndarray, each shape (seqlen, RAW_FEAT_DIM).
        Empty list if the video is shorter than seqlen usable frames.
    """
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {vpath}")
        return []

    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_f = int(round(fps * head_sec))
    end_f   = total - int(round(fps * tail_sec)) if total > 0 else int(1e9)

    if start_f > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    frame_kpts = []
    with mp_solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
    ) as holistic:
        idx = start_f
        while True:
            ok, frame = cap.read()
            if not ok or idx >= end_f:
                break
            idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            frame_kpts.append(extract_keypoints_from_frame(res))

    cap.release()

    arr = np.array(frame_kpts, dtype=np.float32)
    if len(arr) < seqlen:
        return []

    return [arr[i : i + seqlen] for i in range(0, len(arr) - seqlen + 1, step)]
