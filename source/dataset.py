"""
dataset.py — Discover videos, split train/val, and build numpy arrays.

Split strategy:
  - Videos whose filename contains 'section_' go to validation.
  - All other videos go to training.

Usage:
    python dataset.py
"""

import os
import glob
import json
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import DATA_DIR, SAVE_DIR, SEQLEN, RAW_FEAT_DIM, STRIDE, MAX_WORKERS
from keypoints import read_video_keypoints, normalize_sequence


# ── Label discovery ───────────────────────────────────────────────────────────

def discover_actions(data_dir: str = DATA_DIR) -> list:
    """Return sorted list of class names (one per subdirectory)."""
    actions = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    assert len(actions) >= 2, f"Need ≥2 class folders in {data_dir}"
    return actions


# ── Train / val split ─────────────────────────────────────────────────────────

def split_samples(actions: list, data_dir: str = DATA_DIR):
    """
    Split videos into train/val sets.

    Videos with 'section_' in the filename → val; rest → train.

    Returns:
        train_samples: list of (video_path, label)
        val_samples:   list of (video_path, label)
    """
    train_samples, val_samples = [], []

    for label in actions:
        folder = os.path.join(data_dir, label)
        for vpath in sorted(glob.glob(os.path.join(folder, "*.mp4"))):
            fname = os.path.basename(vpath).lower()
            target = val_samples if "section_" in fname else train_samples
            target.append((vpath, label))

    return train_samples, val_samples


def print_split_stats(train_samples, val_samples, actions):
    print("=== VIDEO SPLIT SUMMARY ===")
    for label in actions:
        t = sum(1 for _, l in train_samples if l == label)
        v = sum(1 for _, l in val_samples   if l == label)
        print(f"  {label:14s} | train={t:3d} | val={v:3d}")
    print(f"  Total          | train={len(train_samples):3d} | val={len(val_samples):3d}")


# ── Parallel dataset builder ──────────────────────────────────────────────────

def _process_video(vpath, label, actions, stride):
    seqs    = read_video_keypoints(vpath, seqlen=SEQLEN, step=stride, head_sec=0.3)
    seqs    = [normalize_sequence(s) for s in seqs]
    onehot  = tf.keras.utils.to_categorical(actions.index(label), num_classes=len(actions))
    X_v, y_v = [], []
    for s in seqs:
        if s.shape == (SEQLEN, RAW_FEAT_DIM):
            X_v.append(s)
            y_v.append(onehot)
    return X_v, y_v, label


def build_dataset(samples, actions, stride=STRIDE, max_workers=MAX_WORKERS):
    """
    Extract features from all videos in parallel and stack into numpy arrays.

    Returns:
        X: np.ndarray of shape (N, SEQLEN, RAW_FEAT_DIM)
        y: np.ndarray of shape (N, num_classes)
        counts: dict {label: num_sequences}
    """
    X_list, y_list, counts = [], [], {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_video, v, l, actions, stride): (v, l)
            for v, l in samples
        }
        for fut in tqdm(as_completed(futures), total=len(futures)):
            X_v, y_v, label = fut.result()
            X_list.extend(X_v)
            y_list.extend(y_v)
            counts[label] = counts.get(label, 0) + len(X_v)

    return np.array(X_list), np.array(y_list), counts


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    actions = discover_actions()
    print(f"Classes ({len(actions)}): {actions}")

    with open(os.path.join(SAVE_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(actions, f, ensure_ascii=False, indent=2)

    train_samples, val_samples = split_samples(actions)
    print_split_stats(train_samples, val_samples, actions)

    print("\nBuilding training set...")
    X_train, y_train, counts_train = build_dataset(train_samples, actions)

    print("Building validation set...")
    X_val, y_val, counts_val = build_dataset(val_samples, actions)

    np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(SAVE_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(SAVE_DIR, "y_val.npy"),   y_val)

    print(f"\n✅  X_train {X_train.shape} | y_train {y_train.shape}")
    print(f"✅  X_val   {X_val.shape}   | y_val   {y_val.shape}")
    print(f"📊  Train counts: {counts_train}")
    print(f"📊  Val   counts: {counts_val}")
    print(f"\nArrays saved to {SAVE_DIR}")
