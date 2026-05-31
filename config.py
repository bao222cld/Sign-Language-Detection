"""
config.py — Centralized paths, hyperparameters, and constants.
Change DATA_DIR and BASE_DIR to match your environment.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = "/kaggle/working"
SAVE_DIR = os.path.join(BASE_DIR, "sign_language_detection")
PRED_DIR = os.path.join(BASE_DIR, "pred_videos")
DATA_DIR = "/kaggle/input/dataset/SignLanguageVideo_VipCut"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# ── Feature dimensions ────────────────────────────────────────────────────────
SEQLEN       = 48    # frames per sliding window
RAW_FEAT_DIM = 134   # 4 upper-body pts×2 + 21 left-hand pts×3 + 21 right-hand pts×3

# ── Training hyperparameters ──────────────────────────────────────────────────
EPOCHS        = 50
BATCH_SIZE    = 32
LEARNING_RATE = 2e-4
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.02
STRIDE        = 8    # sliding-window stride for dataset augmentation
MAX_WORKERS   = 8    # parallel video processing threads

# ── Class labels (auto-discovered from DATA_DIR at runtime) ───────────────────
# Populated by dataset.py; listed here for reference.
ACTIONS = [
    "Again", "BathRoom", "Finish", "Hello", "Help",
    "No", "Please", "SeeYouLater", "ThankYou", "Yes",
]
