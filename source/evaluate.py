"""
evaluate.py — Load the best model and produce evaluation plots.

Outputs (saved to SAVE_DIR):
  - accuracy_loss.png    : Train vs. val accuracy and loss curves
  - overfitting_gap.png  : Val − Train loss gap per epoch
  - confusion_matrix.png : 10×10 confusion matrix on the val set
  - per_class_acc.png    : Per-class accuracy bar chart

Usage:
    python evaluate.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import SAVE_DIR
from model import load_model


# ── Load ──────────────────────────────────────────────────────────────────────

X_val = np.load(os.path.join(SAVE_DIR, "X_val.npy"))
y_val = np.load(os.path.join(SAVE_DIR, "y_val.npy"))

with open(os.path.join(SAVE_DIR, "labels.json"), encoding="utf-8") as f:
    actions = json.load(f)

history = np.load(os.path.join(SAVE_DIR, "history.npy"), allow_pickle=True).item()
model   = load_model(os.path.join(SAVE_DIR, "best_model.keras"))

y_pred        = model.predict(X_val, batch_size=32)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_val,  axis=1)


# ── 1. Accuracy & Loss curves ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history["accuracy"],     label="Train Acc")
axes[0].plot(history["val_accuracy"], label="Val Acc")
axes[0].set_title("Training vs Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.4)

axes[1].plot(history["loss"],     label="Train Loss")
axes[1].plot(history["val_loss"], label="Val Loss")
axes[1].set_title("Training vs Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "accuracy_loss.png"), dpi=150)
plt.show()
print("Saved accuracy_loss.png")


# ── 2. Overfitting gap ────────────────────────────────────────────────────────

gap = np.array(history["val_loss"]) - np.array(history["loss"])

plt.figure(figsize=(10, 4))
plt.plot(gap, color="red", label="Val − Train Loss")
plt.axhline(0, linestyle="--", color="gray")
plt.title("Overfitting Gap per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Gap  (> 0 → overfitting)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "overfitting_gap.png"), dpi=150)
plt.show()
print("Saved overfitting_gap.png")


# ── 3. Confusion matrix ───────────────────────────────────────────────────────

cm = confusion_matrix(y_true_labels, y_pred_labels)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="viridis",
    xticklabels=actions, yticklabels=actions,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Validation Set")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=150)
plt.show()
print("Saved confusion_matrix.png")


# ── 4. Per-class accuracy ─────────────────────────────────────────────────────

class_acc = []
for cls in range(len(actions)):
    idx = np.where(y_true_labels == cls)[0]
    acc = np.mean(y_pred_labels[idx] == cls) if len(idx) > 0 else 0.0
    class_acc.append(acc)

plt.figure(figsize=(12, 5))
plt.bar(actions, class_acc, color="skyblue")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy — Validation Set")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "per_class_acc.png"), dpi=150)
plt.show()
print("Saved per_class_acc.png")

print("\n📊 Per-class breakdown:")
for a, acc in zip(actions, class_acc):
    print(f"  {a:14s}: {acc * 100:5.1f}%")
