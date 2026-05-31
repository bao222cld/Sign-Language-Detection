# Original NoteBook : https://www.kaggle.com/code/lngcbonguyn/sign-language-detection
#  Sign Language Detection

Real-time Vietnamese sign language recognition using **MediaPipe** skeleton keypoints and a **Transformer / BiLSTM** hybrid model, achieving **99.82% validation accuracy** across 10 gestures.

---

##  Overview

| | |
|---|---|
| **Task** | Multi-class gesture classification from video |
| **Classes** | Again · BathRoom · Finish · Hello · Help · No · Please · SeeYouLater · ThankYou · Yes |
| **Input** | MP4 video → MediaPipe Holistic keypoints |
| **Feature vector** | 134 dims / frame — upper body (8) + left hand (63) + right hand (63) |
| **Sequence length** | 48 frames |
| **Val accuracy** | **99.82%** · Top-2: **99.91%** |
| **Environment** | Kaggle CPU, TensorFlow 2.18, Python 3.11 |

---

##  Model Architecture

```
Input  (48, 134)
  ↓  GaussianNoise(0.01)
  ↓  Conv1D(192, k=5) + BatchNorm + Dropout(0.3)   # local temporal features
  ↓  BiLSTM(128, return_sequences=True) + Dropout   # sequential context
  ↓  LearnedPositionalEmbedding                     # position awareness
  ↓  AttentionPooling                               # soft temporal pooling
  ↓  Dense(256, relu) + Dropout(0.3)
  ↓  Dense(10, softmax)
```

Two custom layers — `LearnedPositionalEmbedding` and `AttentionPooling` — are fully serializable and live in `model.py`.

---

##  Project Structure

```
Sign-Language-Detection/
├── config.py           # All paths, hyperparameters, and constants
├── keypoints.py        # MediaPipe extraction + body-relative normalization
├── dataset.py          # Video discovery, train/val split, parallel dataset build
├── model.py            # Model architecture + load helper
├── train.py            # Training loop (reads .npy arrays, saves checkpoints)
├── evaluate.py         # Accuracy/loss curves, confusion matrix, per-class accuracy
├── predict_video.py    # CLI: run inference on a video, write annotated output
├── requirements.txt
└── README.md
```

---

##  Quickstart

```bash
pip install -r requirements.txt
```

### 1 — Build dataset
```bash
# Edit DATA_DIR in config.py first
python dataset.py
```
Videos with `section_` in their filename go to the validation set; the rest go to training.

### 2 — Train
```bash
python train.py
```
Saves `best_model.keras` (lowest val loss) and `final_model.keras`.

### 3 — Evaluate
```bash
python evaluate.py
```
Produces four PNG plots inside `SAVE_DIR`:
`accuracy_loss.png` · `overfitting_gap.png` · `confusion_matrix.png` · `per_class_acc.png`

### 4 — Run on a video
```bash
python predict_video.py --input my_video.mp4 --output result.mp4
```

---

##  Training Results

| Metric | Train | Validation |
|---|---|---|
| Accuracy (epoch 50) | 99.99% | **99.82%** |
| Top-2 Accuracy | 100% | **99.91%** |
| Loss | 0.136 | 0.139 |

Val accuracy exceeded 80% by epoch 2 and plateaued above 99% from around epoch 20.

---

## 🔧 Key Design Decisions

| Decision | Rationale |
|---|---|
| **Skeleton-only input** | No raw pixels → robust to lighting, background, camera angle |
| **Body-relative normalization** | Centers + scales by shoulder–hip distance → invariant to subject height and distance |
| **Sliding window (stride 8)** | Augments sequence count without duplicating full clips |
| **Middle-2/3 inference window** | Skips motion blur at clip boundaries for more stable predictions |
| **Parallel video processing** | `ThreadPoolExecutor` for fast dataset build on CPU |

---

## 🛠️ Tech Stack

`Python 3.11` · `TensorFlow 2.18` · `MediaPipe 0.10.14` · `OpenCV` · `scikit-learn` · `seaborn`
