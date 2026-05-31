"""
model.py — Transformer-based sign language classification model.

Architecture:
  Input (48, 134)
    → GaussianNoise
    → Conv1D(192) + BatchNorm + Dropout
    → BiLSTM(128, return_sequences=True) + Dropout
    → LearnedPositionalEmbedding
    → AttentionPooling
    → Dense(256) + Dropout
    → Dense(num_classes, softmax)

Custom layers (LearnedPositionalEmbedding, AttentionPooling) are fully
serializable — pass them via custom_objects when loading a saved model.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

from config import SEQLEN, RAW_FEAT_DIM, LEARNING_RATE, WEIGHT_DECAY, LABEL_SMOOTH


# ── Custom layers ─────────────────────────────────────────────────────────────

class LearnedPositionalEmbedding(layers.Layer):
    """Trainable positional embedding added element-wise to the sequence."""

    def __init__(self, max_len: int, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        d_model = input_shape[-1]
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(self.max_len, d_model),
            initializer="uniform",
            trainable=True,
        )

    def call(self, x):
        return x + self.pos_emb

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len})
        return cfg


class AttentionPooling(layers.Layer):
    """
    Soft attention pooling over the time axis.
    Learns a single query vector and collapses (T, d) → (d,).
    """

    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(
            shape=(d, 1), initializer="glorot_uniform", trainable=True
        )

    def call(self, x):
        scores = tf.nn.softmax(tf.squeeze(tf.matmul(x, self.W), axis=-1), axis=1)
        return tf.reduce_sum(x * scores[:, :, None], axis=1)

    def get_config(self):
        return super().get_config()


CUSTOM_OBJECTS = {
    "LearnedPositionalEmbedding": LearnedPositionalEmbedding,
    "AttentionPooling": AttentionPooling,
}


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(num_classes: int, seq_len: int = SEQLEN, feat_dim: int = RAW_FEAT_DIM):
    """
    Build and compile the model.

    Args:
        num_classes: Number of gesture classes.
        seq_len:     Sequence length (default: SEQLEN from config).
        feat_dim:    Feature dimension per frame (default: RAW_FEAT_DIM).

    Returns:
        Compiled tf.keras.Model.
    """
    inputs = layers.Input(shape=(seq_len, feat_dim))

    x = layers.GaussianNoise(0.01)(inputs)

    # Local temporal features
    x = layers.Conv1D(192, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Sequential context
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)

    # Position-aware attention pooling
    x = LearnedPositionalEmbedding(max_len=seq_len)(x)
    x = AttentionPooling()(x)

    # Classification head
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="sign_language_detector")

    model.compile(
        optimizer=optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        loss=losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2")],
    )

    return model


def load_model(path: str):
    """Load a saved model with the custom objects registered."""
    return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)
