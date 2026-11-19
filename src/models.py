"""Keras model builders and reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy and TensorFlow before building a model."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_rnn(window: int, n_features: int) -> keras.Model:
    """Build the two-layer SimpleRNN described in the guide."""
    model = keras.Sequential(
        [
            layers.Input(shape=(window, n_features)),
            layers.SimpleRNN(50, return_sequences=True),
            layers.Dropout(0.2),
            layers.SimpleRNN(50),
            layers.Dropout(0.2),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def build_lstm(window: int, n_features: int, units: int = 64) -> keras.Model:
    """Build the two-layer LSTM described in the guide."""
    model = keras.Sequential(
        [
            layers.Input(shape=(window, n_features)),
            layers.LSTM(units, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(units),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
