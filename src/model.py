"""Keras LSTM architecture and training utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocessing import Vocabulary


def create_model(
    vocabulary_size: int,
    sequence_length: int,
    embedding_dim: int,
    lstm_units: int,
    dropout: float,
    learning_rate: float = 1e-3,
    lstm_layers: int = 1,
    recurrent_dropout: float = 0.0,
) -> Any:
    """Build a configurable next-word model with one or two LSTM layers."""
    import tensorflow as tf

    if vocabulary_size < 2:
        raise ValueError("Vocabulary must contain at least two tokens.")
    if sequence_length < 1 or embedding_dim < 1 or lstm_units < 1:
        raise ValueError("Sequence length, embedding size, and LSTM units must be positive.")
    if not 0 <= dropout < 1:
        raise ValueError("Dropout must be in the range [0, 1).")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be a finite positive number.")
    if lstm_layers not in (1, 2):
        raise ValueError("The number of LSTM layers must be 1 or 2.")
    if not 0 <= recurrent_dropout < 1:
        raise ValueError("Recurrent dropout must be in the range [0, 1).")

    layers: list[Any] = [
        tf.keras.layers.Input(shape=(sequence_length,)),
        tf.keras.layers.Embedding(vocabulary_size, embedding_dim),
    ]
    for layer_index in range(lstm_layers):
        layers.append(
            tf.keras.layers.LSTM(
                lstm_units,
                return_sequences=layer_index < lstm_layers - 1,
                recurrent_dropout=recurrent_dropout,
            )
        )
    layers.extend(
        [
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(vocabulary_size, activation="softmax"),
        ]
    )
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: Any,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    vocabulary: Vocabulary,
    model_path: Path,
    vocabulary_path: Path,
    history_path: Path,
    plot_path: Path,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
) -> dict[str, list[float]]:
    """Train with early stopping, save the best model, and persist results."""
    import tensorflow as tf

    if epochs < 1 or batch_size < 1 or patience < 0:
        raise ValueError("Epochs and batch size must be positive; patience cannot be negative.")

    if train_inputs.ndim != 2 or validation_inputs.ndim != 2:
        raise ValueError("Training and validation inputs must be rank-2 arrays.")
    if train_inputs.shape[1] != validation_inputs.shape[1]:
        raise ValueError("Training and validation sequence lengths must match.")
    if len(train_inputs) != len(train_targets) or len(validation_inputs) != len(validation_targets):
        raise ValueError("Input and target counts must match for each dataset.")
    if not len(train_targets) or not len(validation_targets):
        raise ValueError("Training and validation datasets must both be non-empty.")
    if (
        np.min(train_targets) < 0
        or np.max(train_targets) >= vocabulary.size
        or np.min(validation_targets) < 0
        or np.max(validation_targets) >= vocabulary.size
    ):
        raise ValueError("Target IDs must be within the model vocabulary.")

    tf.keras.utils.set_random_seed(seed)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    vocabulary.save(vocabulary_path)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=1,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
    ]
    result = model.fit(
        train_inputs,
        train_targets,
        validation_data=(validation_inputs, validation_targets),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    history = {
        metric: [float(value) for value in values]
        for metric, values in result.history.items()
    }
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    _save_training_plot(history, plot_path)
    return history


def _save_training_plot(history: dict[str, list[float]], destination: Path) -> None:
    """Save training/validation loss and accuracy curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.get("loss", [])) + 1)
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(epochs, history.get("loss", []), marker="o", label="Training")
    loss_axis.plot(epochs, history.get("val_loss", []), marker="o", label="Validation")
    loss_axis.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    loss_axis.legend()
    accuracy_axis.plot(
        epochs, history.get("accuracy", []), marker="o", label="Training"
    )
    accuracy_axis.plot(
        epochs, history.get("val_accuracy", []), marker="o", label="Validation"
    )
    accuracy_axis.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    accuracy_axis.legend()
    figure.suptitle("LSTM training history")
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
