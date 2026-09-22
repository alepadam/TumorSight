"""
Reusable training entry point. Wraps Keras .fit() with sensible defaults
(checkpointing best val_accuracy, early stopping, CSV logging) so every
model in this project is trained and logged consistently.
"""

from __future__ import annotations

import logging
from pathlib import Path

import tensorflow as tf

logger = logging.getLogger(__name__)


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    run_name: str,
    epochs: int = 30,
    checkpoint_dir: Path = Path("models/saved_models"),
    patience: int = 7,
) -> tf.keras.callbacks.History:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_name}_best.keras"
    log_path = checkpoint_dir / f"{run_name}_history.csv"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger(str(log_path)),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
        ),
    ]

    logger.info("Starting training run: %s (%d epochs max)", run_name, epochs)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )
    logger.info(
        "Finished training run: %s — best val_accuracy=%.4f",
        run_name,
        max(history.history["val_accuracy"]),
    )
    return history
