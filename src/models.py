"""
Model architectures: a from-scratch baseline CNN and a DenseNet121 transfer-learning
model. See README / project notes for why transfer learning is expected to outperform
the baseline given the dataset's size (~3k images, 233 patients before augmentation).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import applications, layers, models


def build_custom_cnn(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = 4,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """From-scratch CNN baseline. No pretrained weights."""
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="custom_cnn_from_scratch",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transfer_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    freeze_base: bool = True,
    learning_rate: float = 1e-4,
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    """
    DenseNet121 with a new classification head.

    `weights="imagenet"` (default) downloads pretrained weights — use this for
    actual training. `weights=None` builds the same architecture with random
    initialization and no network call, which is what tests use to verify
    shapes/freezing logic quickly and without depending on internet access.
    """
    base_model = applications.DenseNet121(
        include_top=False, weights=weights, input_shape=input_shape
    )
    base_model.trainable = not freeze_base

    inputs = layers.Input(shape=input_shape)
    x = applications.densenet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="densenet121_transfer_learning")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_for_finetuning(
    model: tf.keras.Model,
    num_layers_to_unfreeze: int = 30,
    learning_rate: float = 1e-5,
) -> tf.keras.Model:
    """Unfreezes the last N layers of the DenseNet base for a fine-tuning phase."""
    base_model = next(layer for layer in model.layers if "densenet" in layer.name)
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
