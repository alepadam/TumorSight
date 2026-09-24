"""
Interpretability utilities for studying what these models actually learn — not just
their final accuracy numbers.

Two tools:
1. Grad-CAM: shows WHERE in an image a model is looking when it makes a prediction
   (a heatmap over the input). Useful for sanity-checking a model is actually looking
   at the tumor region, not some unrelated artifact (e.g. a scanner watermark).
2. Filter/feature-map extraction: shows WHAT patterns a convolutional layer has
   learned to detect. Comparing a from-scratch CNN's early filters (which start as
   random noise) against DenseNet's pretrained filters (which already encode edges/
   textures from ImageNet) is the most direct way to *see* why transfer learning helps.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def grad_cam(
    model: tf.keras.Model,
    image: np.ndarray,
    last_conv_layer_name: str,
    pred_index: int | None = None,
) -> np.ndarray:
    """
    Computes a Grad-CAM heatmap for a single image.

    Parameters
    ----------
    model: a trained Keras model
    image: a single preprocessed image, shape (H, W, C) — NOT batched
    last_conv_layer_name: name of the last convolutional layer to explain from.
        For build_custom_cnn(), this is typically the last Conv2D layer's name.
        For build_transfer_model() (DenseNet), pass the DenseNet sub-model's last
        conv layer — e.g. "conv5_block16_concat" for DenseNet121.
    pred_index: which class to explain. Defaults to the model's top predicted class.

    Returns
    -------
    A 2D heatmap, shape (H', W') matching the last conv layer's spatial size,
    normalized to [0, 1]. Resize it to the input image size for overlay plotting.
    """
    image_batch = np.expand_dims(image, axis=0)

    # Sequential models (e.g. build_custom_cnn) expose `.outputs` (a list) but
    # not the singular `.output` property until built in a functional context —
    # using `.outputs[0]` works for both Sequential and functional models.
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.outputs[0]],
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_batch)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU: we only care about features that positively influence the predicted class
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def grad_cam_transfer_model(
    model: tf.keras.Model,
    image: np.ndarray,
    base_model_layer_name: str = "densenet121",
    pred_index: int | None = None,
) -> np.ndarray:
    """
    Grad-CAM specifically for models built by build_transfer_model() — i.e. a
    pretrained base wrapped as a single nested layer, followed by a custom head.

    Why this isn't just grad_cam() with a different layer name: in this Keras
    version, an inner layer's output inside a NESTED sub-model (the wrapped
    DenseNet121) cannot be connected back to the outer model's declared inputs
    via Model(inputs=..., outputs=...) — it raises "Output ... is not connected
    to inputs". The fix used here: call the base model directly inside a
    GradientTape (producing its final feature map, which for include_top=False
    IS the base model's actual output — no need to reach further inside it),
    then manually replay the outer model's remaining layers (GAP, Dense, etc.)
    on that output within the same tape, so gradients trace correctly.

    Assumes the same preprocessing as build_transfer_model (DenseNet's
    preprocess_input) — this function is intentionally specific to that
    architecture rather than a general-purpose nested-model Grad-CAM.
    """
    from tensorflow.keras import applications

    base_model = model.get_layer(base_model_layer_name)
    image_batch = np.expand_dims(image, axis=0)
    preprocessed = applications.densenet.preprocess_input(image_batch)

    # Layers that come after the base model in the outer model, in original order
    post_base_layers = []
    found_base = False
    for layer in model.layers:
        if layer is base_model:
            found_base = True
            continue
        if found_base:
            post_base_layers.append(layer)

    with tf.GradientTape() as tape:
        conv_output = base_model(preprocessed, training=False)
        tape.watch(conv_output)
        x = conv_output
        for layer in post_base_layers:
            x = layer(x)
        predictions = x
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output_single = conv_output[0]
    heatmap = conv_output_single @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def find_last_conv_layer_name(model: tf.keras.Model) -> str:
    """
    Convenience helper: walks the model's layers in reverse and returns the name
    of the last layer with a 4D output (the signature of a conv/pooling layer),
    so callers don't need to hardcode a layer name if they don't know it offhand.
    """
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except AttributeError:
            continue
    raise ValueError("No 4D-output (convolutional) layer found in this model.")


def get_first_layer_filters(model: tf.keras.Model, conv_layer_index: int = 0) -> np.ndarray:
    """
    Extracts the learned filter weights from a model's first convolutional layer.

    Returns array of shape (kernel_h, kernel_w, in_channels, n_filters).

    For build_custom_cnn(): pass the model directly — random-initialized filters
    before training will look like noise; after training, edge/blob-like patterns
    should start to emerge.

    For build_transfer_model(): pass the DenseNet sub-model (model.get_layer(...)
    for the densenet layer), whose filters are pretrained on ImageNet and already
    show clear edge/color/texture detectors even with freeze_base=True.
    """
    conv_layers = [layer for layer in model.layers if "conv" in layer.name.lower()]
    if not conv_layers:
        raise ValueError("No convolutional layers found in this model.")
    weights = conv_layers[conv_layer_index].get_weights()
    if not weights:
        raise ValueError(f"Layer {conv_layers[conv_layer_index].name} has no weights.")
    return weights[0]  # kernel weights (first element); second element (if present) is bias


def get_feature_maps(model: tf.keras.Model, image: np.ndarray, layer_name: str) -> np.ndarray:
    """
    Runs a single image through the model up to layer_name and returns that
    layer's activation ("feature map") — i.e. what that layer actually
    produces for this specific image, as opposed to get_first_layer_filters()
    which shows the learned weights regardless of input.

    Returns array of shape (H', W', n_channels).
    """
    activation_model = tf.keras.models.Model(
        inputs=model.inputs, outputs=model.get_layer(layer_name).output
    )
    image_batch = np.expand_dims(image, axis=0)
    activations = activation_model.predict(image_batch, verbose=0)
    return activations[0]
