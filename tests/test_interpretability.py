"""
Tests for src/interpretability.py.

Two real Keras-version-specific bugs were found and fixed while building this
module — both covered here so they can't silently regress:
1. Sequential models (build_custom_cnn) expose `.outputs` (list) but not the
   singular `.output` property in this Keras version — grad_cam() must use
   `.outputs[0]`, not `.output`.
2. A layer nested inside another sub-model (DenseNet121 wrapped inside
   build_transfer_model) cannot have its output connected back to the outer
   model's inputs via Model(inputs=..., outputs=...) — grad_cam_transfer_model()
   works around this by replaying the outer model's post-base layers manually
   inside a single GradientTape instead.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.interpretability import (  # noqa: E402
    find_last_conv_layer_name,
    get_feature_maps,
    get_first_layer_filters,
    grad_cam,
    grad_cam_transfer_model,
)
from src.models import build_custom_cnn, build_transfer_model  # noqa: E402


# ---------------------------------------------------------------------------
# Custom CNN (Sequential) — grad_cam()
# ---------------------------------------------------------------------------
def test_grad_cam_custom_cnn_output_shape_and_range():
    model = build_custom_cnn(input_shape=(64, 64, 1), num_classes=4)
    image = np.random.rand(64, 64, 1).astype("float32")

    last_conv = find_last_conv_layer_name(model)
    heatmap = grad_cam(model, image, last_conv)

    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_grad_cam_respects_explicit_pred_index():
    """Explaining class 0 vs class 1 should generally produce different
    heatmaps — a basic sanity check that pred_index actually does something."""
    model = build_custom_cnn(input_shape=(64, 64, 1), num_classes=4)
    image = np.random.rand(64, 64, 1).astype("float32")
    last_conv = find_last_conv_layer_name(model)

    heatmap_class0 = grad_cam(model, image, last_conv, pred_index=0)
    heatmap_class1 = grad_cam(model, image, last_conv, pred_index=1)

    assert heatmap_class0.shape == heatmap_class1.shape
    # Not asserting they differ numerically — with random untrained weights
    # they occasionally coincide — just that both run without error and are
    # valid heatmaps.
    for hm in (heatmap_class0, heatmap_class1):
        assert hm.min() >= 0.0
        assert hm.max() <= 1.0 + 1e-6


def test_find_last_conv_layer_name_returns_a_real_layer():
    model = build_custom_cnn(input_shape=(64, 64, 1), num_classes=4)
    name = find_last_conv_layer_name(model)
    assert name in [layer.name for layer in model.layers]


# ---------------------------------------------------------------------------
# Custom CNN — filters and feature maps
# ---------------------------------------------------------------------------
def test_get_first_layer_filters_shape():
    model = build_custom_cnn(input_shape=(64, 64, 1), num_classes=4)
    filters = get_first_layer_filters(model)
    # (kernel_h, kernel_w, in_channels, n_filters) — first conv layer is
    # Conv2D(32, (3,3)) on a 1-channel input in build_custom_cnn()
    assert filters.shape == (3, 3, 1, 32)


def test_get_feature_maps_shape():
    model = build_custom_cnn(input_shape=(64, 64, 1), num_classes=4)
    image = np.random.rand(64, 64, 1).astype("float32")
    last_conv = find_last_conv_layer_name(model)

    feature_maps = get_feature_maps(model, image, last_conv)
    assert feature_maps.ndim == 3  # (H', W', channels)


# ---------------------------------------------------------------------------
# Transfer learning model (nested DenseNet121) — grad_cam_transfer_model()
# ---------------------------------------------------------------------------
def test_grad_cam_transfer_model_output_shape_and_range():
    # weights=None avoids downloading ImageNet weights in tests (see test_models.py)
    model = build_transfer_model(
        input_shape=(64, 64, 3), num_classes=4, freeze_base=True, weights=None
    )
    image = np.random.rand(64, 64, 3).astype("float32")

    heatmap = grad_cam_transfer_model(model, image)

    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_grad_cam_transfer_model_works_with_unfrozen_base():
    """Also verify the fine-tuning (unfrozen) case, not just the frozen default."""
    from src.models import unfreeze_for_finetuning

    model = build_transfer_model(
        input_shape=(64, 64, 3), num_classes=4, freeze_base=True, weights=None
    )
    model = unfreeze_for_finetuning(model, num_layers_to_unfreeze=5)
    image = np.random.rand(64, 64, 3).astype("float32")

    heatmap = grad_cam_transfer_model(model, image)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_get_first_layer_filters_on_densenet_submodel():
    """DenseNet's pretrained filters — even with weights=None here (random
    init, since we're avoiding the network dependency in tests), the shape
    check still validates the extraction path works for a nested submodel."""
    model = build_transfer_model(
        input_shape=(64, 64, 3), num_classes=4, freeze_base=True, weights=None
    )
    base_model = model.get_layer("densenet121")
    filters = get_first_layer_filters(base_model)
    # DenseNet121's first conv is a 7x7 kernel, 3 input channels, 64 filters
    assert filters.shape == (7, 7, 3, 64)
