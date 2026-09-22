"""
Lightweight tests for src/models.py — checks models build and produce correctly
shaped output, without doing any real training (keeps CI fast).

All build_transfer_model() calls here use weights=None: this builds the same
architecture with random initialization instead of downloading ImageNet weights,
so these tests run fast and don't depend on internet access. Actual training
should use the default weights="imagenet".
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

# These imports are intentionally after importorskip: if tensorflow isn't
# installed, we want the whole module to skip cleanly rather than fail to
# import src.models (which itself imports tensorflow).
from src.models import (  # noqa: E402
    build_custom_cnn,
    build_transfer_model,
    unfreeze_for_finetuning,
)


def test_custom_cnn_output_shape():
    model = build_custom_cnn(input_shape=(224, 224, 1), num_classes=4)
    dummy_input = np.random.rand(2, 224, 224, 1).astype("float32")
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (2, 4)
    # softmax output should sum to ~1 per row
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-4)


def test_transfer_model_output_shape():
    model = build_transfer_model(
        input_shape=(224, 224, 3), num_classes=4, freeze_base=True, weights=None
    )
    dummy_input = np.random.rand(2, 224, 224, 3).astype("float32")
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (2, 4)
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-4)


def test_transfer_model_base_is_frozen_by_default():
    model = build_transfer_model(freeze_base=True, weights=None)
    base = next(layer for layer in model.layers if "densenet" in layer.name)
    assert base.trainable is False


def test_transfer_model_base_trainable_when_unfrozen():
    model = build_transfer_model(freeze_base=False, weights=None)
    base = next(layer for layer in model.layers if "densenet" in layer.name)
    assert base.trainable is True


def test_unfreeze_for_finetuning_makes_base_trainable():
    model = build_transfer_model(freeze_base=True, weights=None)
    model = unfreeze_for_finetuning(model, num_layers_to_unfreeze=10)
    base = next(layer for layer in model.layers if "densenet" in layer.name)
    assert base.trainable is True


def test_unfreeze_for_finetuning_only_unfreezes_last_n_layers():
    """The whole point of unfreeze_for_finetuning is partial unfreezing —
    verify early base layers stay frozen while later ones don't."""
    model = build_transfer_model(freeze_base=True, weights=None)
    model = unfreeze_for_finetuning(model, num_layers_to_unfreeze=5)
    base = next(layer for layer in model.layers if "densenet" in layer.name)

    trainable_flags = [layer.trainable for layer in base.layers]
    # Early layers should remain frozen
    assert not any(trainable_flags[:-5])
    # The last 5 should be trainable
    assert all(trainable_flags[-5:])
