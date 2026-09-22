"""
Tests for src/train.py. Uses a tiny synthetic dataset and 2 epochs — the goal
is to catch integration bugs (wrong callback args, wrong checkpoint paths,
API mismatches) fast, not to test model quality.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

# Intentionally after importorskip — see test_models.py for why.
from src.models import build_custom_cnn  # noqa: E402
from src.train import train_model  # noqa: E402


def _make_synthetic_dataset(n_samples: int, num_classes: int = 4, batch_size: int = 4):
    rng = np.random.default_rng(0)
    x = rng.random((n_samples, 32, 32, 1)).astype("float32")
    y = rng.integers(0, num_classes, size=n_samples)
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)


def test_train_model_runs_and_returns_history(tmp_path):
    train_ds = _make_synthetic_dataset(20)
    val_ds = _make_synthetic_dataset(8)
    model = build_custom_cnn(input_shape=(32, 32, 1), num_classes=4)

    history = train_model(
        model,
        train_ds,
        val_ds,
        run_name="pytest_smoke_test",
        epochs=2,
        checkpoint_dir=tmp_path,
        patience=5,
    )

    assert "val_accuracy" in history.history
    assert len(history.history["val_accuracy"]) == 2


def test_train_model_saves_checkpoint(tmp_path):
    train_ds = _make_synthetic_dataset(20)
    val_ds = _make_synthetic_dataset(8)
    model = build_custom_cnn(input_shape=(32, 32, 1), num_classes=4)

    train_model(
        model,
        train_ds,
        val_ds,
        run_name="pytest_checkpoint_test",
        epochs=2,
        checkpoint_dir=tmp_path,
        patience=5,
    )

    checkpoint_path = tmp_path / "pytest_checkpoint_test_best.keras"
    log_path = tmp_path / "pytest_checkpoint_test_history.csv"
    assert checkpoint_path.exists()
    assert log_path.exists()
