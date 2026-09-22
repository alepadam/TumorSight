"""
Tests for src/evaluate.py — including a hand-checkable case for
no_tumor_miss_rate, since that's the single most clinically relevant metric
this module produces.
"""

import numpy as np
import pytest

from src.evaluate import evaluate_predictions

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]


def test_evaluate_predictions_perfect_classifier():
    y_true = np.array(["glioma", "meningioma", "pituitary", "no_tumor"] * 3)
    y_pred = y_true.copy()

    results = evaluate_predictions(y_true, y_pred, CLASS_NAMES)

    assert results["no_tumor_miss_rate"] == 0.0
    assert results["report"].loc["accuracy"].iloc[0] == 1.0


def test_no_tumor_miss_rate_hand_checkable():
    """4 true no_tumor cases; model misses 1 (predicts it as glioma).
    Miss rate should be exactly 1/4 = 0.25."""
    y_true = np.array(["no_tumor"] * 4 + ["glioma"] * 4)
    y_pred = np.array(["no_tumor", "no_tumor", "no_tumor", "glioma"] + ["glioma"] * 4)

    results = evaluate_predictions(y_true, y_pred, CLASS_NAMES)

    assert results["no_tumor_miss_rate"] == pytest.approx(0.25)


def test_confusion_matrix_shape_and_labels():
    y_true = np.array(["glioma", "meningioma", "pituitary", "no_tumor"])
    y_pred = np.array(["glioma", "glioma", "pituitary", "no_tumor"])

    results = evaluate_predictions(y_true, y_pred, CLASS_NAMES)
    cm = results["confusion_matrix"]

    assert list(cm.index) == CLASS_NAMES
    assert list(cm.columns) == CLASS_NAMES
    assert cm.loc["meningioma", "glioma"] == 1  # the one misclassification
