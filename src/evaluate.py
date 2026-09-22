"""
Evaluation utilities: per-class metrics matter more than overall accuracy here,
since confusing "no tumor" with a tumor class is a much more serious error than
confusing two tumor subtypes with each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """
    Returns a dict with:
      - 'report': pandas DataFrame version of sklearn's classification_report
      - 'confusion_matrix': pandas DataFrame, rows=true, cols=predicted
      - 'no_tumor_miss_rate': fraction of true 'no_tumor' cases predicted as a tumor class
        (the single most clinically relevant error mode to watch)
    """
    # `labels=class_names` is required alongside `target_names`: without it,
    # sklearn infers the label set from whatever classes actually appear in
    # y_true/y_pred, and raises if that set's size doesn't match len(target_names)
    # — which happens any time a batch doesn't include every class (common with
    # small validation sets, or a class the model rarely predicts early on).
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    no_tumor_miss_rate = None
    if "no_tumor" in class_names:
        idx = class_names.index("no_tumor")
        total_no_tumor = cm[idx].sum()
        correctly_identified = cm[idx][idx]
        if total_no_tumor > 0:
            no_tumor_miss_rate = 1 - (correctly_identified / total_no_tumor)

    return {
        "report": report_df,
        "confusion_matrix": cm_df,
        "no_tumor_miss_rate": no_tumor_miss_rate,
    }
