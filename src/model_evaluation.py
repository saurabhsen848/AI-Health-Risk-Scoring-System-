"""Classification metrics and model-comparison helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred, labels=None) -> dict[str, float]:
    """Calculate consistently weighted multiclass metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def confusion_table(y_true, y_pred, labels=None) -> pd.DataFrame:
    """Return a labeled confusion matrix table."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred), key=str)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=[str(label) for label in labels], columns=[str(label) for label in labels])


def comparison_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Format actual evaluation metrics for side-by-side display."""
    frame = pd.DataFrame(rows)
    if not frame.empty:
        metric_columns = [column for column in frame if column != "model"]
        frame[metric_columns] = frame[metric_columns].round(4)
    return frame
