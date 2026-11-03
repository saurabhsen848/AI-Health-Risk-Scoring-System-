"""Inventory condition-specific labels without guessing disease risk."""

from __future__ import annotations

import pandas as pd


CONDITIONS = ("anemia", "dengue", "sepsis")


def disease_label_inventory(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Report whether supplied tables contain clearly named condition label columns."""
    rows = []
    for source, frame in datasets.items():
        normalized = {str(column).strip().lower().replace(" ", "_"): column for column in frame.columns}
        for condition in CONDITIONS:
            candidates = [
                (name, column)
                for name, column in normalized.items()
                if name == condition or name in {f"{condition}_label", f"{condition}_status", f"{condition}_diagnosis"}
            ]
            for _, column in candidates:
                values = frame[column]
                rows.append(
                    {
                        "source": source,
                        "condition_label": str(column),
                        "condition": condition,
                        "labeled_rows": int(values.notna().sum()),
                        "distinct_values": int(values.dropna().nunique()),
                        "model_available": False,
                    }
                )
    return pd.DataFrame(rows, columns=["source", "condition_label", "condition", "labeled_rows", "distinct_values", "model_available"])

