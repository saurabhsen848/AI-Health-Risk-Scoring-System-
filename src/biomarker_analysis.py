"""Descriptive biomarker analysis without embedded clinical reference intervals."""

from __future__ import annotations

import re

import pandas as pd


_IDENTIFIER = re.compile(r"(^id$|_id$|^id_|patient|record|index|serial|^no$)", re.IGNORECASE)


def biomarker_columns(frame: pd.DataFrame) -> list[str]:
    """Select measured numeric columns while avoiding obvious row identifiers."""
    columns = []
    for column in frame.columns:
        if _IDENTIFIER.search(str(column)):
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() >= 1:
            columns.append(str(column))
    return columns


def biomarker_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed values; this does not interpret disease or abnormality."""
    columns = biomarker_columns(frame)
    rows = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        rows.append(
            {
                "biomarker": column,
                "observed": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "mean": float(values.mean()) if values.notna().any() else None,
                "median": float(values.median()) if values.notna().any() else None,
                "std_dev": float(values.std()) if values.notna().sum() > 1 else None,
                "minimum": float(values.min()) if values.notna().any() else None,
                "maximum": float(values.max()) if values.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def patient_biomarker_summary(frame: pd.DataFrame, row_position: int) -> pd.DataFrame:
    """Return one selected row's available measured values, without range judgments."""
    if row_position < 0 or row_position >= len(frame):
        raise IndexError("Selected biomarker row is outside the dataset.")
    row = frame.iloc[row_position]
    return pd.DataFrame(
        [
            {"biomarker": column, "value": pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]}
            for column in biomarker_columns(frame)
        ]
    )


def compare_reference_ranges(values: dict[str, object], ranges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Compare to explicitly provided reference bounds; no defaults are supplied."""
    rows = []
    for biomarker, value in values.items():
        if biomarker not in ranges or pd.isna(value):
            continue
        lower, upper = ranges[biomarker]
        numeric = float(value)
        status = "Below supplied range" if numeric < lower else "Above supplied range" if numeric > upper else "Within supplied range"
        rows.append({"biomarker": biomarker, "value": numeric, "lower": lower, "upper": upper, "status": status})
    return pd.DataFrame(rows)

