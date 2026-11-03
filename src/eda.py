"""Exploratory-analysis figures for user-provided datasets."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import re


_IDENTIFIER = re.compile(r"(^id$|[_\s]id$|^id[_\s]|^index$|^serial(?:[_\s]number)?$|^no$)", re.IGNORECASE)


def numeric_eda_columns(frame: pd.DataFrame) -> list[str]:
    """Select numeric measurements while omitting obvious row identifiers."""
    return [
        str(column)
        for column in frame.select_dtypes(include="number").columns
        if not _IDENTIFIER.search(str(column))
    ]


def feature_distribution(frame: pd.DataFrame, feature: str):
    """Create a histogram for one selected numeric feature."""
    if feature not in frame.columns or not pd.api.types.is_numeric_dtype(frame[feature]):
        raise ValueError(f"'{feature}' is not a numeric feature in this dataset.")
    fig, ax = plt.subplots(figsize=(7, 4))
    values = frame[feature].dropna()
    ax.hist(values, bins=min(30, max(5, int(len(values) ** 0.5))), color="#287c8e", edgecolor="white")
    ax.set(title=f"Distribution of {feature.replace('_', ' ').title()}", xlabel=feature, ylabel="Rows")
    fig.tight_layout()
    return fig


def numeric_boxplots(frame: pd.DataFrame):
    """Create boxplots for numeric columns without labeling values as abnormal."""
    columns = numeric_eda_columns(frame)
    if not columns:
        return None
    fig, ax = plt.subplots(figsize=(max(7, len(columns) * 1.2), 4))
    frame[columns].plot(kind="box", ax=ax, rot=35, grid=True)
    ax.set_title("Numeric feature ranges (descriptive only)")
    fig.tight_layout()
    return fig


def correlation_heatmap(frame: pd.DataFrame):
    """Create a correlation heatmap when at least two numeric columns exist."""
    numeric = frame[numeric_eda_columns(frame)]
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(6, len(corr.columns) * 0.8), max(4, len(corr) * 0.65)))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.set_title("Feature correlation (descriptive, not causal)")
    fig.colorbar(image, ax=ax, fraction=0.045)
    fig.tight_layout()
    return fig


def category_distribution(frame: pd.DataFrame, column: str):
    """Create a count chart for a selected outcome/category column."""
    if column not in frame.columns:
        raise ValueError(f"'{column}' is not present in this dataset.")
    counts = frame[column].fillna("Missing").astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax, color="#d17a22")
    ax.set(title=f"Distribution of {column.replace('_', ' ').title()}", xlabel=column, ylabel="Rows")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    return fig
