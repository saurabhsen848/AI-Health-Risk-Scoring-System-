"""Historical regional aggregation for datasets that explicitly contain geography."""

from __future__ import annotations

import pandas as pd


REGION_CANDIDATES = ("region", "state", "district", "city", "location", "area")
DATE_CANDIDATES = ("date", "datetime", "reported_at", "month", "year")


def suggested_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lookup = {str(column).strip().lower().replace(" ", "_"): str(column) for column in frame.columns}
    return next((lookup[name] for name in candidates if name in lookup), None)


def aggregate_regional_trends(
    frame: pd.DataFrame,
    region_column: str,
    *,
    category_column: str | None = None,
    date_column: str | None = None,
    period: str = "M",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate observed row counts by region/category/time; not an outbreak detector."""
    if region_column not in frame:
        raise ValueError(f"Region column '{region_column}' is not in this dataset.")
    if category_column and category_column not in frame:
        raise ValueError(f"Category column '{category_column}' is not in this dataset.")
    working = frame.copy()
    working[region_column] = working[region_column].astype("string").fillna("Unknown")
    group_columns = [region_column]
    if category_column:
        group_columns.append(category_column)
    if date_column:
        if date_column not in frame:
            raise ValueError(f"Date column '{date_column}' is not in this dataset.")
        parsed = pd.to_datetime(working[date_column], errors="coerce")
        working = working.loc[parsed.notna()].copy()
        working["_period"] = parsed.loc[parsed.notna()].dt.to_period(period).astype(str)
        group_columns.append("_period")
    if working.empty:
        raise ValueError("No usable rows remain after parsing the selected date column.")
    counts = working.groupby(group_columns, dropna=False, observed=False).size().rename("records").reset_index()
    return counts, counts.sort_values("records", ascending=False).reset_index(drop=True)


def regional_summary(counts: pd.DataFrame, region_column: str) -> str:
    if counts.empty:
        return "No regional records are available for this selection."
    totals = counts.groupby(region_column)["records"].sum().sort_values(ascending=False)
    return f"{len(totals)} regions appear in the selected historical dataset. Highest record count: {totals.index[0]} ({int(totals.iloc[0])} rows). Counts describe the supplied data only; they do not establish population rates or detect outbreaks."

