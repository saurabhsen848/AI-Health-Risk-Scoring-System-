from __future__ import annotations

import pandas as pd

from src.regional_analysis import aggregate_regional_trends, regional_summary


class RegionalTrendAgent:
    name = "Regional Trend Agent"

    def analyze(self, frame: pd.DataFrame, region_column: str, **kwargs) -> dict:
        counts, ranked = aggregate_regional_trends(frame, region_column, **kwargs)
        return {"counts": counts, "ranked": ranked, "summary": regional_summary(counts, region_column)}

