from __future__ import annotations

import pandas as pd

from src.biomarker_analysis import biomarker_summary, patient_biomarker_summary


class BiomarkerAnalysisAgent:
    name = "Biomarker Analysis Agent"

    def analyze(self, frame: pd.DataFrame | None, row_position: int | None = None) -> dict:
        if frame is None:
            return {"available": False, "summary": pd.DataFrame(), "patient_values": pd.DataFrame()}
        values = patient_biomarker_summary(frame, row_position) if row_position is not None else pd.DataFrame()
        return {"available": True, "summary": biomarker_summary(frame), "patient_values": values}
