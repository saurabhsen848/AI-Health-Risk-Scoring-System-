from __future__ import annotations

from src.patient_summary import build_patient_summary, summary_markdown


class ReportSummaryAgent:
    name = "Report/Summary Agent"

    def create(self, patient_features, risk_result, explanations, biomarker_values=None) -> dict:
        summary = build_patient_summary(patient_features, risk_result, explanations, biomarker_values)
        return {"summary": summary, "markdown": summary_markdown(summary)}

