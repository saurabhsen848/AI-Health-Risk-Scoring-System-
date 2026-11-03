from __future__ import annotations

import pandas as pd

from .biomarker_agent import BiomarkerAnalysisAgent
from .explanation_agent import ExplanationAgent
from .patient_agent import PatientAnalysisAgent
from .report_agent import ReportSummaryAgent
from .risk_agent import RiskPredictionAgent
from .regional_agent import RegionalTrendAgent


class AssessmentOrchestrator:
    """Coordinate local agents and return a transparent, structured assessment."""

    def __init__(self):
        self.patient_agent = PatientAnalysisAgent()
        self.biomarker_agent = BiomarkerAnalysisAgent()
        self.risk_agent = RiskPredictionAgent()
        self.explanation_agent = ExplanationAgent()
        self.report_agent = ReportSummaryAgent()
        self.regional_agent = RegionalTrendAgent()

    def run(
        self,
        artifact: dict,
        patient_features: dict,
        biomarker_frame: pd.DataFrame | None = None,
        biomarker_row: int | None = None,
        regional_frame: pd.DataFrame | None = None,
        regional_options: dict | None = None,
    ) -> dict:
        patient = self.patient_agent.analyze(patient_features)
        risk = self.risk_agent.predict(artifact, patient["feature_values"])
        explanation = self.explanation_agent.explain(artifact, patient["feature_values"], risk)
        biomarkers = self.biomarker_agent.analyze(biomarker_frame, biomarker_row)
        regional = None
        if regional_frame is not None:
            options = dict(regional_options or {})
            region_column = options.pop("region_column", None)
            if not region_column:
                raise ValueError("regional_options must include region_column when regional data is supplied.")
            regional = self.regional_agent.analyze(regional_frame, region_column, **options)
        biomarker_values = {}
        if not biomarkers["patient_values"].empty:
            biomarker_values = dict(zip(biomarkers["patient_values"]["biomarker"], biomarkers["patient_values"]["value"]))
        report_context = list(explanation["potential_contributors"])
        report_context.extend(f"Legacy prototype rule context: {item}" for item in explanation.get("legacy_rule_context", []))
        report = self.report_agent.create(patient["feature_values"], risk, report_context, biomarker_values)
        return {
            "patient": patient,
            "risk": risk,
            "explanation": explanation,
            "biomarkers": biomarkers,
            "report": report,
            "regional": regional,
            "agents": [
                self.patient_agent.name,
                self.risk_agent.name,
                self.explanation_agent.name,
                self.biomarker_agent.name,
                *([self.regional_agent.name] if regional is not None else []),
                self.report_agent.name,
            ],
        }
