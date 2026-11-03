from __future__ import annotations

from model import predict_health_risk


class RiskPredictionAgent:
    name = "Risk Prediction Agent"

    def predict(self, artifact: dict, patient_features: dict) -> dict:
        return predict_health_risk(artifact, patient_features)
