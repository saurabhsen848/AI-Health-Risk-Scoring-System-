from __future__ import annotations

from typing import Any

from src.preprocessing import HEALTH_FEATURES


class PatientAnalysisAgent:
    name = "Patient Analysis Agent"

    def analyze(self, features: dict[str, Any]) -> dict[str, Any]:
        missing = [feature for feature in HEALTH_FEATURES if feature not in features or features[feature] is None]
        if missing:
            raise ValueError(f"Patient assessment is missing: {', '.join(missing)}")
        return {"feature_values": {name: features[name] for name in HEALTH_FEATURES}, "complete": True}

