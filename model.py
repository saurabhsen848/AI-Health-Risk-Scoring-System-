"""Load prediction artifacts and score one health-risk feature row."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import HEALTH_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINED_MODEL_PATH = PROJECT_ROOT / "models" / "health_risk_pipeline.joblib"
LEGACY_MODEL_PATH = PROJECT_ROOT / "health_risk_model.pkl"
RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}


def load_prediction_artifact(
    trained_path: str | Path = TRAINED_MODEL_PATH,
    legacy_path: str | Path = LEGACY_MODEL_PATH,
) -> dict[str, Any] | None:
    """Prefer a properly trained pipeline; fall back to the original prototype pickle."""
    trained = Path(trained_path)
    if trained.is_file():
        payload = joblib.load(trained)
        if isinstance(payload, dict) and "pipeline" in payload:
            return {"pipeline": payload["pipeline"], "metadata": payload.get("metadata", {}), "source": "trained"}
        if hasattr(payload, "predict"):
            return {"pipeline": payload, "metadata": {}, "source": "trained"}
        raise ValueError(f"Model artifact at {trained} does not contain a usable prediction model.")

    legacy = Path(legacy_path)
    if legacy.is_file():
        payload = joblib.load(legacy)
        if hasattr(payload, "predict"):
            return {
                "pipeline": payload,
                "metadata": {"training_rows": 12, "model_name": "Random Forest (legacy prototype)"},
                "source": "legacy_demo",
            }
        raise ValueError(f"Legacy model artifact at {legacy} does not contain a usable prediction model.")
    return None


def _display_label(value: Any) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return RISK_LABELS.get(int(value), str(value))
    normalized = str(value).strip().lower()
    known = {"low": "Low Risk", "low risk": "Low Risk", "medium": "Medium Risk", "medium risk": "Medium Risk", "high": "High Risk", "high risk": "High Risk"}
    return known.get(normalized, str(value))


def predict_health_risk(artifact: dict[str, Any], patient_features: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
    """Predict using class ordering reported by the estimator, never assumed ordering."""
    if artifact is None or "pipeline" not in artifact:
        raise ValueError("No trained or prototype model is available.")
    model = artifact["pipeline"]
    if isinstance(patient_features, pd.DataFrame):
        row = patient_features.loc[:, HEALTH_FEATURES]
    else:
        missing = [name for name in HEALTH_FEATURES if name not in patient_features]
        if missing:
            raise ValueError(f"Missing patient features: {', '.join(missing)}")
        row = pd.DataFrame([[patient_features[name] for name in HEALTH_FEATURES]], columns=HEALTH_FEATURES)

    if artifact.get("source") == "legacy_demo":
        model_input: Any = row.to_numpy(dtype=float)
    else:
        model_input = row

    predicted_value = model.predict(model_input)[0]
    classes = list(getattr(model, "classes_", []))
    if not classes and hasattr(model, "named_steps"):
        classes = list(getattr(model.named_steps.get("classifier"), "classes_", []))
    labels = [_display_label(value) for value in classes]
    probabilities = model.predict_proba(model_input)[0] if hasattr(model, "predict_proba") else None
    probability_map = (
        {label: float(probability) for label, probability in zip(labels, probabilities)} if probabilities is not None else {}
    )
    predicted_label = _display_label(predicted_value)
    return {
        "risk_level": predicted_label,
        "predicted_class": str(predicted_value),
        "probabilities": probability_map,
        "class_labels": labels,
        "feature_values": row.iloc[0].to_dict(),
    }


# Backwards-compatible name used by older project scripts.
def predict_risk(model, age, bmi, blood_pressure, glucose, smoking, exercise):
    artifact = {"pipeline": model, "source": "legacy_demo"}
    result = predict_health_risk(
        artifact,
        dict(age=age, bmi=bmi, blood_pressure=blood_pressure, glucose=glucose, smoking=smoking, exercise=exercise),
    )
    return result["risk_level"], np.array(list(result["probabilities"].values()), dtype=float)
