"""Model explanations using native coefficients, feature importance, and local perturbations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .preprocessing import HEALTH_FEATURES


def _model_parts(artifact: dict[str, Any]):
    model = artifact["pipeline"]
    if hasattr(model, "named_steps"):
        return model, model.named_steps.get("preprocessor"), model.named_steps.get("classifier")
    return model, None, model


def _base_feature(transformed_name: str) -> str:
    for feature in HEALTH_FEATURES:
        if transformed_name == feature or transformed_name.startswith(feature + "_"):
            return feature
    return transformed_name


def global_feature_importance(artifact: dict[str, Any]) -> pd.DataFrame:
    """Return global model importance; positive/negative directions are not inferred for trees."""
    _, preprocessor, estimator = _model_parts(artifact)
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
        direction = None
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        importance = np.mean(np.abs(coefficients), axis=0)
        direction = np.mean(coefficients, axis=0) if coefficients.shape[0] == 1 else None
    else:
        return pd.DataFrame(columns=["feature", "importance", "direction"])

    names = list(preprocessor.get_feature_names_out()) if preprocessor is not None else HEALTH_FEATURES[: len(importance)]
    if len(names) != len(importance):
        names = [f"feature_{index}" for index in range(len(importance))]
    rows = []
    for index, (name, value) in enumerate(zip(names, importance)):
        rows.append(
            {
                "feature": _base_feature(str(name)),
                "importance": float(value),
                "direction": float(direction[index]) if direction is not None else np.nan,
            }
        )
    table = pd.DataFrame(rows).groupby("feature", as_index=False).agg(
        importance=("importance", "sum"), direction=("direction", "mean")
    )
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def local_probability_effects(
    artifact: dict[str, Any], patient_features: dict[str, Any], predicted_label: str
) -> pd.DataFrame:
    """Show one-at-a-time probability changes against training reference values, if stored."""
    reference = artifact.get("metadata", {}).get("feature_reference_values", {})
    if not reference:
        return pd.DataFrame(columns=["feature", "patient_value", "reference_value", "probability_change"])
    model, _, _ = _model_parts(artifact)
    if not hasattr(model, "predict_proba"):
        return pd.DataFrame(columns=["feature", "patient_value", "reference_value", "probability_change"])
    classes = list(getattr(model, "classes_", []))
    if not classes and hasattr(model, "named_steps"):
        classes = list(getattr(model.named_steps.get("classifier"), "classes_", []))
    target_index = next(
        (i for i, item in enumerate(classes) if _display_class(item) == predicted_label), None
    )
    if target_index is None:
        return pd.DataFrame(columns=["feature", "patient_value", "reference_value", "probability_change"])

    base = pd.DataFrame([[patient_features[name] for name in HEALTH_FEATURES]], columns=HEALTH_FEATURES)
    actual_probability = float(model.predict_proba(base)[0][target_index])
    rows = []
    for feature in HEALTH_FEATURES:
        if feature not in reference:
            continue
        counterfactual = base.copy()
        counterfactual.loc[0, feature] = reference[feature]
        reference_probability = float(model.predict_proba(counterfactual)[0][target_index])
        rows.append(
            {
                "feature": feature,
                "patient_value": patient_features[feature],
                "reference_value": reference[feature],
                "probability_change": actual_probability - reference_probability,
            }
        )
    return pd.DataFrame(rows).sort_values("probability_change", key=lambda series: series.abs(), ascending=False)


def _display_class(value: Any) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}.get(int(value), str(value))
    normalized = str(value).strip().lower()
    return {
        "low": "Low Risk",
        "low risk": "Low Risk",
        "medium": "Medium Risk",
        "medium risk": "Medium Risk",
        "high": "High Risk",
        "high risk": "High Risk",
    }.get(normalized, str(value))
