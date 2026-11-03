from __future__ import annotations

from src.explainability import global_feature_importance, local_probability_effects
from utils.explanation import get_risk_explanation


class ExplanationAgent:
    name = "Explanation Agent"

    def explain(self, artifact: dict, patient_features: dict, risk_result: dict) -> dict:
        global_importance = global_feature_importance(artifact)
        local_effects = local_probability_effects(artifact, patient_features, risk_result["risk_level"])
        legacy_context = []
        if artifact.get("source") == "legacy_demo":
            legacy_context = get_risk_explanation(
                risk_result["risk_level"],
                patient_features["age"],
                patient_features["bmi"],
                patient_features["blood_pressure"],
                patient_features["glucose"],
                patient_features["smoking"],
                patient_features["exercise"],
            )
        if not local_effects.empty:
            contributors = []
            for row in local_effects.head(3).itertuples(index=False):
                direction = "increased" if row.probability_change > 0 else "decreased" if row.probability_change < 0 else "did not change"
                contributors.append(
                    f"Changing {row.feature.replace('_', ' ')} from {row.patient_value} to the training reference value {row.reference_value} {direction} this class probability by {abs(row.probability_change):.1%}."
                )
            basis = "One-feature-at-a-time comparison with training reference values; this is model behavior, not a causal or medical explanation."
        elif not global_importance.empty:
            contributors = [
                f"{row.feature.replace('_', ' ').title()} has global importance {row.importance:.3f} in this model."
                for row in global_importance.head(3).itertuples(index=False)
            ]
            basis = "Global model importance only; patient-specific contribution and direction are not supported by this artifact."
        else:
            contributors = ["This model does not expose feature contribution information."]
            basis = "No model-specific explanation is available."
        return {
            "global_importance": global_importance,
            "local_effects": local_effects,
            "potential_contributors": contributors,
            "legacy_rule_context": legacy_context,
            "basis": basis,
        }
