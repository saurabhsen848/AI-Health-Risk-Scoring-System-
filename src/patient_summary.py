"""Patient-friendly structured summaries of the assessment prototype."""

from __future__ import annotations

from typing import Any


def build_patient_summary(
    patient_features: dict[str, Any],
    risk_result: dict[str, Any] | None = None,
    explanations: list[str] | None = None,
    biomarker_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine only the values and results passed in; no diagnosis is inferred."""
    return {
        "health_parameters": dict(patient_features),
        "biomarker_values": dict(biomarker_values or {}),
        "risk_assessment": dict(risk_result or {}),
        "potential_contributors": list(explanations or []),
        "interpretation": "Decision-support prototype only. This summary is not a medical diagnosis.",
    }


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = ["## Patient health summary", "", "### Entered health parameters"]
    for name, value in summary.get("health_parameters", {}).items():
        lines.append(f"- {name.replace('_', ' ').title()}: {value}")
    risk = summary.get("risk_assessment", {})
    if risk:
        lines.extend(["", "### Risk assessment", f"- Category: {risk.get('risk_level', 'Unavailable')}"])
        if risk.get("probabilities"):
            lines.extend([f"- {name}: {probability:.1%}" for name, probability in risk["probabilities"].items()])
    if summary.get("biomarker_values"):
        lines.extend(["", "### Biomarker values"])
        lines.extend([f"- {name}: {value}" for name, value in summary["biomarker_values"].items()])
    if summary.get("potential_contributors"):
        lines.extend(["", "### Model or rule-based context"])
        lines.extend([f"- {item}" for item in summary["potential_contributors"]])
    lines.extend(["", f"_{summary.get('interpretation', '')}_"])
    return "\n".join(lines)
