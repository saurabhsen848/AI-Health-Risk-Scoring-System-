"""Legacy prototype rule context, separate from model explanations."""


def get_risk_explanation(risk_level, age, bmi, blood_pressure, glucose, smoking, exercise):
    """Describe which fixed demonstration rules fired; these are not medical cutoffs."""
    observations = []
    if age >= 50:
        observations.append("The prototype's age-at-least-50 demonstration rule was triggered.")
    if bmi >= 25:
        observations.append("The prototype's BMI-at-least-25 demonstration rule was triggered.")
    if blood_pressure >= 130:
        observations.append("The prototype's systolic-pressure-at-least-130 demonstration rule was triggered.")
    if glucose >= 126:
        observations.append("The prototype's glucose-at-least-126 demonstration rule was triggered.")
    if smoking == 1:
        observations.append("The smoking input was set to Yes.")
    if exercise < 3:
        observations.append("The prototype's fewer-than-3-exercise-days demonstration rule was triggered.")
    if not observations:
        observations.append("None of the prototype's fixed demonstration rules were triggered.")
    return observations
