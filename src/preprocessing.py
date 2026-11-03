"""Dataset preparation and sklearn preprocessing utilities."""

from __future__ import annotations

import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import validate_dataframe


HEALTH_FEATURES = ["age", "bmi", "blood_pressure", "glucose", "smoking", "exercise"]
NUMERIC_FEATURES = ["age", "bmi", "blood_pressure", "glucose", "exercise"]
CATEGORICAL_FEATURES = ["smoking"]
IDENTIFIER_COLUMN = re.compile(r"(^id$|[_\s]id$|^id[_\s]|^index$|^serial(?:[_\s]number)?$|^no$)", re.IGNORECASE)
FEATURE_ALIASES = {
    "age": {"age", "patient_age"},
    "bmi": {"bmi", "body_mass_index"},
    "blood_pressure": {"blood_pressure", "systolic_blood_pressure", "systolic_bp", "sbp"},
    "glucose": {"glucose", "glucose_level", "blood_glucose"},
    "smoking": {"smoking", "smoker", "smoking_status"},
    "exercise": {"exercise", "exercise_days_per_week", "physical_activity_days"},
}


def canonicalize_health_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common column spellings to the six supported assessment features."""
    checked = validate_dataframe(frame)
    lookup = {str(column).strip().lower().replace(" ", "_"): column for column in checked.columns}
    rename: dict[object, str] = {}
    for canonical, aliases in FEATURE_ALIASES.items():
        match = next((lookup[alias] for alias in aliases if alias in lookup), None)
        if match is not None:
            rename[match] = canonical
    canonical = checked.rename(columns=rename)
    if canonical.columns.has_duplicates:
        raise ValueError("Multiple input columns map to the same health feature.")
    return canonical


def prepare_health_dataset(frame: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare actual labeled health rows; missing features are imputed in the model pipeline."""
    canonical = canonicalize_health_columns(frame)
    missing = [feature for feature in HEALTH_FEATURES if feature not in canonical.columns]
    if missing:
        raise ValueError(f"Dataset is missing required model features: {', '.join(missing)}")
    if target_column not in canonical.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the dataset.")
    if target_column in HEALTH_FEATURES:
        raise ValueError("Select a separate outcome/target column; model input features cannot be used as the target.")

    data = canonical[HEALTH_FEATURES + [target_column]].copy()
    data = data.dropna(subset=[target_column])
    if data.empty:
        raise ValueError("No rows with a known target remain after removing missing targets.")

    smoking = data["smoking"].astype("string").str.strip().str.lower()
    smoking_mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    non_missing = smoking.notna()
    mapped = smoking.map(smoking_mapping)
    unknown = sorted(smoking[non_missing & mapped.isna()].dropna().unique().tolist())
    if unknown:
        raise ValueError(f"Unsupported smoking values: {unknown}. Use yes/no, true/false, or 1/0.")
    data["smoking"] = mapped.astype("float64")

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    entirely_missing = [column for column in HEALTH_FEATURES if data[column].isna().all()]
    if entirely_missing:
        raise ValueError(
            "These model features have no usable values and cannot be imputed safely: "
            + ", ".join(entirely_missing)
        )
    y = data.pop(target_column)
    if y.nunique() < 2:
        raise ValueError("At least two target classes are needed for classification.")
    return data[HEALTH_FEATURES], y


def validate_risk_labels(target: pd.Series) -> None:
    """Ensure a supervised target represents the supported risk categories."""
    allowed = {"0", "1", "2", "low", "low risk", "medium", "medium risk", "high", "high risk"}
    values = set()
    for value in target.dropna().unique():
        if isinstance(value, (int, float)) and float(value).is_integer():
            values.add(str(int(value)))
        else:
            values.add(str(value).strip().lower())
    unsupported = sorted(values - allowed)
    if unsupported:
        raise ValueError(
            "This risk-scoring pipeline accepts only Low/Medium/High risk labels "
            f"(or 0/1/2); unsupported labels found: {unsupported}"
        )


def build_preprocessor() -> ColumnTransformer:
    """Impute missing feature values and scale continuous inputs for linear models."""
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def outlier_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Count potential IQR outliers by numeric column; never remove rows."""
    numeric_columns = [
        column
        for column in frame.select_dtypes(include="number").columns
        if not IDENTIFIER_COLUMN.search(str(column))
    ]
    numeric = frame[numeric_columns]
    rows: list[dict[str, object]] = []
    for column in numeric.columns:
        values = numeric[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
        rows.append(
            {
                "feature": str(column),
                "non_missing": int(values.size),
                "potential_outliers": int(((values < lower) | (values > upper)).sum()),
                "lower_iqr_bound": float(lower),
                "upper_iqr_bound": float(upper),
            }
        )
    return pd.DataFrame(rows)
