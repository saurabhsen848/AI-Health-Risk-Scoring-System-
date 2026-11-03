"""Reproducible preparation and exploratory analysis for official NHANES data.

This module only prepares survey records. It never synthesizes observations or
fits a prediction model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CYCLE = "NHANES August 2021-August 2023"
SUFFIX = "L"
RAW_BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles"
FILES = {
    "DEMO_L": "DEMO_L.xpt",
    "BMX_L": "BMX_L.xpt",
    "BPXO_L": "BPXO_L.xpt",
    "GLU_L": "GLU_L.xpt",
    "SMQ_L": "SMQ_L.xpt",
    "PAQ_L": "PAQ_L.xpt",
    "BPQ_L": "BPQ_L.xpt",
}
TARGET = "hypertension_history"
FEATURE_COLUMNS = [
    "age",
    "bmi",
    "systolic_bp",
    "fasting_glucose",
    "current_smoker",
    "physical_activity_sessions_per_week",
]
INITIAL_MODEL_FEATURE_COLUMNS = [
    "age",
    "bmi",
    "fasting_glucose",
    "current_smoker",
    "physical_activity_sessions_per_week",
]
RAW_FEATURE_COLUMNS = {
    "age": ["RIDAGEYR"],
    "bmi": ["BMXBMI"],
    "systolic_bp": ["BPXOSY1", "BPXOSY2", "BPXOSY3"],
    "fasting_glucose": ["LBXGLU"],
    "current_smoker": ["SMQ020", "SMQ040"],
    "physical_activity_sessions_per_week": [
        "PAD790Q", "PAD790U", "PAD810Q", "PAD810U"
    ],
}
WEIGHT_COLUMNS = ["WTSAF2YR", "SDMVSTRA", "SDMVPSU"]
# pandas' XPORT reader exposes SAS numeric missing values in some fields as this
# tiny positive floating-point value instead of NaN. It is far outside every
# documented NHANES feature domain and must be normalized before filtering.
SAS_XPORT_NUMERIC_MISSING = 5.397605346934028e-79


def _load_xpt(path: Path) -> pd.DataFrame:
    """Read one unchanged NHANES SAS transport file."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing NHANES source file: {path}")
    frame = pd.read_sas(path, format="xport", encoding="utf-8")
    frame.columns = [str(column).upper() for column in frame.columns]
    zero_valid = {
        "RIDAGEYR", "WTMEC2YR", "WTSAF2YR", "PAD790Q", "PAD810Q"
    }
    for column in frame.select_dtypes(include="number").columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        sas_special = np.isfinite(numeric) & np.isclose(
            numeric, SAS_XPORT_NUMERIC_MISSING, rtol=1e-6, atol=0.0
        )
        # In NHANES XPT files this representation can also be a legitimate 0
        # (e.g. age 0, zero reported activity, or a zero subsample weight).
        # Preserve zeros in those fields and map it to NaN elsewhere.
        if column in zero_valid:
            numeric = numeric.mask(sas_special, 0.0)
        else:
            numeric = numeric.mask(sas_special)
        frame[column] = numeric
    if "SEQN" not in frame:
        raise ValueError(f"{path.name} has no NHANES respondent key SEQN")
    if frame["SEQN"].duplicated().any():
        raise ValueError(f"{path.name} has duplicate SEQN values; refusing many-to-many merge")
    return frame


def load_merge_sources(raw_dir: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load required XPT files and left-merge them by SEQN onto DEMO_L."""
    root = Path(raw_dir)
    frames = {key: _load_xpt(root / filename) for key, filename in FILES.items()}
    original_rows = {key: len(frame) for key, frame in frames.items()}
    merged = frames["DEMO_L"].copy()
    for key, frame in frames.items():
        if key == "DEMO_L":
            continue
        keep = ["SEQN"] + [column for column in frame.columns if column != "SEQN"]
        merged = merged.merge(frame[keep], on="SEQN", how="left", validate="one_to_one")
    return merged, original_rows


def _to_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _weekly_frequency(quantity: pd.Series, unit: pd.Series, maximum: float) -> pd.Series:
    """Convert documented activity frequency/unit pairs to sessions per week."""
    q = pd.to_numeric(quantity, errors="coerce")
    q = q.where(np.isfinite(q) & q.between(0, maximum, inclusive="both"))
    u = unit.astype("string").str.strip().str.upper()
    result = pd.Series(np.nan, index=quantity.index, dtype="float64")
    # A reported zero means no activity; it is a valid response even if no unit applies.
    zero = q.eq(0)
    result.loc[zero] = 0.0
    factors = {"D": 7.0, "W": 1.0, "M": 12.0 / 52.0, "Y": 1.0 / 52.0}
    for code, factor in factors.items():
        mask = q.notna() & q.gt(0) & u.eq(code)
        result.loc[mask] = q.loc[mask] * factor
    return result


def _activity_soft_edit_exceeded(quantity: pd.Series, unit: pd.Series) -> pd.Series:
    """Flag NHANES PAQ soft-edit thresholds without excluding or changing values."""
    q = pd.to_numeric(quantity, errors="coerce")
    u = unit.astype("string").str.strip().str.upper()
    soft_limits = {"D": 4, "W": 28, "M": 31, "Y": 365}
    limits = u.map(soft_limits)
    return q.notna() & limits.notna() & q.gt(limits)


def prepare_dataset(raw_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create complete-case feature/target rows and an auditable quality report."""
    merged, source_rows = load_merge_sources(raw_dir)
    original_n = len(merged)

    required = {
        "RIDAGEYR", "BMXBMI", "BPXOSY1", "BPXOSY2", "BPXOSY3", "LBXGLU",
        "WTSAF2YR", "SDMVSTRA", "SDMVPSU", "SMQ020", "SMQ040",
        "PAD790Q", "PAD790U", "PAD810Q", "PAD810U", "BPQ020",
    }
    missing_columns = sorted(required - set(merged.columns))
    if missing_columns:
        raise ValueError(f"NHANES files are missing required variables: {missing_columns}")

    merged = _to_numeric(
        merged,
        [
            "RIDAGEYR", "BMXBMI", "BPXOSY1", "BPXOSY2", "BPXOSY3", "LBXGLU",
            "WTSAF2YR", "SDMVSTRA", "SDMVPSU", "SMQ020", "SMQ040",
            "PAD790Q", "PAD810Q", "BPQ020",
        ],
    )
    counts: dict[str, Any] = {
        "cycle": CYCLE,
        "source_file_rows": source_rows,
        "duplicate_SEQN_rows_by_source": {key: 0 for key in FILES},
        "duplicate_key_validation": "All files validated unique on SEQN; duplicates abort the merge.",
        "original_rows_DEMO_L": original_n,
        "rows_after_SEQN_left_merges": len(merged),
        "rows_removed_by_merges": original_n - len(merged),
        "merge_key": "SEQN",
        "raw_missing_counts_after_merge": {
            col: int(merged[col].isna().sum())
            for col in sorted(required)
            if col in merged
        },
    }

    # The activity/smoking measures are adult questions. The model cohort is adults 18+.
    age_ok = merged["RIDAGEYR"].between(18, 80, inclusive="both")
    counts["rows_removed_age_ineligible_or_invalid"] = int((~age_ok).sum())
    stage = merged.loc[age_ok].copy()
    counts["rows_after_age_filter"] = len(stage)

    target_ok = stage["BPQ020"].isin([1, 2])
    counts["rows_removed_target_refused_unknown_or_missing"] = int((~target_ok).sum())
    stage = stage.loc[target_ok].copy()
    counts["rows_after_valid_target_filter"] = len(stage)
    stage[TARGET] = stage["BPQ020"].map({1: 1, 2: 0}).astype("int8")

    # Smoking skip logic: never smokers are non-current; follow-up applies to ever smokers.
    smoking = pd.Series(np.nan, index=stage.index, dtype="float64")
    smoking.loc[stage["SMQ020"].eq(2)] = 0.0
    ever = stage["SMQ020"].eq(1)
    smoking.loc[ever & stage["SMQ040"].isin([1, 2])] = 1.0
    smoking.loc[ever & stage["SMQ040"].eq(3)] = 0.0
    stage["current_smoker"] = smoking

    # Mean available NHANES oscillometric SBP readings. NHANES capture rules cap SBP at 300.
    bpx = stage[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].where(
        stage[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].gt(0)
        & stage[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].le(300)
    )
    stage["systolic_bp"] = bpx.mean(axis=1, skipna=True)

    stage["age"] = stage["RIDAGEYR"]
    stage["bmi"] = stage["BMXBMI"].where(
        np.isfinite(stage["BMXBMI"]) & stage["BMXBMI"].gt(0)
    )
    stage["fasting_glucose"] = stage["LBXGLU"].where(
        np.isfinite(stage["LBXGLU"]) & stage["LBXGLU"].gt(0)
    )

    moderate_week = _weekly_frequency(stage["PAD790Q"], stage["PAD790U"], maximum=180)
    vigorous_week = _weekly_frequency(stage["PAD810Q"], stage["PAD810U"], maximum=200)
    stage["physical_activity_sessions_per_week"] = moderate_week + vigorous_week
    moderate_soft_edit = _activity_soft_edit_exceeded(stage["PAD790Q"], stage["PAD790U"])
    vigorous_soft_edit = _activity_soft_edit_exceeded(stage["PAD810Q"], stage["PAD810U"])

    # A valid fasting-subset survey weight is required for use of fasting glucose.
    stage["WTSAF2YR"] = stage["WTSAF2YR"].where(stage["WTSAF2YR"].gt(0))
    feature_missing = stage[FEATURE_COLUMNS].isna()
    counts["feature_missing_counts_after_derivation_before_complete_case_filter"] = {
        column: int(feature_missing[column].sum()) for column in FEATURE_COLUMNS
    }
    counts["rows_removed_missing_or_invalid_required_feature_or_weight"] = int(
        feature_missing.any(axis=1).sum()
        + (~stage["WTSAF2YR"].notna() & ~feature_missing.any(axis=1)).sum()
    )

    complete = stage[FEATURE_COLUMNS + [TARGET, "WTSAF2YR", "SDMVSTRA", "SDMVPSU"]].copy()
    complete = complete.dropna(
        subset=FEATURE_COLUMNS + [TARGET, "WTSAF2YR", "SDMVSTRA", "SDMVPSU"]
    )
    # Preserve values/units and survey fields needed to audit or perform survey-aware evaluation.
    complete[TARGET] = complete[TARGET].astype("int8")
    complete["current_smoker"] = complete["current_smoker"].astype("int8")
    counts["final_usable_sample_count"] = len(complete)
    counts["rows_after_feature_and_weight_filter"] = len(complete)
    counts["rows_removed_total_from_DEMO_L"] = original_n - len(complete)
    counts["target_counts"] = {
        "positive_reported_ever_diagnosed_yes_1": int((complete[TARGET] == 1).sum()),
        "negative_reported_ever_diagnosed_no_0": int((complete[TARGET] == 0).sum()),
        "positive_percent": float((complete[TARGET] == 1).mean() * 100),
        "negative_percent": float((complete[TARGET] == 0).mean() * 100),
    }
    counts["final_feature_columns"] = FEATURE_COLUMNS
    counts["target_column"] = TARGET
    counts["final_columns"] = list(complete.columns)
    counts["missing_in_final_dataset"] = {
        column: int(complete[column].isna().sum()) for column in complete.columns
    }
    counts["outlier_policy"] = (
        "No statistical outliers removed. Invalid/nonfinite and nonpositive BMI/glucose are excluded; "
        "SBP readings must be positive and <=300 mmHg per NHANES capture rule."
    )
    counts["activity_definition"] = (
        "Sum of self-reported moderate and vigorous leisure-time activity frequencies, "
        "converted to sessions per week from PAD790Q/PAD790U and PAD810Q/PAD810U. "
        "This is session frequency, not unique days active."
    )
    counts["initial_model_feature_columns"] = INITIAL_MODEL_FEATURE_COLUMNS
    counts["systolic_bp_initial_model_policy"] = (
        "Retained as a documented contemporaneous measurement in this analysis dataset, "
        "but excluded from the proposed initial model features because the target is ever-reported "
        "prior hypertension history and the timing relative to diagnosis is unknown."
    )
    counts["physical_activity_soft_edit_review"] = {
        "official_thresholds": {
            "D": "more than 4 times per day",
            "W": "more than 28 times per week",
            "M": "more than 31 times per month",
            "Y": "more than 365 times per year",
        },
        "interpretation": (
            "NHANES describes these as CAPI soft edits asking the respondent to verify the value; "
            "they are review flags, not automatic invalid-value limits."
        ),
        "action": "Flag only. No rows or values are removed, capped, or changed by these checks.",
        "complete_case_rows_over_soft_edit_by_variable": {
            "PAD790Q_moderate": int(moderate_soft_edit.loc[complete.index].sum()),
            "PAD810Q_vigorous": int(vigorous_soft_edit.loc[complete.index].sum()),
        },
        "complete_case_rows_over_soft_edit_in_either_variable": int(
            (moderate_soft_edit | vigorous_soft_edit).loc[complete.index].sum()
        ),
        "complete_case_rows_total_sessions_over_56_per_week": int(
            complete["physical_activity_sessions_per_week"].gt(56).sum()
        ),
        "combined_56_sessions_note": (
            "56 is a review reference obtained by combining 4 per day for each of two activity "
            "intensities over 7 days. It is not an official NHANES limit for the summed feature."
        ),
    }
    return complete.reset_index(drop=True), counts


def run_eda(merged: pd.DataFrame, processed: pd.DataFrame, report: dict[str, Any], output_dir: str | Path) -> None:
    """Write descriptive missingness, distributions, associations, and target plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})

    raw_cols = [col for group in RAW_FEATURE_COLUMNS.values() for col in group]
    raw_cols = list(dict.fromkeys(raw_cols + ["BPQ020"]))
    missing = pd.DataFrame(
        {
            "column": raw_cols,
            "missing_count": [int(merged[col].isna().sum()) for col in raw_cols],
            "missing_percent": [float(merged[col].isna().mean() * 100) for col in raw_cols],
        }
    )
    missing.to_csv(out / "missing_values.csv", index=False)
    processed.describe(include="all").transpose().to_csv(out / "processed_summary.csv")
    (out / "data_quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    for feature in FEATURE_COLUMNS:
        values = processed[feature].dropna()
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = min(35, max(8, int(np.sqrt(max(1, len(values))))))
        ax.hist(values, bins=bins, color="#287c8e", edgecolor="white")
        ax.set(title=f"Distribution: {feature.replace('_', ' ')}", xlabel=feature, ylabel="Respondents")
        fig.tight_layout()
        fig.savefig(out / f"distribution_{feature}.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    processed[FEATURE_COLUMNS].plot(kind="box", ax=ax, rot=35, grid=True)
    ax.set_title("Feature boxplots (descriptive; points are not automatically discarded)")
    fig.tight_layout()
    fig.savefig(out / "feature_boxplots.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    processed[TARGET].value_counts().reindex([0, 1], fill_value=0).plot(
        kind="bar", ax=ax, color=["#287c8e", "#d17a22"]
    )
    ax.set(title="BPQ020 reported hypertension history", xlabel="0 = No; 1 = Yes", ylabel="Respondents")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out / "target_distribution.png")
    plt.close(fig)

    numeric_cols = FEATURE_COLUMNS + [TARGET]
    corr = processed[numeric_cols].corr(numeric_only=True)
    corr.to_csv(out / "correlations.csv")
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.set_title("Feature/target Pearson correlations (descriptive, not causal)")
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(out / "correlation_heatmap.png")
    plt.close(fig)

    outlier_rows = []
    for feature in FEATURE_COLUMNS:
        values = processed[feature].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
        outlier_rows.append({
            "feature": feature,
            "non_missing": int(values.size),
            "potential_iqr_outliers": int(((values < lower) | (values > upper)).sum()),
            "lower_iqr_bound_descriptive_only": float(lower),
            "upper_iqr_bound_descriptive_only": float(upper),
        })
    pd.DataFrame(outlier_rows).to_csv(out / "potential_outliers_iqr.csv", index=False)

    for feature in ["age", "bmi", "systolic_bp", "fasting_glucose", "physical_activity_sessions_per_week"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        groups = [processed.loc[processed[TARGET] == label, feature].dropna() for label in (0, 1)]
        ax.boxplot(groups, tick_labels=["No", "Yes"], showfliers=True)
        ax.set(title=f"{feature.replace('_', ' ')} by reported hypertension history", xlabel="BPQ020", ylabel=feature)
        fig.tight_layout()
        fig.savefig(out / f"relationship_{feature}_by_target.png")
        plt.close(fig)

    smoking_counts = pd.crosstab(processed["current_smoker"], processed[TARGET])
    smoking_counts.to_csv(out / "relationship_smoking_by_target.csv")
    activity_summary = processed.groupby(TARGET)["physical_activity_sessions_per_week"].agg(
        ["count", "mean", "median", "std"]
    )
    activity_summary.to_csv(out / "relationship_activity_by_target.csv")


def write_config(path: str | Path) -> None:
    """Save a machine-readable record of source/target/features/rules."""
    config = {
        "source": "U.S. CDC/NCHS National Health and Nutrition Examination Survey",
        "cycle": CYCLE,
        "source_base_url": RAW_BASE_URL,
        "files": FILES,
        "merge_key": "SEQN",
        "cohort": "Age 18 years and older; positive fasting subsample weight required.",
        "target": {
            "source": "BPQ_L.BPQ020",
            "meaning": "Reported ever told by a doctor/health professional that respondent had hypertension",
            "positive_raw_code": 1,
            "negative_raw_code": 2,
            "excluded_raw_codes": [7, 9, "missing"],
            "processed_name": TARGET,
            "positive_processed_value": 1,
            "negative_processed_value": 0,
        },
        "features": {
            "age": "DEMO_L.RIDAGEYR",
            "bmi": "BMX_L.BMXBMI",
            "systolic_bp": ["BPXO_L.BPXOSY1", "BPXO_L.BPXOSY2", "BPXO_L.BPXOSY3"],
            "fasting_glucose": "GLU_L.LBXGLU",
            "current_smoker": ["SMQ_L.SMQ020", "SMQ_L.SMQ040"],
            "physical_activity_sessions_per_week": [
                "PAQ_L.PAD790Q", "PAQ_L.PAD790U", "PAQ_L.PAD810Q", "PAQ_L.PAD810U"
            ],
        },
        "initial_model_features": INITIAL_MODEL_FEATURE_COLUMNS,
        "excluded_from_initial_model_features": {
            "systolic_bp": (
                "Retained for descriptive analysis only; it is measured at the survey while "
                "BPQ020 reports ever-diagnosed history, with unknown timing relative to diagnosis."
            )
        },
        "missing_values": "No imputation in development dataset; remove rows missing any required feature, target, survey design field, or positive WTSAF2YR. Report source missingness before filtering.",
        "invalid_values": {
            "age": "Require age 18 through 80; NHANES public file top-codes age at 80.",
            "bmi": "Require finite BMI > 0; preserve valid extremes and report IQR outliers rather than deleting statistical outliers.",
            "systolic_bp": "Use mean of available readings that are >0 and <=300 mmHg; NHANES capture rule says SBP cannot exceed 300.",
            "fasting_glucose": "Require finite fasting glucose >0 and positive WTSAF2YR.",
            "target": "Only raw codes 1 and 2 are labeled; 7/9/missing excluded.",
        },
        "survey_design": {
            "weight": "GLU_L.WTSAF2YR",
            "stratum": "DEMO_L.SDMVSTRA",
            "psu": "DEMO_L.SDMVPSU",
        },
        "model_training": "Not performed. No classifier or prediction model is fit by this workflow.",
        "interpretation": "BPQ020 is reported ever-diagnosed hypertension history, not future hypertension prediction, not medical diagnosis, and not clinically validated.",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(config, indent=2), encoding="utf-8")
