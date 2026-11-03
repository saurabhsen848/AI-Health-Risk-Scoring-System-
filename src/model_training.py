"""Supervised model comparison and persistence for labeled health-risk CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .model_evaluation import classification_metrics, comparison_table, confusion_table
from .preprocessing import HEALTH_FEATURES, build_preprocessor


@dataclass
class ModelEvaluation:
    comparison: pd.DataFrame
    estimators: dict[str, Pipeline]
    best_model_name: str
    labels: list[object]
    y_test: pd.Series
    y_pred: np.ndarray
    confusion: pd.DataFrame
    selection_basis: str
    cv_folds: int


def _classifier_candidates(random_state: int = 42) -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=random_state, min_samples_leaf=2),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def evaluate_models(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_fraction: float = 0.25,
    random_state: int = 42,
    max_cv_folds: int = 5,
) -> ModelEvaluation:
    """Compare models on a stratified holdout and CV the training partition where feasible."""
    if list(features.columns) != HEALTH_FEATURES:
        raise ValueError(f"Features must be ordered as: {', '.join(HEALTH_FEATURES)}")
    if len(features) != len(target):
        raise ValueError("Features and target must contain the same number of rows.")
    counts = target.value_counts(dropna=True)
    if counts.empty or counts.min() < 2:
        raise ValueError("Each target class needs at least two labeled rows for a stratified train/test split.")
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5.")

    classes = sorted(target.dropna().unique().tolist(), key=str)
    test_rows = max(int(np.ceil(len(target) * test_fraction)), len(classes))
    if len(target) - test_rows < len(classes):
        raise ValueError("Not enough rows for a stratified holdout. Add labeled examples for each class.")
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_rows,
        random_state=random_state,
        stratify=target,
    )

    min_train_class = int(y_train.value_counts().min())
    cv_folds = min(max_cv_folds, min_train_class)
    if cv_folds < 3:
        cv_folds = 0
    candidates = _classifier_candidates(random_state)
    estimators: dict[str, Pipeline] = {}
    rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}

    for name, classifier in candidates.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("classifier", classifier)])
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        predictions[name] = prediction
        metrics = classification_metrics(y_test, prediction, labels=classes)
        cv_mean = np.nan
        if cv_folds:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=1)
            cv_mean = float(scores.mean())
        rows.append(
            {
                "model": name,
                **metrics,
                "cv_f1_weighted_mean": cv_mean,
                "holdout_rows": len(y_test),
                "training_rows": len(y_train),
            }
        )
        estimators[name] = pipeline

    results = comparison_table(rows)
    if cv_folds:
        selected = results.sort_values(
            ["cv_f1_weighted_mean", "f1_weighted", "recall_weighted"], ascending=False, na_position="last"
        ).iloc[0]["model"]
        basis = f"mean {cv_folds}-fold weighted F1 on the training partition; holdout metrics are reported separately"
    else:
        selected = results.sort_values(["f1_weighted", "recall_weighted", "accuracy"], ascending=False).iloc[0]["model"]
        basis = "stratified holdout weighted F1, with weighted recall and accuracy as tie-breakers; CV skipped because class counts were too small"

    selected = str(selected)
    prediction = predictions[selected]
    return ModelEvaluation(
        comparison=results,
        estimators=estimators,
        best_model_name=selected,
        labels=classes,
        y_test=y_test,
        y_pred=prediction,
        confusion=confusion_table(y_test, prediction, labels=classes),
        selection_basis=basis,
        cv_folds=cv_folds,
    )


def fit_full_model(features: pd.DataFrame, target: pd.Series, model_name: str, random_state: int = 42) -> Pipeline:
    """Fit a selected classifier on all supplied labeled data for later use."""
    candidates = _classifier_candidates(random_state)
    if model_name not in candidates:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {', '.join(candidates)}")
    pipeline = Pipeline([("preprocessor", build_preprocessor()), ("classifier", candidates[model_name])])
    return pipeline.fit(features, target)


def save_model_artifact(
    model: Pipeline,
    path: str | Path,
    *,
    target_name: str,
    training_rows: int,
    model_name: str,
    evaluation: ModelEvaluation,
    training_features: pd.DataFrame | None = None,
) -> Path:
    """Save a fitted pipeline together with transparent provenance/evaluation metadata."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": model,
        "metadata": {
            "model_name": model_name,
            "target_name": target_name,
            "training_rows": int(training_rows),
            "feature_names": HEALTH_FEATURES.copy(),
            "class_labels": [str(label) for label in evaluation.labels],
            "feature_reference_values": (
                {
                    **training_features.median(numeric_only=True).to_dict(),
                    **({"smoking": training_features["smoking"].mode(dropna=True).iloc[0]} if training_features is not None and "smoking" in training_features and not training_features["smoking"].dropna().empty else {}),
                }
                if training_features is not None
                else {}
            ),
            "selection_basis": evaluation.selection_basis,
            "holdout_metrics": evaluation.comparison.set_index("model").loc[model_name].to_dict(),
            "evaluation_holdout_rows": int(len(evaluation.y_test)),
            "cross_validation_folds": int(evaluation.cv_folds),
        },
    }
    joblib.dump(payload, artifact_path)
    return artifact_path
