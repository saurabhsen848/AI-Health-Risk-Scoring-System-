"""Train and compare risk classifiers from an owner-provided labeled CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_csv
from src.model_training import evaluate_models, fit_full_model, save_model_artifact
from src.preprocessing import prepare_health_dataset, validate_risk_labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to an actual labeled CSV; no sample data is generated.")
    parser.add_argument("--target", required=True, help="Name of the risk-category label column.")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "models" / "health_risk_pipeline.joblib"),
        help="Where to save the selected full-data pipeline.",
    )
    args = parser.parse_args()

    raw = load_csv(args.data)
    features, target = prepare_health_dataset(raw, args.target)
    validate_risk_labels(target)
    result = evaluate_models(features, target)
    print("Holdout and cross-validation results:")
    print(result.comparison.to_string(index=False))
    print(f"Selected: {result.best_model_name}")
    print(f"Selection basis: {result.selection_basis}")
    print("Selected model holdout confusion matrix:")
    print(result.confusion.to_string())

    fitted = fit_full_model(features, target, result.best_model_name)
    output = save_model_artifact(
        fitted,
        args.output,
        target_name=args.target,
        training_rows=len(features),
        model_name=result.best_model_name,
        evaluation=result,
        training_features=features,
    )
    print(f"Full-data pipeline saved to: {output}")
    print("Metrics describe only the stratified holdout; no clinical validation is implied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
