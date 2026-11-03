"""Mechanics-only tests using clearly artificial test rows, not clinical data."""

import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.model_evaluation import classification_metrics, confusion_table
from src.model_training import evaluate_models, fit_full_model, save_model_artifact
from model import predict_health_risk
from src.explainability import local_probability_effects
from src.preprocessing import HEALTH_FEATURES


def artificial_fixture():
    rows = []
    labels = []
    for class_id in range(3):
        for offset in range(12):
            rows.append(
                {
                    "age": 20 + 15 * class_id + offset,
                    "bmi": 20 + 3 * class_id + (offset % 4) * 0.1,
                    "blood_pressure": 110 + 15 * class_id + offset,
                    "glucose": 85 + 25 * class_id + offset,
                    "smoking": offset % 2,
                    "exercise": 6 - class_id - (offset % 2),
                }
            )
            labels.append(["Low Risk", "Medium Risk", "High Risk"][class_id])
    return pd.DataFrame(rows, columns=HEALTH_FEATURES), pd.Series(labels, name="risk")


class ModelEvaluationTests(unittest.TestCase):
    def test_metrics_and_confusion_labels(self):
        metrics = classification_metrics(["a", "b", "b"], ["a", "a", "b"])
        self.assertEqual(metrics["accuracy"], 2 / 3)
        matrix = confusion_table(["a", "b"], ["a", "a"], labels=["a", "b"])
        self.assertEqual(matrix.loc["a", "a"], 1)
        self.assertEqual(matrix.loc["b", "a"], 1)

    def test_compares_models_selects_using_reported_metrics_and_saves_pipeline(self):
        features, target = artificial_fixture()
        result = evaluate_models(features, target, max_cv_folds=3)
        self.assertEqual(set(result.comparison["model"]), {"Logistic Regression", "Decision Tree", "Random Forest"})
        self.assertIn("accuracy", result.comparison.columns)
        self.assertIn("precision_weighted", result.comparison.columns)
        self.assertIn("recall_weighted", result.comparison.columns)
        self.assertIn("f1_weighted", result.comparison.columns)
        self.assertEqual(result.cv_folds, 3)
        self.assertEqual(int(result.confusion.to_numpy().sum()), len(result.y_test))
        self.assertIn("weighted F1", result.selection_basis)

        full_model = fit_full_model(features, target, result.best_model_name)
        with tempfile.TemporaryDirectory() as directory:
            artifact = save_model_artifact(
                full_model,
                Path(directory) / "model.joblib",
                target_name="risk",
                training_rows=len(features),
                model_name=result.best_model_name,
                evaluation=result,
                training_features=features,
            )
            loaded = joblib.load(artifact)
        self.assertEqual(loaded["metadata"]["training_rows"], len(features))
        self.assertEqual(loaded["metadata"]["feature_names"], HEALTH_FEATURES)
        self.assertEqual(len(loaded["pipeline"].predict(features.head(2))), 2)
        artifact = {**loaded, "source": "trained"}
        prediction = predict_health_risk(artifact, features.iloc[0].to_dict())
        self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1.0, places=5)
        local = local_probability_effects(artifact, features.iloc[0].to_dict(), prediction["risk_level"])
        self.assertEqual(len(local), len(HEALTH_FEATURES))

    def test_legacy_random_forest_prediction_respects_estimator_class_order(self):
        class StubClassifier:
            classes_ = [2, 0, 1]

            def predict(self, values):
                return [2]

            def predict_proba(self, values):
                return [[0.2, 0.3, 0.5]]

        patient = {name: 1 for name in HEALTH_FEATURES}
        result = predict_health_risk({"pipeline": StubClassifier(), "source": "legacy_demo"}, patient)
        self.assertEqual(result["risk_level"], "High Risk")
        self.assertEqual(result["probabilities"]["High Risk"], 0.2)
        self.assertEqual(result["probabilities"]["Low Risk"], 0.3)

    def test_rejects_classes_that_cannot_be_split(self):
        features, target = artificial_fixture()
        target.iloc[-1] = "rare"
        with self.assertRaisesRegex(ValueError, "at least two labeled rows"):
            evaluate_models(features, target)


if __name__ == "__main__":
    unittest.main()
