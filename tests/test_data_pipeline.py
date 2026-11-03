"""Mechanics-only tests; fixtures below are not clinical data."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_loader import csv_inventory, dataset_quality, load_csv, validate_dataframe
from src.eda import category_distribution, correlation_heatmap, feature_distribution, numeric_boxplots
from src.preprocessing import canonicalize_health_columns, outlier_report, prepare_health_dataset


class DataLoaderTests(unittest.TestCase):
    def test_load_csv_and_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.csv"
            path.write_text("value,group\n1,a\n1,a\n,b\n", encoding="utf-8")
            frame = load_csv(path)
        quality = dataset_quality(frame)
        self.assertEqual(quality["rows"], 3)
        self.assertEqual(quality["duplicate_rows"], 1)
        self.assertEqual(quality["missing_by_column"]["value"], 1)

    def test_rejects_duplicate_column_names(self):
        with self.assertRaisesRegex(ValueError, "Duplicate column"):
            validate_dataframe(pd.DataFrame([[1, 2]], columns=["x", "x"]))

    def test_loads_and_removes_sequential_saved_index_without_losing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indexed.csv"
            path.write_text("Unnamed: 0,value\n0,alpha\n1,beta\n", encoding="utf-8")
            frame = load_csv(path)
        self.assertEqual(frame.columns.tolist(), ["value"])
        self.assertEqual(frame["value"].tolist(), ["alpha", "beta"])

    def test_removes_only_genuinely_empty_unnamed_column(self):
        frame = validate_dataframe(pd.DataFrame({"Unnamed: 0": [None, None], "value": [1, 2]}))
        self.assertEqual(frame.columns.tolist(), ["value"])
        with self.assertRaisesRegex(ValueError, "not an empty or sequential"):
            validate_dataframe(pd.DataFrame({"Unnamed: 0": [7, 9], "value": [1, 2]}))
        with self.assertRaisesRegex(ValueError, "not an empty or sequential"):
            validate_dataframe(pd.DataFrame({"Unnamed: 0": [0, None, 2], "value": [1, 2, 3]}))

    def test_loader_rejects_duplicate_raw_headers_and_malformed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.csv"
            duplicate.write_text("value,value\n1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate column"):
                load_csv(duplicate)
            malformed = Path(directory) / "malformed.csv"
            malformed.write_text('value,other\n"unterminated,2\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parse CSV"):
                load_csv(malformed)

    def test_inventory_skips_archived_backups_and_generated_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "processed/data.csv", "backups/reports/correlations.csv",
                "reports/summary.csv", "raw/source.csv",
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("value\n1\n", encoding="utf-8")
            self.assertEqual(
                {path.relative_to(root).as_posix() for path in csv_inventory(root)},
                {"processed/data.csv", "raw/source.csv"},
            )


class PreprocessingTests(unittest.TestCase):
    def test_normalizes_supported_feature_aliases_and_preserves_duplicate_rows(self):
        # Small artificial fixture only; it is not a health dataset or model-training source.
        frame = pd.DataFrame(
            {
                "Age": [30, 30, 45],
                "BMI": [22.0, 22.0, None],
                "Systolic Blood Pressure": [115, 115, 135],
                "Glucose Level": [95, 95, 120],
                "Smoking Status": ["No", "No", "Yes"],
                "Exercise Days Per Week": [4, 4, 2],
                "risk": ["Low", "Low", "Medium"],
            }
        )
        x, y = prepare_health_dataset(frame, "risk")
        self.assertEqual(list(x.columns), ["age", "bmi", "blood_pressure", "glucose", "smoking", "exercise"])
        self.assertEqual(len(x), 3)
        self.assertEqual(y.tolist(), ["Low", "Low", "Medium"])
        self.assertEqual(x["smoking"].tolist(), [0.0, 0.0, 1.0])
        self.assertTrue(pd.isna(x.iloc[2]["bmi"]))

    def test_reports_but_does_not_remove_iqr_outliers(self):
        frame = pd.DataFrame({"record_id": [1, 2, 3, 4, 999], "measurement": [1, 1, 1, 1, 50]})
        report = outlier_report(frame)
        self.assertEqual(report["feature"].tolist(), ["measurement"])
        self.assertEqual(int(report.loc[0, "potential_outliers"]), 1)
        self.assertEqual(len(frame), 5)

    def test_requires_all_supported_features(self):
        with self.assertRaisesRegex(ValueError, "missing required model features"):
            prepare_health_dataset(pd.DataFrame({"age": [20], "risk": ["A"]}), "risk")

    def test_rejects_a_feature_with_no_observed_values(self):
        frame = pd.DataFrame(
            {
                "age": [20, 30],
                "bmi": [None, None],
                "blood_pressure": [110, 120],
                "glucose": [90, 100],
                "smoking": [0, 1],
                "exercise": [4, 3],
                "risk": ["Low Risk", "High Risk"],
            }
        )
        with self.assertRaisesRegex(ValueError, "cannot be imputed safely"):
            prepare_health_dataset(frame, "risk")

    def test_preserves_partial_missing_categorical_values_for_pipeline_imputation(self):
        frame = pd.DataFrame(
            {
                "age": [20, 30],
                "bmi": [20.0, 25.0],
                "blood_pressure": [110, 120],
                "glucose": [90, 100],
                "smoking": ["no", None],
                "exercise": [4, 3],
                "risk": ["Low Risk", "High Risk"],
            }
        )
        x, _ = prepare_health_dataset(frame, "risk")
        self.assertTrue(pd.isna(x.iloc[1]["smoking"]))

    def test_rejects_using_a_model_feature_as_the_target(self):
        frame = pd.DataFrame(
            {
                "age": [20, 30], "bmi": [20, 25], "blood_pressure": [110, 120],
                "glucose": [90, 100], "smoking": [0, 1], "exercise": [4, 3],
            }
        )
        with self.assertRaisesRegex(ValueError, "separate outcome/target"):
            prepare_health_dataset(frame, "age")

    def test_preserves_duplicate_records_during_model_preparation(self):
        row = {"age": 20, "bmi": 20, "blood_pressure": 110, "glucose": 90,
               "smoking": 0, "exercise": 4, "risk": "Low"}
        frame = pd.DataFrame([row, row, {**row, "age": 30, "risk": "High"}])
        x, y = prepare_health_dataset(frame, "risk")
        self.assertEqual(len(x), 3)
        self.assertEqual(len(y), 3)


class EdaTests(unittest.TestCase):
    def test_chart_builders_return_or_skip_figures_as_expected(self):
        frame = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6], "group": ["x", "y", "x"]})
        figures = [feature_distribution(frame, "a"), numeric_boxplots(frame), correlation_heatmap(frame), category_distribution(frame, "group")]
        self.assertTrue(all(figure is not None for figure in figures))
        import matplotlib.pyplot as plt

        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
