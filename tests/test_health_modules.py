"""Mechanics-only module tests; all frames are small artificial fixtures."""

import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.agents.orchestrator import AssessmentOrchestrator
from src.biomarker_analysis import biomarker_summary, compare_reference_ranges, patient_biomarker_summary
from src.pdf_biomarker import PDFExtractionError, extract_cbc_pdf, parse_cbc_text
from src.disease_risk import disease_label_inventory
from src.explainability import global_feature_importance, local_probability_effects
from src.preprocessing import HEALTH_FEATURES
from src.regional_analysis import aggregate_regional_trends, regional_summary
from utils.explanation import get_risk_explanation


class StubLegacyModel:
    classes_ = [0, 1, 2]
    feature_importances_ = [0.28, 0.19, 0.17, 0.14, 0.12, 0.10]

    def predict(self, values):
        return [1]

    def predict_proba(self, values):
        return [[0.2, 0.6, 0.2]]


def patient_values():
    return dict(zip(HEALTH_FEATURES, [45, 24, 120, 100, 0, 4]))


class BiomarkerTests(unittest.TestCase):
    def test_summary_ignores_obvious_identifier_and_does_not_infer_status(self):
        frame = pd.DataFrame({"patient_id": [1, 2], "marker_x": [4.0, None], "note": ["a", "b"]})
        summary = biomarker_summary(frame)
        self.assertEqual(summary["biomarker"].tolist(), ["marker_x"])
        self.assertEqual(int(summary.iloc[0]["missing"]), 1)
        self.assertNotIn("status", summary.columns)
        values = patient_biomarker_summary(frame, 0)
        self.assertEqual(values.iloc[0]["value"], 4.0)

    def test_reference_comparison_requires_explicit_bounds(self):
        self.assertTrue(compare_reference_ranges({"marker_x": 10}, {}).empty)
        compared = compare_reference_ranges({"marker_x": 10}, {"marker_x": (2, 8)})
        self.assertEqual(compared.iloc[0]["status"], "Above supplied range")

    def test_parses_cbc_values_units_ranges_and_report_relative_status(self):
        text = """Hemoglobin 13.5 g/dL 12.0 - 16.0
WBC: 12.0 10^3/uL (4.0-11.0)
RBC 3.5 10^6/uL 4.0 - 5.5
Platelets 200 x10^3/uL 150 - 400
Hematocrit 34 % (36 - 46)"""
        result = parse_cbc_text(text).set_index("Biomarker")
        self.assertEqual(
            result.reset_index().columns.tolist(),
            ["Biomarker", "Result", "Unit", "Reference Range", "Status"],
        )
        self.assertEqual(result.loc["Hemoglobin", "Unit"], "g/dL")
        self.assertEqual(result.loc["Hemoglobin", "Reference Range"], "12 - 16")
        self.assertEqual(result.loc["Hemoglobin", "Status"], "Within range")
        self.assertEqual(result.loc["WBC", "Status"], "High")
        self.assertEqual(result.loc["WBC", "Unit"], "10^3/uL")
        self.assertEqual(result.loc["RBC", "Status"], "Low")
        self.assertEqual(result.loc["Platelets", "Status"], "Within range")
        self.assertEqual(result.loc["Hematocrit", "Status"], "Low")

    def test_missing_results_and_missing_ranges_are_never_inferred(self):
        result = parse_cbc_text("Hemoglobin: -- g/dL\nWBC 5.1 10^3/uL")
        by_name = result.set_index("Biomarker")
        self.assertEqual(by_name.loc["Hemoglobin", "Result"], "Result unavailable")
        self.assertEqual(by_name.loc["Hemoglobin", "Status"], "Result unavailable")
        self.assertEqual(by_name.loc["WBC", "Reference Range"], "Range unavailable")
        self.assertEqual(by_name.loc["WBC", "Status"], "Range unavailable")

    def test_parses_unit_before_result_when_report_uses_that_layout(self):
        result = parse_cbc_text("RBC (10^6/uL) 4.5 (4.0 - 5.5)").iloc[0]
        self.assertEqual(result["Result"], "4.5")
        self.assertEqual(result["Unit"], "10^6/uL")
        self.assertEqual(result["Reference Range"], "4 - 5.5")

    def test_extracts_text_from_a_real_selectable_text_pdf(self):
        writer = PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        font = writer._add_object(
            DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        content = DecodedStreamObject()
        content.set_data(b"BT /F1 12 Tf 72 720 Td (Hemoglobin 13.5 g/dL 12 - 16) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(content)
        pdf_bytes = BytesIO()
        writer.write(pdf_bytes)

        result = extract_cbc_pdf(pdf_bytes.getvalue())
        self.assertEqual(result.iloc[0]["Biomarker"], "Hemoglobin")
        self.assertEqual(result.iloc[0]["Result"], "13.5")
        self.assertEqual(result.iloc[0]["Unit"], "g/dL")
        self.assertEqual(result.iloc[0]["Reference Range"], "12 - 16")

    def test_invalid_and_scanned_pdfs_raise_clear_extraction_errors(self):
        with self.assertRaisesRegex(PDFExtractionError, "valid PDF"):
            extract_cbc_pdf(b"not a PDF")
        with patch(
            "src.pdf_biomarker.PdfReader",
            return_value=SimpleNamespace(is_encrypted=False, pages=[SimpleNamespace(extract_text=lambda: None)]),
        ):
            with self.assertRaisesRegex(PDFExtractionError, "scanned PDF"):
                extract_cbc_pdf(b"%PDF scanned fixture")


class DiseaseAndRegionalTests(unittest.TestCase):
    def test_disease_inventory_only_reports_present_labeled_columns(self):
        empty = disease_label_inventory({"supplied": pd.DataFrame({"age": [1, 2]})})
        self.assertTrue(empty.empty)
        present = disease_label_inventory({"owner_csv": pd.DataFrame({"dengue_label": ["yes", None]})})
        self.assertEqual(present.iloc[0]["labeled_rows"], 1)
        self.assertFalse(bool(present.iloc[0]["model_available"]))

    def test_regional_counts_group_by_region_category_and_month(self):
        frame = pd.DataFrame(
            {
                "district": ["North", "North", "South", "South"],
                "reported": ["2025-01-02", "2025-01-19", "2025-02-02", "invalid"],
                "category": ["A", "B", "A", "A"],
            }
        )
        counts, ranked = aggregate_regional_trends(
            frame, "district", category_column="category", date_column="reported"
        )
        self.assertEqual(int(counts["records"].sum()), 3)
        self.assertIn("_period", counts.columns)
        self.assertIn("historical dataset", regional_summary(counts, "district"))
        self.assertEqual(int(ranked.iloc[0]["records"]), 1)


class ExplainabilityAndAgentsTests(unittest.TestCase):
    def test_legacy_feature_importance_is_global_and_local_is_unavailable(self):
        artifact = {"pipeline": StubLegacyModel(), "metadata": {}, "source": "legacy_demo"}
        importance = global_feature_importance(artifact)
        self.assertEqual(importance.iloc[0]["feature"], "age")
        self.assertTrue(local_probability_effects(artifact, patient_values(), "Medium Risk").empty)
        context = get_risk_explanation("Medium Risk", 52, 25, 130, 126, 1, 2)
        self.assertTrue(any("demonstration rule" in item for item in context))
        self.assertFalse(any("normal range" in item.lower() for item in context))

    def test_orchestrator_combines_patient_risk_explanation_and_biomarkers(self):
        artifact = {"pipeline": StubLegacyModel(), "metadata": {}, "source": "legacy_demo"}
        biomarker_frame = pd.DataFrame({"marker_x": [4.0], "patient_id": [1]})
        regional_frame = pd.DataFrame({"region": ["North", "South"]})
        result = AssessmentOrchestrator().run(
            artifact,
            patient_values(),
            biomarker_frame,
            0,
            regional_frame=regional_frame,
            regional_options={"region_column": "region"},
        )
        self.assertEqual(result["risk"]["risk_level"], "Medium Risk")
        self.assertIn("Risk Prediction Agent", result["agents"])
        self.assertEqual(result["report"]["summary"]["biomarker_values"]["marker_x"], 4.0)
        self.assertIn("not a medical diagnosis", result["report"]["markdown"])
        self.assertTrue(result["explanation"]["legacy_rule_context"])
        self.assertIn("Regional Trend Agent", result["agents"])
        self.assertIn("regions appear", result["regional"]["summary"])


if __name__ == "__main__":
    unittest.main()
