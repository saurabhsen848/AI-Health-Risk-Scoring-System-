"""Streamlit dashboard smoke tests."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class DashboardSmokeTests(unittest.TestCase):
    def test_home_and_assessment_pages_run_without_streamlit_exceptions(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("AI-Driven Multi-Agent" in item.value for item in app.title))
        navigation = app.sidebar.radio[0]
        navigation.set_value("Health Risk Assessment")
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Health Risk Assessment" in item.value for item in app.title))
        next(button for button in app.button if button.label == "Assess risk").click()
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Risk assessment result" in item.value for item in app.subheader))

    def test_all_navigation_pages_run_without_uploaded_data(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        pages = [
            "Patient Summary",
            "Data & EDA",
            "Biomarker Analysis",
            "Disease Risk",
            "Regional Trends",
            "Model Performance",
            "About Project",
            "Home",
        ]
        for page in pages:
            app.sidebar.radio[0].set_value(page)
            app.run()
            self.assertFalse(app.exception, f"Streamlit page failed: {page}")

    def test_biomarker_page_runs_with_no_dataset(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        app.sidebar.radio[0].set_value("Biomarker Analysis")
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("CBC / Biomarker Analysis" in item.value for item in app.title))


if __name__ == "__main__":
    unittest.main()
