# Feature Implementation Status

“Implemented” below means code was found during static review. It does not imply the feature launched successfully in this environment. The test module column identifies existing test coverage; **none of these tests were executed during this review**.

| Feature | Code location | Status from audit | Test coverage in repository | Remaining verification / caveat |
|---|---|---|---|---|
| Home and sidebar navigation | `app.py` | Implemented in code | `tests/test_app_smoke.py` covers selected pages | AppTest could not run; patient report page is absent from the all-pages navigation loop. |
| Six-input health risk assessment | `app.py`, `model.py` | Implemented as illustrative prototype | Model and selected assessment smoke tests | Runtime unverified; default model is 12 illustrative rows and is not clinically validated. |
| Patient summary | `app.py`, `src/patient_summary.py` | Implemented in code | Summary/orchestrator-related module tests | Streamlit page not live-checked. |
| Patient report storage | `app.py`, `src/patient_records.py` | Implemented local SQLite upload/list/download/delete | No direct report-database test module identified in reviewed tests | UI/database integration unverified. Unencrypted, no login/access control; use only test/de-identified reports. |
| CSV validation and EDA | `src/data_loader.py`, `src/eda.py`, `app.py` | Implemented in code | `tests/test_data_pipeline.py` | Test execution and actual Streamlit upload flow unverified. |
| Biomarker CSV summaries | `src/biomarker_analysis.py`, `app.py` | Implemented in code | `tests/test_health_modules.py` | UI upload and charts unverified at runtime. Only supplied columns/ranges are used. |
| CBC PDF extraction | `src/pdf_biomarker.py`, `app.py` | Partial by design: selectable-text CBC labels only | Tests cover text parse, generated selectable-text PDF, invalid and scanned-PDF errors | No OCR; representative real report formats and live UI errors not verified. |
| Disease risk model | `src/disease_risk.py`, `app.py` | Incomplete/not present | Label inventory test only | No anemia, dengue, sepsis, or other condition model. Page only inventories named target columns. |
| Regional trends | `src/regional_analysis.py`, `app.py` | Implemented as historical aggregation | Regional grouping test | Uploaded CSV workflow unverified; counts are not population rates or outbreak alerts. |
| Model comparison/training | `src/model_training.py`, `train_model.py`, `app.py` | Implemented for owner-provided labeled CSV | Mechanics tests for metrics, comparison, artifact save/predict | Not run on a real suitable dataset; no genuine model metrics available. |
| NHANES download/preparation | `scripts/`, `src/nhanes_data.py`, `config/` | Implementation and generated outputs are present | No test execution established here | Source hashes/files and output inventory exist; workflow not rerun in this review. Does not train a model. |
| Local multi-agent orchestration | `src/agents/` | Implemented as six local Python components and a coordinating orchestrator | Orchestration test exists | Test unrun; these are not autonomous or LLM agents. |
| Hospital finder | `app.py` | Partial: Google Maps query link only | No test coverage identified | No in-app facility result, GPS, distance, or availability. Google Maps/browser handles search. Not an emergency service. |
| Emergency link | `app.py` | Implemented as `tel:112` link | No test coverage identified | Device/browser behavior not verified; number suitability depends on user location. |

## Status summary

The main workflow modules exist in code. Runtime status is **unverified** because Python could not be launched. Features explicitly incomplete include OCR, disease-specific prediction, GPS/API-backed facility lookup, and clinical validation. No screenshots from a running application are included in this review.
