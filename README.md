# AI-Driven Multi-Agent Predictive Healthcare and Health Risk Scoring System

An academic Streamlit capstone prototype for health-risk scoring workflows, CSV quality checks, exploratory analysis, biomarker summaries, regional aggregation, model comparison, and modular local agents.

> **Medical disclaimer:** This application is an educational decision-support prototype. It does not diagnose disease, recommend treatment, or decide whether a person needs emergency care. It is not clinically validated and must not be used in place of a licensed healthcare professional.

## Problem statement

Health data can be difficult to inspect and summarize consistently. This project demonstrates how preprocessing, supervised classification, model interpretation, and specialist software modules can be combined in one dashboard. The checkout includes a public NHANES survey dataset for one documented binary outcome and a local-only feature for user-uploaded patient report files; no patient files are bundled with the project.

## Objectives

- Provide an interactive six-feature prototype health-risk assessment.
- Offer a reusable CSV validation, preprocessing, and EDA workflow.
- Compare Logistic Regression, Decision Tree, and Random Forest models when a suitable labeled dataset is supplied.
- Report weighted precision, recall, F1, accuracy, a confusion matrix, and cross-validation where class counts allow it.
- Expose global feature importance and, for newly trained artifacts, one-feature-at-a-time probability changes against training reference values.
- Summarize uploaded biomarker values without inventing CBC fields or reference intervals.
- Aggregate explicit region, category, and time fields as historical counts.
- Coordinate patient, risk, explanation, biomarker, regional, and report modules as local Python agents.
- Preserve emergency-call and nearby-hospital links.

## Features

- **Home:** project status, limitations, and emergency/facility links.
- **Health Risk Assessment:** age, BMI, systolic blood pressure, glucose, smoking, exercise, risk category, and model probabilities.
- **Patient Summary:** downloadable in-session Markdown summary. Assessment inputs are not persisted.
- **Patient Reports:** upload, group by a generated or entered reference code, download, and delete PDF/image/DOCX reports stored in a local SQLite database.
- **Data & EDA:** uploaded CSV validation, duplicate/missing-value review, IQR outlier counts, numeric distributions, boxplots, correlations, and category counts.
- **Biomarker Analysis:** descriptive summaries and charts for numeric columns in an uploaded file. Optional reference comparisons use only a user-supplied `biomarker,lower,upper` file.
- **Disease Risk:** checks whether supplied tables contain named anemia, dengue, or sepsis outcome columns. It does not create or claim a disease model.
- **Regional Trends:** counts by explicit region/category fields and optional parsed time periods. This is not outbreak detection.
- **Model Performance:** stratified holdout comparison and a saved final pipeline from an uploaded, labeled risk dataset.
- **About Project:** architecture, safety, and data limitations.

## Current dataset and model status

Current status:

- Official NHANES August 2021–August 2023 public-use XPT files are stored unchanged in `data/raw/`.
- The reproducible NHANES preparation creates `data/processed/nhanes_2021_2023_bpq020.csv` and descriptive EDA/quality artifacts under `reports/nhanes_2021_2023/`.
- `BPQ020` is a documented binary outcome for **reported ever-diagnosed hypertension history**. It is not future hypertension prediction, a medical diagnosis, or clinically validated.
- The processed dataset has 2,923 complete adult records: 1,058 positive and 1,865 negative. These are unweighted participant counts from the prepared analysis subset.
- No NHANES model has been trained or connected to the dashboard. `health_risk_model.pkl` remains the original Random Forest prototype built from 12 hard-coded illustrative rows; those rows are not NHANES or real clinical observations.
- The current Health Risk Assessment page still presents the legacy Low/Medium/High prototype behavior. It must not be interpreted as an NHANES output. Integrating the binary target would require a separately approved UI/model change.

## NHANES dataset preparation (no model training)

### Source and files

The source is the official [CDC/NCHS NHANES August 2021–August 2023 cycle](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023). The following public-use SAS transport files were downloaded from the CDC `DataFiles` release and retained unchanged in `data/raw/`:

- `DEMO_L.xpt` — demographics, MEC weights, strata/PSU, age (`RIDAGEYR`).
- `BMX_L.xpt` — measured BMI (`BMXBMI`).
- `BPXO_L.xpt` — oscillometric systolic readings (`BPXOSY1`, `BPXOSY2`, `BPXOSY3`).
- `GLU_L.xpt` — fasting glucose (`LBXGLU`) and fasting-subsample weight (`WTSAF2YR`).
- `SMQ_L.xpt` — `SMQ020` and `SMQ040`, used together to derive current cigarette smoking.
- `PAQ_L.xpt` — moderate/vigorous activity frequencies and units (`PAD790Q`, `PAD790U`, `PAD810Q`, `PAD810U`).
- `BPQ_L.xpt` — target response `BPQ020`.
- `SOURCE_MANIFEST.json` — source URLs, byte sizes, and SHA-256 checksums for those seven XPT files.

### Target and features

`BPQ020` asks whether a doctor or other health professional has **ever** told the participant they had hypertension (high blood pressure). Raw code `1` (Yes) is the positive class; raw code `2` (No) is the negative class. Refused (`7`), Don't know (`9`), and missing are excluded. The processed target column is `hypertension_history` with `1=Yes` and `0=No`. It measures **reported ever-diagnosed hypertension history**. The project does not call it future hypertension prediction, a medical diagnosis, or clinical validation.

Processed feature columns:

| Processed feature | NHANES source | Preparation |
|---|---|---|
| `age` | `DEMO_L.RIDAGEYR` | Adults age 18+; public ages 80+ are top-coded as 80. |
| `bmi` | `BMX_L.BMXBMI` | Require finite value greater than 0. |
| `systolic_bp` | `BPXO_L.BPXOSY1`–`BPXOSY3` | Mean of available positive readings no higher than 300 mmHg; no IQR-based trimming. |
| `fasting_glucose` | `GLU_L.LBXGLU` | Require finite value greater than 0 and positive fasting-subsample weight. This is fasting glucose, not a random glucose measurement. |
| `current_smoker` | `SMQ_L.SMQ020`, `SMQ040` | Never-smoked-100-cigarettes response maps to non-current; otherwise use current smoking response. Unresolved codes remain missing. |
| `physical_activity_sessions_per_week` | `PAQ_L.PAD790Q/U`, `PAD810Q/U` | Convert documented day/week/month/year frequencies to sessions per week, then sum moderate and vigorous leisure-time frequency. This is sessions, **not unique days active**. |

The PAQ codebook describes quantity checks of more than 4 per day, 28 per week, 31 per month, or 365 per year as **soft edits** that ask respondents to verify their answers. This workflow records these as review flags only; it does not cap or delete those observations. In the current processed file, five summed activity values exceed 56 sessions/week, a review reference formed by adding 4/day for each of two intensities across seven days. This is not an official NHANES limit for the summed feature. The initial model feature proposal excludes `systolic_bp`; it remains in this analysis table for descriptive use only because it is contemporaneous with, and may proxy, reported prior hypertension history.

The activity feature is not semantically identical to the app's current “Exercise days per week” field. A later model/UI integration must align that input to the prepared activity definition or choose a consistent one-intensity measure before training.

### Patient report storage

The **Patient Reports** page accepts PDF, PNG, JPG/JPEG, and DOCX files up to 20 MB. Files are stored as BLOBs in SQLite under the current user's local application-data directory (`%LOCALAPPDATA%\AIHealthRiskScoring\patient_reports.sqlite3` on Windows), outside the project/OneDrive folder by default. The user can supply a short reference code or have the app generate one, then use that reference to find, download, or delete saved reports. Patient names and identity documents are not required. Uploads are not OCR-processed, analyzed, or sent to an external service. This report archive is separate from the NHANES analytic dataset and does not constitute a searchable history of assessment measurements.

**Security limit:** local SQLite storage is not encrypted and the dashboard has no login or access control. Keep only test or de-identified documents in this academic prototype. Do not use it for identifiable patient files until storage encryption, authentication/authorization, protected backups, deletion procedures, and appropriate organizational review are in place. Anyone with access to the local account and database file can read stored reports.

### Merge and filtering

All seven files are merged one-to-one by NHANES respondent sequence number `SEQN`, using `DEMO_L` as the left/base table. Duplicate `SEQN` values abort preparation. The workflow retains only age 18+, valid BPQ020 codes, valid required features, nonmissing design variables, and positive `WTSAF2YR`. Missing values are not imputed: complete cases are saved, while component-level and derived feature missing counts are reported. The XPT reader normalizes the SAS transport representation for valid zero values (age, activity frequency, and weights) without changing any source file. Documented invalid codes are excluded; unusual but valid measurements are retained and reported through descriptive IQR outlier counts.

### Observed data quality and outputs

| Stage | Rows |
|---|---:|
| Original `DEMO_L` rows | 11,933 |
| Rows after `SEQN` left-merges | 11,933 |
| Removed as under 18 / invalid age | 3,780 |
| Removed for target refused / unknown / missing | 14 |
| Removed for incomplete/invalid feature or nonpositive fasting weight | 5,216 |
| Final complete-case dataset | **2,923** |

Final target distribution: **1,058 positive (reported Yes)** and **1,865 negative (reported No)**. These are unweighted row counts. The processed table also retains `WTSAF2YR`, `SDMVSTRA`, and `SDMVPSU` for survey-aware evaluation. The final row-level data has no missing values in its selected features, target, or retained design columns.

The reported source missing counts are calculated over the full 11,933-row demographic base after left-joining all components. Many blanks are expected because files have age eligibility rules, the glucose fasting subsample, or questionnaire skip patterns; they should not all be read as ordinary item nonresponse. The workflow writes:

- `data/processed/nhanes_2021_2023_bpq020.csv`
- `config/nhanes_2021_2023_bpq020.json` — source, variable mapping, target, exclusions, and design metadata.
- `reports/nhanes_2021_2023/data_quality_report.json`, `missing_values.csv`, `processed_summary.csv`, and `potential_outliers_iqr.csv`.
- `reports/nhanes_2021_2023/preprocessing_run_manifest.json` — source/script hashes, Python and dependency versions, execution timing, and processed CSV hash for a completed run.
- Histograms for each feature, `feature_boxplots.png`, `target_distribution.png`, `correlations.csv`, `correlation_heatmap.png`, and feature-vs-target plots/tables.

Reproduce with PowerShell from the project root (existing raw files are not overwritten):

```powershell
py -3.11 scripts\download_nhanes.py
py -3.11 scripts\prepare_nhanes.py
```

The EDA is descriptive and unweighted. Use fasting-subsample weight `WTSAF2YR` with NHANES stratum `SDMVSTRA` and PSU `SDMVPSU` for survey-aware population analysis. The valid positive-weight fasting subset is smaller than all participants with a glucose value; CDC notes that glucose was measured in a fasting subsample and requires its subsample weight. NHANES is cross-sectional, so BPQ020 describes past diagnosis report, not the timing of disease onset or a future outcome. The cohort is limited to complete adult records and may be affected by selection/nonresponse. No ML model or clinical validation has been performed.

## System architecture

```text
CSV upload / assessment form
        │
        ├── Data loader → validation → preprocessing → EDA
        │                                  │
        │                                  └── comparison/evaluation → saved sklearn pipeline
        │
        ├── Patient Analysis Agent → Risk Prediction Agent → Explanation Agent
        │                                                   │
        ├── Biomarker Analysis Agent ──────────────────────┤
        ├── Regional Trend Agent (historical aggregation)  │
        └────────────────────────────── Report/Summary Agent
```

The agents are ordinary Python components coordinated by `src/agents/orchestrator.py`; the project uses no external LLM API. Assessment inputs are held in the current Streamlit session only. Uploaded report documents persist separately in local SQLite storage.

## Project structure

```text
app.py
model.py
train_model.py
requirements.txt
README.md
health_risk_model.pkl              # legacy illustrative prototype
data/
  raw/                              # official NHANES XPT files, retained unchanged
  processed/                       # complete-case BPQ020 development table
config/                             # NHANES source/target/preprocessing configuration
reports/nhanes_2021_2023/            # quality report, EDA tables, and plots
scripts/
  download_nhanes.py
  prepare_nhanes.py
models/                             # created by model-training workflows
src/
  patient_records.py                 # local SQLite report-file storage
  nhanes_data.py                    # SEQN merge, filtering, quality, and EDA
  data_loader.py
  preprocessing.py
  eda.py
  model_training.py
  model_evaluation.py
  explainability.py
  biomarker_analysis.py
  patient_summary.py
  disease_risk.py
  regional_analysis.py
  agents/
utils/explanation.py               # clearly labeled legacy rule context
tests/                             # unittest and Streamlit smoke tests
```

## Technology stack

Python, Streamlit, Pandas, NumPy, scikit-learn, Joblib, Matplotlib, and Python's `unittest` framework.

## Installation

Use a working CPython installation. Phase 1 verification was run with Python 3.14.2. To preserve any existing environment, create a separate environment named `.venv-phase1` from the project root.

PowerShell:

```powershell
py -3.14 -m venv .venv-phase1
.\.venv-phase1\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python 3.14 is not available, use another supported CPython installation. In VS Code, select this project's `.venv-phase1\Scripts\python.exe` after creating the environment.

## Run the dashboard

```powershell
python -m streamlit run app.py
```

The dashboard normally opens at `http://localhost:8501`.

## Train and compare models

The existing dashboard training workflow remains a prototype **Low/Medium/High risk-category** workflow. It has not been adapted to the NHANES binary outcome. **Do not pass `nhanes_2021_2023_bpq020.csv` to this trainer**: its 0/1 values mean reported hypertension-history Yes/No, not Low/High risk. NHANES model training is intentionally deferred until the processed dataset is approved and the application is changed to use the documented binary target precisely.

For the existing prototype-only risk-category workflow, provide a separately sourced and labeled CSV with these columns:

```text
age,bmi,blood_pressure,glucose,smoking,exercise,risk_category
```

Target values must represent `Low Risk`, `Medium Risk`, `High Risk` (or `0`, `1`, `2`). The example above shows a schema only; no example patient rows are supplied. Keep the data source, label definition, units, inclusion criteria, and any de-identification information with your project records.

From the project root:

```powershell
python train_model.py --data path\to\your_labeled_risk_data.csv --target risk_category
```

The script validates required features, removes duplicate rows and rows with missing target labels, imputes missing model features inside the training pipeline, and compares Logistic Regression, Decision Tree, and Random Forest on a stratified holdout. It computes weighted accuracy, precision, recall, F1, a confusion matrix, and up to five-fold weighted F1 cross-validation when the training split has enough observations per class. Model choice prioritizes mean cross-validation weighted F1 when available; otherwise it uses holdout weighted F1, then recall and accuracy as tie-breakers. The selected pipeline is refit on all supplied rows and saved to `models/health_risk_pipeline.joblib` with provenance and evaluation metadata.

These metrics describe only the uploaded dataset and its split. They do not demonstrate clinical accuracy or validation. A small, biased, or unrepresentative dataset can produce misleading results.

The Streamlit **Model Performance** page performs the same evaluation and saves the selected pipeline. Training data is used for that session's action and is not automatically copied into the repository.

## Explainable AI

For supported estimators, the dashboard shows native global feature importance or coefficient magnitude. Newly trained artifacts also support a one-feature-at-a-time probability comparison against numeric medians and the training-mode smoking value. This describes model behavior relative to a reference row; it is not causal and is not a medical interpretation. The legacy Random Forest has no stored training reference values, so only its global feature importance and separately labeled old rule context are available. SHAP is not included as a dependency.

## Biomarker / CBC methodology

Only columns present in an uploaded CSV are analyzed. Numeric measured columns receive counts, missingness, mean, median, standard deviation, minimum, maximum, and descriptive plots. Selectable-text PDF reports can also be parsed for explicitly labeled hemoglobin, WBC, RBC, platelets, and hematocrit results, units, and same-report reference intervals. Scanned PDFs are not OCR-processed; absent or unparseable values remain unavailable. High/low labels appear only when a valid numeric interval is present in that report. No reference intervals are bundled, and the page is not a diagnostic tool.

## Disease and regional analysis

No condition-specific anemia, dengue, or sepsis model is supplied. A named label column alone does not establish enough valid training data or clinical utility. Regional charts count observed records by the uploaded region/category/time fields. Without population denominators, those counts are not rates; without validated time series, they are not outbreak monitoring.

## Testing

The test fixtures are small artificial mechanics-only examples. They are not clinical data and are never used to train the application model.

```powershell
python -m unittest discover -s tests -v
```

Tests cover CSV validation, preprocessing, EDA chart generation, training/evaluation mechanics, artifact persistence, prediction class ordering, biomarker summaries, explicit reference comparisons, disease-label inventory, regional grouping, agent orchestration, and Streamlit page smoke checks.

### Phase 1 functional verification (2026-11-03)

All dashboard interaction checks below used Streamlit AppTest. Patient report checks redirected `LOCALAPPDATA` to a temporary directory and used fictional PDFs. No existing patient database, project dataset, report, or model file was modified.

| Check | Status | Result |
|---|---|---|
| Complete automated suite | **PASSED** | `py -3.14 -m unittest discover -s tests -v`: 32 tests, 0 failures, 0 errors. |
| All navigation pages | **PASSED** | Home, Assessment, Patient Summary, Patient Reports, Data & EDA, Biomarker Analysis, Disease Risk, Regional Trends, Model Performance, About Project. |
| Patient Reports dashboard | **PASSED** | Fictional PDF uploaded and saved, searched by reference, retrieved, Download control displayed and clicked, and deleted after confirmation in isolated SQLite. |
| CBC PDF and biomarker CSV | **PASSED** | Fictional selectable-text CBC PDF extracted; CSV with a missing numeric value analyzed. Malformed PDF warning and malformed CSV error both displayed clear messages. |
| Assessment and patient summary | **PASSED** | Assessment submitted; summary rendered and Markdown download control displayed. |
| Data & EDA, Disease Risk, Regional Trends | **PASSED** | Fictional CSVs uploaded; missing-value CSV inspected, named disease label inventoried, and regional historical counts aggregated. |
| Model Performance NHANES guard | **PASSED** | Actual processed NHANES file recognized; `hypertension_history` training action disabled. No NHANES training was run. |
| Model Performance training/save UI | **PASSED (artificial data only)** | Temporary copy of the app compared classifiers and displayed the comparison and confusion matrix; saved artifact stayed in a temporary directory and was discarded. No fixture metrics are claimed as project performance. |
| Random Forest mechanics | **PASSED (artificial data only)** | Legacy artifact loaded; separate Random Forest trained and predicted from artificial rows. No score is reported. |
| Real risk-model evaluation | **NOT TESTED** | No appropriate owner-provided Low/Medium/High labeled dataset is present. No accuracy, precision, recall, or F1 is claimed. |
| Live browser session / deployment | **NOT TESTED** | AppTest only; no separate browser session or deployment was used. |

The first test run failed because `pypdf` was missing from the active Python environment. Installing the dependency already declared in `requirements.txt` resolved that environment issue; the full suite then passed. No project source changes were needed.

#### Verified software versions

Observed in the Python environment used for these checks; these versions are not pinned in `requirements.txt`.

| Component | Version |
|---|---:|
| Python | 3.14.2 |
| Streamlit | 1.64.0 |
| pandas | 3.0.6 |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.1 |
| Joblib | 1.6.0 |
| Matplotlib | 3.11.2 |
| pypdf | 6.21.4 |

#### Dataset and model actually checked

`health_risk_model.pkl` loads as a six-feature `RandomForestClassifier` with three classes and 100 trees. Project documentation identifies it as a legacy demonstration model trained from 12 illustrative rows. Its accuracy, precision, recall, F1, calibration, and clinical validity are not established; no performance numbers are attributed to it.

`data/processed/nhanes_2021_2023_bpq020.csv` has 2,923 rows: 1,865 reported-history negatives and 1,058 positives. Its target is reported ever-diagnosed hypertension history in cross-sectional NHANES data. It is not a future outcome or Low/Medium/High risk label, and it was not used to train the assessment model.

#### Remaining work and screenshot checklist

- [ ] Capture and include the screenshots below in the capstone report.
- [ ] If the submission requires model metrics, obtain a suitable labeled risk dataset and document its provenance, label meaning, and evaluation design. Do not use the NHANES history target as risk categories.
- [ ] Use only de-identified/test reports until authentication, authorization, encryption, backup protection, and appropriate privacy review are addressed.
- [ ] If required by the rubric, separately verify browser rendering on the intended presentation machine; AppTest does not validate browser/device behavior.

Screenshots to capture (fictional inputs only):

- [ ] Home, including model-status and academic-prototype notices.
- [ ] Health Risk Assessment form and demonstration-only result.
- [ ] Patient Summary and Markdown download control.
- [ ] Patient Reports with a fictional saved report, Download control, and delete confirmation (isolated test database).
- [ ] Data & EDA preview, missing-value summary, and outlier review.
- [ ] Biomarker Analysis with extracted fictional CBC and CSV summary; optionally malformed-file alerts.
- [ ] Disease Risk label inventory and no-disease-model notice.
- [ ] Regional Trends fictional historical counts and the not-a-rate caption.
- [ ] Model Performance NHANES-target restriction, or a real, appropriate owner-approved evaluation if one becomes available; never stage invented metrics.
- [ ] About Project architecture, limitations, and privacy notice.

### Historical preliminary audit record (2026-11-02; superseded by Phase 1 above)

- **Interpreter discovery:** `py -0p` reports registered Python 3.14 and 3.11 installations. Launching the venv interpreter fails because its base Python 3.14 executable is inaccessible; launching the registered Microsoft Store Python 3.11 also fails with “Access is denied”. The `python` command is not on PATH.
- **Unit and UI smoke tests:** Not run. The interpreter could not start, so no test result is claimed.
- **Streamlit startup/navigation:** Not run for the same reason. A successful app launch and page-by-page verification remain required on a working Python installation.
- **Dependency installation:** Not attempted because no usable interpreter/pip is available. `requirements.txt` contains Streamlit, pandas, NumPy, scikit-learn, Joblib, Matplotlib, and pypdf; install them in a recreated project environment.
- **Audit inventory:** The checkout contains seven NHANES XPT source files and a source manifest, one prepared CSV, analysis reports, and the legacy model artifact. They were left unchanged.

## Historical audit checklist (superseded by Phase 1)

### Audit findings, ordered by importance

1. **[High] Runtime verification is blocked on the current machine.** Both discoverable Python executables return “Access is denied”, the venv points to the inaccessible Python 3.14 base, and `python` is absent from PATH. Recreate the venv after installing usable CPython, then run tests and launch the app.
2. **[High] Patient report privacy controls are incomplete.** SQLite report BLOBs are local but unencrypted and there is no login or authorization. Do not store identifiable reports in this prototype.
3. **[High] Model output is not clinically validated.** The default pickle is a 12-row illustrative Random Forest. Its risk classes/probabilities are interface demonstrations only; no metrics should be attributed to it.
4. **[Medium] Dataset-to-model mismatch is intentional and must remain clear.** NHANES files and prepared analysis are present, but the binary `BPQ020` outcome is not connected to the Low/Medium/High assessment model. It describes reported history, not future disease.
5. **[Medium] “Nearby hospitals” is a third-party search shortcut.** It opens Google Maps using typed area text or “near me”; the app does not access device GPS, enumerate verified hospitals, estimate distance, or confirm availability. The user's browser and Google Maps handle location permissions and results. No app API key is required.
6. **[Low] Multi-agent wording needs precision.** Six separate local Python component classes exist and the orchestrator calls them, but they are not autonomous processes or LLM agents. Current documentation now describes that implementation precisely.
7. **[Low] PDF extraction is deliberately limited.** It extracts a small set of explicit labels from selectable text only; malformed, encrypted, or scanned PDFs should surface errors, and no OCR or inference is performed.

### Feature implementation inventory

Implemented in code: assessment, in-session patient summary, local report storage, CSV quality/EDA, biomarker CSV analysis, selectable-text CBC extraction, named disease-label inspection, regional aggregation, model comparison/training from uploaded labeled CSV, local agent orchestration, and emergency/Google Maps links. The availability of these code paths is not a claim of runtime verification in this review. No condition-specific disease model, GPS-backed facility finder, OCR, or clinical validation is present.

### Historical app, source, and test audit scope

Reviewed `app.py`, `model.py`, `train_model.py`, `src/`, `utils/`, `tests/`, `requirements.txt`, and this README. No test suite could be executed because the local Python executables could not start. Static inspection found that imports and sidebar dispatch are present for the documented pages; this does not establish that startup or each path succeeds at runtime. `model.py` can load a joblib/pickle artifact, so only load artifacts from a trusted source; pickle-based model files are not safe for arbitrary untrusted uploads.

### Historical remaining verification tasks (superseded by Phase 1)

- [ ] Install a working CPython 3.11+ and recreate `.venv` from the project root.
- [ ] Install `requirements.txt` into that environment and record installed versions.
- [ ] Run `python -m unittest discover -s tests -v`; investigate any failures without weakening assertions.
- [ ] Start the dashboard with the command below and smoke-check every sidebar page, especially report upload/download/delete, CSV workflows, CBC PDF errors, and model evaluation.
- [ ] Confirm Google Maps opens a search for the entered area; results, distance, location permission, and availability are provided by Google Maps and are not verified by this app. This is not an emergency service.
- [ ] Keep the legacy 12-row illustrative model clearly labeled; do not report it as validated or use its probabilities for care.
- [ ] Before training, document the labeled dataset source, target meaning, feature units, missingness, leakage checks, split strategy, and suitable held-out evaluation.
- [ ] Keep report files de-identified/test-only until authentication, authorization, encryption, protected backup, and deployment controls are reviewed.
- [ ] Capture current screenshots and confirm the final README claims against the running app before university submission.

## Viva preparation questions

1. What problem does this project address, and who is its intended user?
2. Which modules are called agents, and how does `AssessmentOrchestrator` coordinate them? Are they autonomous or LLM-based?
3. What does the legacy model predict, what data trained it, and why are its probabilities illustrative?
4. Why is the NHANES `BPQ020` label not a future hypertension diagnosis or prediction?
5. How were the NHANES files joined and filtered, and why is the fasting subsample weight relevant?
6. What leakage could arise from using systolic blood pressure to predict a previously reported hypertension diagnosis?
7. How does the training workflow separate the holdout from model selection and what do weighted F1 and the confusion matrix communicate?
8. What limitations follow from cross-sectional survey data, complete-case selection, class imbalance, and the absence of clinical validation?
9. How are missing values handled in CSV analysis and in the model pipeline?
10. What can the CBC PDF extractor read, and how does it respond to malformed or scanned documents?
11. Where are uploaded patient reports stored, who can access them, and what privacy controls are missing?
12. Does the hospital feature return verified facilities or provide emergency response? What external service handles its search?
13. Which checks have been executed, and which features remain unverified?

## Screenshots

Screenshots are not included in this checkout. Use the Phase 1 screenshot checklist above; use fictional or owner-approved de-identified data only.

## Limitations

- The NHANES file represents a public-use population survey sample, not patient-level clinical records or longitudinal outcomes.
- The legacy 12-row model is illustrative only; its probability values are not calibrated.
- No clinical validation, prospective evaluation, fairness analysis, or external validation has been performed.
- The app accepts a limited six-feature schema; it is not a full electronic health record system.
- Model explanations are associations with model output, not causes or clinical conclusions.
- Assessment inputs are not kept as longitudinal history. Uploaded reports persist locally without encryption, authentication, access control, or a deployment security configuration; see Patient report storage above.
- Emergency calling and Google Maps behavior depends on the user's device/browser and location permissions/settings; the app does not dispatch emergency services.

## Future scope

With lawful access to appropriately governed data: document dataset provenance and label meaning; add validated disease-specific models only with adequate labels; add lab-specific, population-appropriate reference ranges and units; perform calibration, subgroup/fairness, and external validation; evaluate privacy/security before persistence; and obtain clinical and ethics review before any real-world use.

## Disclaimer

This capstone is for education and demonstration. It does not diagnose, prevent, monitor, or treat any disease; does not provide emergency triage; and has not been clinically validated. Seek qualified medical care for health concerns. In an emergency, contact local emergency services directly.
