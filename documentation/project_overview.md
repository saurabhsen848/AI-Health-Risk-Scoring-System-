# Project Overview

## Title

**AI-Driven Multi-Agent Predictive Healthcare and Health Risk Scoring System**

## Purpose

This university capstone is an educational Streamlit prototype that brings together a six-input health-risk demonstration, data inspection tools, CBC text extraction, report-file storage, model-training utilities, and small local Python modules coordinated by an orchestrator. It demonstrates how these software components can be organized and surfaced in one dashboard. It is not a clinical system.

## Objectives represented in the code

- Present a prototype Low/Medium/High risk assessment using six fields.
- Accept CSV files for validation, descriptive exploration, biomarker summaries, regional aggregation, and optional supervised model comparison.
- Extract a small set of explicitly labeled blood count results from selectable-text PDFs.
- Store uploaded report files in a local SQLite database.
- Keep patient, risk, explanation, biomarker, regional, and report-summary logic in separate Python classes coordinated by `AssessmentOrchestrator`.
- Prepare and document a public NHANES analysis table without automatically training a disease model.

## Actual status and data

The default `health_risk_model.pkl` is documented by the project as a Random Forest trained on 12 hard-coded illustrative examples. It is not trained on the bundled NHANES data, and its outputs are not clinically meaningful or validated. NHANES files from August 2021–August 2023 and a derived 2,923-row complete-case table are present. That table's `hypertension_history` target means a participant reported having ever been told by a health professional that they had hypertension. The repository's NHANES preparation configuration explicitly says model training was not performed.

No trained model evaluation metrics are reported in this documentation. The test suite is present but was not run during this review because neither available Python interpreter could start.

## Intended audience and safety

The intended audience is students, instructors, and reviewers examining a software prototype. The app must not be used to diagnose, predict an individual's future disease, recommend treatment, or triage emergencies. It has not been clinically validated.

## Technical stack

Python, Streamlit, pandas, NumPy, scikit-learn, Joblib, Matplotlib, pypdf, SQLite (Python standard library), and `unittest`. The dependency names are listed in the root `requirements.txt`; versions are mostly unpinned and installed versions were not verified for this review.

## Project layout

```text
app.py                         Streamlit pages and user interactions
model.py                       Prediction artifact loading and scoring
train_model.py                 CLI training/evaluation from labeled CSV
requirements.txt               Runtime dependencies
.streamlit/config.toml         Theme settings
data/raw/                      Official NHANES XPT files + source manifest
data/processed/                Derived NHANES analysis table
config/                        NHANES variables and preparation decisions
reports/nhanes_2021_2023/      Data-quality and descriptive EDA outputs
scripts/                       NHANES download and preparation workflows
src/                           Data, analysis, preprocessing, models, storage
src/agents/                    Local component classes and orchestrator
utils/                         Legacy illustrative rule explanation
tests/                         unittest and Streamlit AppTest test modules
documentation/                 Technical project documentation
```

## Run command

From the project root, after creating a working Python environment and installing `requirements.txt`:

```powershell
python -m streamlit run app.py
```

At review time the checked-in `.venv` and registered Store Python were inaccessible on the machine. See [testing_results.md](testing_results.md) for exact status and next steps.
