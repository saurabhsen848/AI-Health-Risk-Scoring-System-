# Methodology

## Software workflow

The user selects a dashboard page, supplies assessment values or uploads a file, and the relevant page function calls reusable functions in `src/`. CSV workflows validate the DataFrame before descriptive analysis. The optional training workflow prepares an input dataset, partitions labeled rows for evaluation, compares three classifiers, reports held-out metrics, and can save the selected full-data pipeline. The bundled NHANES preprocessing pipeline is separate from the risk model workflow.

## Health assessment workflow

`app.py` collects age, BMI, blood pressure, glucose, smoking, and exercise inputs. `model.py` loads a trained pipeline if present or falls back to the legacy `health_risk_model.pkl`. Prediction uses the feature order in `src/preprocessing.py`; where estimator probabilities are available, they are displayed using the estimator's class ordering. The default legacy artifact is labeled in the UI as a 12-row illustrative model. Its probabilities are not calibrated and must not be interpreted clinically.

The orchestrator calls locally implemented components to create patient feature values, risk output, explanation details, optional biomarker and regional summaries, and a report summary. No external AI service is involved in this orchestration.

## CSV and descriptive analysis

Uploaded CSVs go through `src/data_loader.py`. The dashboard uses quality summaries, missingness and data types, numeric distributions, boxplots, correlations, and categorical counts where the uploaded columns support them. Biomarker analysis describes observed numeric values; optional comparisons require ranges supplied by the user. Regional aggregation counts uploaded rows by explicit fields and is not population-rate estimation or outbreak detection.

## CBC PDF extraction

`src/pdf_biomarker.py` reads PDF text with pypdf and searches a limited set of explicit CBC labels and same-line values. It does not OCR images or infer absent results. Invalid, encrypted, or textless/scanned PDFs have explicit extraction errors. A result is flagged against a range only if a numeric interval is explicitly parsed from the same report.

## NHANES preparation methodology

The preparation workflow reads seven unchanged XPT files, validates that respondent IDs are unique, and performs one-to-one left joins by `SEQN` onto `DEMO_L`. It derives a selected adult complete-case dataset using `BPQ020` for reported lifetime hypertension diagnosis history, age, BMI, available oscillometric systolic pressure readings, fasting glucose, smoking questionnaire responses, and weekly activity frequency. It requires positive fasting-subsample weights and retains design variables. The documented preparation does not impute missing development data, does not fit a model, and creates descriptive quality/EDA outputs.

The app assessment's six-field feature schema is different from the NHANES prepared columns. The NHANES table is not a ready-made assessment-model training set; `BPQ020` describes prior diagnosis report rather than future disease. In particular, systolic pressure is contemporaneous and can proxy the existing history label, so leakage/temporal validity concerns would need to be addressed before any model design.

## Machine learning evaluation approach

The current training code is designed for an owner-provided labeled CSV with six expected features and an explicitly chosen target. It uses preprocessing within a scikit-learn pipeline, stratified train/test splitting, and compares Logistic Regression, Decision Tree, and Random Forest. It computes weighted accuracy, precision, recall, F1, a confusion matrix, and weighted-F1 cross-validation on the training partition if class counts permit. Cross-validation informs selection when available; holdout metrics are reported separately. The final selected estimator is refit on all supplied labeled rows and saved with metadata. This code describes intended mechanics; no such project-dataset evaluation was run for this review.

The legacy model artifact's training rows are illustrative and do not constitute a valid held-out dataset. No genuine project model performance metrics are available to report.
