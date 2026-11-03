# Machine Learning Documentation

## What the current Random Forest does

The application can load `health_risk_model.pkl`, the legacy Random Forest prototype. `model.py` accepts one row with six values (age, BMI, blood pressure, glucose, smoking, and exercise), runs the model, maps its class to Low/Medium/High Risk, and returns class probabilities when the estimator supports them. `src/explainability.py` can show global feature importance. Legacy rule text is clearly a demonstration; it is not a medical explanation.

The project describes this artifact as trained on 12 hard-coded illustrative rows. That tiny illustrative sample is not an appropriate clinical dataset, does not support reliable generalization, and provides no trustworthy validation or probability calibration. The app must keep the prototype label and medical disclaimer visible. The bundle does not provide genuine performance metrics for this model.

## Optional training workflow

`train_model.py` and the Model Performance page accept a separately provided labeled CSV and target column. `src/preprocessing.py` canonicalizes required feature names, validates/normalizes target labels, and builds a preprocessing pipeline. `src/model_training.py` compares:

1. Logistic Regression
2. Decision Tree
3. Random Forest (200 trees, class-balanced subsampling, fixed random state)

The data is split with a stratified holdout (default 25%, random state 42). Preprocessing is fit inside each model pipeline so imputation/scaling/encoding are learned from its training fold rather than globally beforehand. When class counts support it, weighted F1 cross-validation is run on the training partition and used for selection. The holdout is used to report weighted accuracy, precision, recall, F1 and a confusion matrix. If CV is infeasible due to sparse class counts, the code falls back to holdout weighted F1 with recall/accuracy tie-breakers. The selected pipeline is then refit on all supplied labeled records for later prediction and saved with metadata.

Metrics from any future run would describe only that uploaded dataset and split. They would not be clinical validation. The user must establish the dataset source, target semantics, feature units, exclusion rules, missingness, independence of records, potential target leakage, class support, and suitability of train/test separation before training.

## Dataset status and metric reporting

The NHANES `hypertension_history` table is not used by the training workflow and is not interchangeable with the app's Low/Medium/High risk categories. Its outcome is a cross-sectional report of prior diagnosis. No appropriate labeled Low/Medium/High risk dataset is documented in the checkout, and no model training/evaluation results were obtained during this review. Accordingly, this documentation reports no accuracy, precision, recall, F1, AUC, or other performance metric for a project model.

The only sample rows in automated training tests are generated artificial mechanics fixtures. They test code behavior, not model validity, and their calculated test output is not a project result.

## Limitations and cautions

- No model has clinical validation, prospective validation, external validation, calibration assessment, fairness evaluation, or clinician review.
- A small, synthetic, mislabeled, unrepresentative, duplicated, or leaked dataset can produce misleading held-out metrics.
- Random stratified splitting does not address repeated patient records, temporal leakage, or site-level generalization unless the dataset and split design explicitly account for them.
- The prepared NHANES target is not future disease; its survey design, cross-sectional nature, missingness, and selected cohort must be addressed in any research model.
- Joblib/pickle artifacts can execute code when loaded; only load artifacts from trusted sources.

This is an academic software prototype, not a diagnostic device or clinical decision tool.
