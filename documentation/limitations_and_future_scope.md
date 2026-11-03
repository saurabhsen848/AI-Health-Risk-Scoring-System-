# Limitations and Future Scope

## Current limitations

### Clinical and model validity

This project is an academic prototype, not a clinically validated decision-support tool. The default Random Forest is described as trained on 12 illustrative rows. It is not a clinical dataset, its probabilities are not calibrated, and no genuine held-out model metrics are available. The NHANES-derived table is not connected to that model. The NHANES target records reported prior hypertension diagnosis history, not future disease or a diagnosis made by the application.

No clinical, prospective, external, subgroup/fairness, or calibration validation has been performed. Model explanations describe model behavior, not causal effects or medical reasons.

### Data and statistical validity

NHANES is cross-sectional and has a complex survey design. The prepared table applies adult and complete-case restrictions and a fasting subsample requirement. Descriptive row counts are unweighted. Any population estimate or uncertainty analysis would need correct weights and design methods. Missingness, selection, response rates, and leakage must be considered. Measured systolic blood pressure may overlap conceptually with a reported previous hypertension diagnosis.

### Application and feature boundaries

- Six-input risk scoring uses a limited schema and does not constitute diagnosis or future risk prediction.
- CBC PDF support is limited to a small label set in selectable text. There is no OCR.
- Disease Risk inspects named label columns; it does not provide disease-specific models.
- Regional Trends counts uploaded observations. Without denominators and a validated time-series method, it does not report rates or detect outbreaks.
- The hospital button delegates a textual search to Google Maps; it does not verify locations, distance, open status, or capacity and is not emergency response.
- Streamlit startup and page behavior were not verified during this review because the available Python interpreters failed to start.

### Privacy and security

Patient reports are persisted in local SQLite without encryption, account login, or access control. There is no deployment hardening or protected backup mechanism. Only synthetic/test or appropriately de-identified files should be used in this academic prototype. Joblib/pickle artifacts must be loaded only from trusted sources. No private patient data, passwords, or API keys are included in this documentation.

## Future scope

1. Establish a reproducible Python environment and complete the test and live-page verification recorded in `testing_results.md`.
2. Add integration tests for Patient Reports and file-upload paths, and perform a page-by-page manual review.
3. For any model study, select lawful, well-documented labeled data; define the target and time horizon; audit leakage and missingness; use a suitable split design; report held-out metrics and uncertainty; and retain a clear distinction from clinical validation.
4. If NHANES modeling is pursued, plan a survey-aware analysis and explicitly address the cross-sectional target, fasting subsample, weights/strata/PSUs, missingness, and temporal leakage.
5. Add calibrated, population-appropriate reference ranges only from authoritative sources and preserve units and provenance.
6. Before any identifiable report storage, implement authentication, authorization, encryption at rest and in transit, protected backup/retention/deletion procedures, and institutional privacy/security review.
7. If a verified hospital locator is required, integrate a documented places provider with appropriate licensing/API key management, explicit location consent, clear errors, and a verified-data timestamp; never present it as emergency dispatch.
8. Add OCR only with explicit scope, confidence handling, user review, and tested document formats.

All future work remains academic until governance, clinical evaluation, and applicable institutional approvals are addressed.
