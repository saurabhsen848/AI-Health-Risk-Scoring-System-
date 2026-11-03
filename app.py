"""Streamlit dashboard for the AI-driven predictive-health capstone prototype."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from model import load_prediction_artifact
from src.agents.orchestrator import AssessmentOrchestrator
from src.agents.regional_agent import RegionalTrendAgent
from src.biomarker_analysis import biomarker_columns, biomarker_summary, compare_reference_ranges, patient_biomarker_summary
from src.data_loader import DATA_DIR, csv_inventory, dataset_quality, load_csv, load_uploaded_csv
from src.disease_risk import disease_label_inventory
from src.eda import category_distribution, correlation_heatmap, feature_distribution, numeric_boxplots, numeric_eda_columns
from src.model_training import evaluate_models, fit_full_model, save_model_artifact
from src.patient_records import (
    ALLOWED_REPORT_TYPES,
    MAX_REPORT_BYTES,
    REPORT_TYPE_LABELS,
    count_patient_reports,
    database_path as patient_reports_database_path,
    delete_patient_report,
    list_patient_reports,
    new_patient_reference,
    normalize_patient_reference,
    read_patient_report,
    save_patient_report,
)
from src.pdf_biomarker import PDFExtractionError, extract_cbc_pdf
from src.preprocessing import HEALTH_FEATURES, outlier_report, prepare_health_dataset, validate_risk_labels
from src.regional_analysis import DATE_CANDIDATES, REGION_CANDIDATES, suggested_column


st.set_page_config(
    page_title="Predictive Healthcare | Capstone",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Shared medical palette and low-contrast-free page treatment for every route. */
.stApp { background: linear-gradient(135deg, #eaf6fb 0%, #f8fcff 48%, #e8f3f8 100%); }
[data-testid="stAppViewContainer"] { background: transparent; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #e3f1f8, #f2f8fc); border-right: 1px solid #c5dce8; }
[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255,255,255,.88); border-color: #c9dce7; }
.stButton > button, [data-testid="stDownloadButton"] button { border-radius: 9px; font-weight: 600; }
input, textarea, [data-baseweb="select"] > div { background-color: #fff !important; }
[data-testid="stAlert"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the available risk model…")
def get_model_artifact():
    return load_prediction_artifact()


def read_uploaded_csv(uploaded) -> pd.DataFrame | None:
    if uploaded is None:
        return None
    try:
        return load_uploaded_csv(uploaded)
    except Exception as exc:  # surface parser/validation errors in the page, not as a crash
        st.error(f"Could not read this CSV: {exc}")
        return None


def show_table_quality(frame: pd.DataFrame) -> None:
    quality = dataset_quality(frame)
    left, middle, right = st.columns(3)
    with left.container(border=True):
        st.metric("Rows", f"{quality['rows']:,}")
    with middle.container(border=True):
        st.metric("Columns", quality["columns"])
    with right.container(border=True):
        st.metric("Duplicate rows", quality["duplicate_rows"])
    with st.expander("Missing values and data types"):
        quality_frame = pd.DataFrame(
            {
                "column": list(quality["dtypes"]),
                "type": list(quality["dtypes"].values()),
                "missing": [quality["missing_by_column"][column] for column in quality["dtypes"]],
            }
        )
        st.dataframe(quality_frame, width="stretch", hide_index=True)


def show_figure(figure) -> None:
    if figure is None:
        st.info("This chart needs at least one suitable numeric column.")
        return
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def emergency_panel() -> None:
    emergency, hospitals = st.columns([1, 1])
    with emergency:
        st.markdown(
            '<a href="tel:112" style="display:block;text-align:center;background:#c62828;color:white;'
            'padding:14px;border-radius:10px;font-weight:700;text-decoration:none">☎ Call emergency number 112</a>',
            unsafe_allow_html=True,
        )
    with hospitals:
        location = st.text_input("City or area for hospital search", key="hospital_location", placeholder="e.g., Pune")
        query = quote(f"hospital near {location}" if location else "hospital near me")
        st.markdown(
            f'<a href="https://www.google.com/maps/search/{query}" target="_blank" '
            'style="display:block;text-align:center;background:#1769aa;color:white;padding:14px;'
            'border-radius:10px;font-weight:700;text-decoration:none">⌖ Find nearby hospitals</a>',
            unsafe_allow_html=True,
        )
    st.caption("Emergency and hospital links open your device dialer or Google Maps. The AI does not determine emergency need.")


def render_home(artifact: dict | None) -> None:
    st.title("AI-Driven Multi-Agent Predictive Healthcare")
    st.write("A modular BCA capstone dashboard for exploratory health-data analysis and prototype risk assessment.")
    st.info(
        "This is an educational decision-support prototype. It does not diagnose disease, provide treatment advice, "
        "or replace a licensed healthcare professional. No clinical validation is claimed."
    )
    available_csvs = csv_inventory(DATA_DIR)
    model_label = "Not available" if artifact is None else (
        "Legacy demo model (12 illustrative rows)" if artifact.get("source") == "legacy_demo" else "Trained project pipeline"
    )
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.metric("Risk model", model_label)
    with c2.container(border=True):
        st.metric("Local CSV datasets", len(available_csvs))
    with c3.container(border=True):
        st.metric("Patient reports stored locally", count_patient_reports())
    if artifact and artifact.get("source") == "legacy_demo":
        st.warning(
            "The available Random Forest is the original prototype trained on 12 hard-coded illustrative rows. "
            "It is not a real clinical dataset, its probabilities are not calibrated, and it has not been validated. "
            "Use it only to demonstrate the interface."
        )
    st.subheader("Dashboard modules")
    cards = st.columns(4)
    for column, heading, detail in zip(
        cards,
        ["Risk assessment", "Data & EDA", "Biomarkers", "Regional trends"],
        ["Risk category and model probabilities", "Validation, distributions, and outlier review", "Observed values only unless reference ranges are supplied", "Historical counts only when geography/time columns exist"],
    ):
        with column:
            with st.container(border=True):
                st.markdown(f"#### {heading}")
                st.write(detail)
    st.subheader("Emergency assistance and facility discovery")
    emergency_panel()


def render_assessment(artifact: dict | None) -> None:
    st.title("Health Risk Assessment")
    st.write("Enter the six features accepted by the available prototype model.")
    if artifact is None:
        st.error("No model file is available. Add a trained, labeled risk model before assessing risk.")
        return
    if artifact.get("source") == "legacy_demo":
        st.warning("Legacy prototype model: predictions are illustrative only and are not clinically validated.")
    else:
        metadata = artifact.get("metadata", {})
        st.success(f"Model: {metadata.get('model_name', 'trained pipeline')} · training rows: {metadata.get('training_rows', 'not recorded')}")

    biomarker_upload = st.file_uploader(
        "Optional biomarker/CBC CSV to include one row in the patient summary",
        type=["csv"],
        key="assessment_biomarker_csv",
    )
    biomarker_frame = read_uploaded_csv(biomarker_upload)
    biomarker_row = None
    if biomarker_frame is not None:
        if not biomarker_columns(biomarker_frame):
            st.warning("The uploaded file has no numeric biomarker-like columns; it will not be attached to the assessment.")
            biomarker_frame = None
        else:
            biomarker_row = st.selectbox("Biomarker dataset row to attach", range(len(biomarker_frame)), key="assessment_biomarker_row")

    with st.container(border=True):
        with st.form("risk_assessment_form"):
            left, right = st.columns(2)
            with left:
                age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
                bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=22.0, step=0.1)
                blood_pressure = st.number_input("Systolic blood pressure", min_value=50, max_value=260, value=120)
            with right:
                glucose = st.number_input("Glucose level", min_value=20, max_value=500, value=100)
                smoking_option = st.selectbox("Smoking status", ["No", "Yes"])
                exercise = st.number_input("Exercise days per week", min_value=0, max_value=7, value=3)
            submitted = st.form_submit_button("Assess risk", type="primary", width="stretch")

    if submitted:
        patient_features = {
            "age": age,
            "bmi": bmi,
            "blood_pressure": blood_pressure,
            "glucose": glucose,
            "smoking": 1 if smoking_option == "Yes" else 0,
            "exercise": exercise,
        }
        try:
            result = AssessmentOrchestrator().run(
                artifact,
                patient_features,
                biomarker_frame=biomarker_frame,
                biomarker_row=biomarker_row,
            )
            st.session_state["last_assessment"] = result
            st.session_state["last_patient_features"] = patient_features
        except Exception as exc:
            st.error(f"Risk assessment could not be completed: {exc}")

    result = st.session_state.get("last_assessment")
    if result:
        st.divider()
        st.subheader("Risk assessment result")
        level = result["risk"]["risk_level"]
        if level == "Low Risk":
            st.success(f"Category: {level}")
        elif level == "Medium Risk":
            st.warning(f"Category: {level}")
        else:
            st.error(f"Category: {level}")
        probabilities = result["risk"].get("probabilities", {})
        if probabilities:
            st.caption("Model probability output. These values are not clinically calibrated.")
            st.bar_chart(pd.Series(probabilities, name="Model probability"), y_label="Probability")
        else:
            st.caption("This model does not expose class probabilities.")
        st.subheader("Model explanation")
        st.caption(result["explanation"]["basis"])
        importance = result["explanation"]["global_importance"]
        if not importance.empty:
            st.bar_chart(importance.set_index("feature")["importance"], y_label="Global importance")
        for item in result["explanation"]["potential_contributors"]:
            st.write(f"• {item}")
        if result["explanation"].get("legacy_rule_context"):
            st.markdown("**Legacy prototype input context (fixed demo rules, not medical cutoffs):**")
            for item in result["explanation"]["legacy_rule_context"]:
                st.write(f"• {item}")
        if result["explanation"]["local_effects"].empty:
            st.caption("Patient-specific positive/negative contributions are unavailable for this artifact.")
        st.info("These are model-related patterns, not medical conclusions or a disease diagnosis.")
    st.subheader("Emergency assistance")
    emergency_panel()


def render_patient_summary() -> None:
    st.title("Patient Health Summary")
    result = st.session_state.get("last_assessment")
    if not result:
        st.info("Complete a risk assessment first to create an in-session summary.")
        return
    st.markdown(result["report"]["markdown"])
    st.caption("Assessment inputs and results are held only in this session. Uploaded report files can be stored separately on this computer from Patient Reports.")
    st.download_button(
        "Download summary as Markdown",
        result["report"]["markdown"],
        file_name="patient_health_summary.md",
        mime="text/markdown",
    )


def render_patient_records() -> None:
    st.title("Patient Report Storage")
    st.write("Upload and keep report files on this computer, grouped by a patient reference code.")
    st.info(
        "No patient name, contact details, or medical-record number is required. Leave the reference blank "
        "to generate a code and keep it to find this patient's reports later. Files are stored locally and "
        "are not read, summarized, OCR-processed, or sent to an AI service."
    )
    st.warning(
        "This academic prototype does not encrypt stored files or require a user login. "
        "Use only test or de-identified reports until appropriate protections are added for real patient data."
    )
    st.caption(f"Local database: `{patient_reports_database_path()}` · Maximum upload size: 20 MB")

    with st.form("patient_report_upload_form", clear_on_submit=True):
        patient_reference = st.text_input(
            "Patient reference code (optional; do not enter a name or government ID)",
            max_chars=64,
            placeholder="e.g., PT-001 or leave blank to generate one",
        )
        report_type = st.selectbox("Report type", REPORT_TYPE_LABELS)
        uploaded_report = st.file_uploader(
            "Choose a patient report",
            type=sorted(ALLOWED_REPORT_TYPES),
            accept_multiple_files=False,
            help="Accepted formats: PDF, PNG, JPG/JPEG, DOCX. Maximum 20 MB.",
        )
        save_clicked = st.form_submit_button("Store report locally", type="primary", width="stretch")

    if save_clicked:
        if uploaded_report is None:
            st.error("Choose a report file before saving.")
        elif uploaded_report.size > MAX_REPORT_BYTES:
            st.error("This file exceeds the 20 MB upload limit.")
        else:
            try:
                reference = normalize_patient_reference(patient_reference) if patient_reference.strip() else new_patient_reference()
                record_id = save_patient_report(
                    reference,
                    report_type,
                    uploaded_report.name,
                    uploaded_report.type,
                    uploaded_report.getvalue(),
                )
                st.session_state["patient_reports_lookup"] = reference
                st.session_state["patient_reports_saved_reference"] = reference
                st.session_state["patient_reports_confirm_delete"] = None
                st.success(f"Report saved on this computer. Patient reference: **{reference}**")
            except (OSError, ValueError) as exc:
                st.error(f"Could not save this report: {exc}")
            except Exception as exc:
                st.error(f"Could not save this report: {exc}")

    saved_reference = st.session_state.pop("patient_reports_saved_reference", None)
    if saved_reference:
        st.info(f"Keep this patient reference for future uploads and lookup: **{saved_reference}**")

    st.subheader("Find saved reports")
    lookup = st.text_input(
        "Patient reference code",
        key="patient_reports_lookup",
        max_chars=64,
        placeholder="Enter the code used when saving the report",
    )
    if not lookup.strip():
        st.caption("Enter a patient reference code to view its stored reports.")
        return
    try:
        reference = normalize_patient_reference(lookup)
        reports = list_patient_reports(reference)
    except ValueError as exc:
        st.error(str(exc))
        return
    if not reports:
        st.info("No saved reports were found for that reference.")
        return

    st.caption(f"{len(reports)} report(s) stored for reference {reference}.")
    for report in reports:
        report_id = str(report["report_id"])
        with st.container(border=True):
            st.markdown(f"**{report['original_filename']}** · {report['report_type']}")
            st.caption(
                f"Uploaded {report['uploaded_at_utc']} UTC · "
                f"{int(report['byte_count']):,} bytes · Reference {report['patient_reference']}"
            )
            stored = read_patient_report(report_id)
            if stored is None:
                st.warning("This report is no longer in local storage.")
                continue
            actions = st.columns([1, 1, 3])
            with actions[0]:
                st.download_button(
                    "Download",
                    data=stored["file_bytes"],
                    file_name=str(stored["original_filename"]),
                    mime=str(stored["content_type"]),
                    key=f"download_patient_report_{report_id}",
                    width="stretch",
                )
            with actions[1]:
                if st.button("Delete", key=f"delete_patient_report_{report_id}", width="stretch"):
                    st.session_state["patient_reports_confirm_delete"] = report_id
            if st.session_state.get("patient_reports_confirm_delete") == report_id:
                st.warning("Delete this stored report permanently from the local database?")
                confirm, cancel, _ = st.columns([1, 1, 3])
                with confirm:
                    if st.button("Confirm delete", key=f"confirm_delete_patient_report_{report_id}"):
                        delete_patient_report(report_id)
                        st.session_state["patient_reports_confirm_delete"] = None
                        st.rerun()
                with cancel:
                    if st.button("Cancel", key=f"cancel_delete_patient_report_{report_id}"):
                        st.session_state["patient_reports_confirm_delete"] = None
                        st.rerun()


def render_data_eda() -> None:
    st.title("Data Management & Exploratory Analysis")
    st.write("Upload a CSV to inspect its structure. Files are processed in this session and are not added to the project automatically.")
    uploaded = st.file_uploader("Health or research CSV", type=["csv"], key="eda_csv")
    frame = read_uploaded_csv(uploaded)
    if frame is None:
        if not csv_inventory(DATA_DIR):
            st.info("No CSV datasets are currently present under data/. No synthetic dataset is created.")
        return
    show_table_quality(frame)
    st.subheader("Preview")
    st.dataframe(frame.head(20), width="stretch")
    st.subheader("Potential outlier review")
    report = outlier_report(frame)
    if report.empty:
        st.info("No numeric columns are available for an IQR outlier review.")
    else:
        st.dataframe(report, width="stretch", hide_index=True)
        st.caption("IQR flags are statistical review prompts; rows are not removed and flags are not medical abnormality indicators.")
    numeric = numeric_eda_columns(frame)
    if numeric:
        st.subheader("Feature distribution")
        selected = st.selectbox("Numeric feature", numeric, key="eda_numeric_feature")
        show_figure(feature_distribution(frame, selected))
        st.subheader("Numeric boxplots")
        show_figure(numeric_boxplots(frame))
        st.subheader("Correlation")
        show_figure(correlation_heatmap(frame))
    categorical = [column for column in frame.columns if column not in numeric]
    if categorical:
        st.subheader("Category distribution")
        selected_category = st.selectbox("Category or target column", categorical, key="eda_category")
        show_figure(category_distribution(frame, selected_category))


def render_biomarker_analysis() -> None:
    st.title("CBC / Biomarker Analysis")
    st.write("Analyze a biomarker CSV or extract clearly labeled CBC values from a selectable-text PDF. No OCR or diagnosis is performed.")
    uploaded = st.file_uploader("Biomarker CSV", type=["csv"], key="biomarker_csv")
    pdf_upload = st.file_uploader("CBC lab report PDF", type=["pdf"], key="biomarker_pdf")
    if pdf_upload is not None:
        try:
            extracted = extract_cbc_pdf(pdf_upload.getvalue())
        except PDFExtractionError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Could not process the CBC PDF: {exc}")
        else:
            if extracted.empty:
                st.info("No supported CBC result labels were found in selectable PDF text. No values were inferred.")
            else:
                st.subheader("Values extracted from this report")
                st.dataframe(extracted, width="stretch", hide_index=True)
                st.caption("Reference ranges and high/low labels are shown only when parsed from this report. This is not a diagnosis.")
    frame = read_uploaded_csv(uploaded)
    if frame is None:
        if pdf_upload is None:
            st.info("Upload a CSV table or a selectable-text CBC PDF to analyze available values.")
        return
    show_table_quality(frame)
    columns = biomarker_columns(frame)
    if not columns:
        st.warning("No numeric biomarker-like columns were found in this CSV.")
        st.dataframe(frame.head(20), width="stretch")
        return
    summary = biomarker_summary(frame)
    st.subheader("Observed biomarker summary")
    st.dataframe(summary, width="stretch", hide_index=True)
    st.caption("No disease interpretation is made. High/low flags require reference bounds supplied for the applicable lab and context.")
    numeric_view = pd.DataFrame({column: pd.to_numeric(frame[column], errors="coerce") for column in columns})
    selected = st.selectbox("Biomarker distribution", columns, key="biomarker_feature")
    show_figure(feature_distribution(numeric_view, selected))
    show_figure(numeric_boxplots(numeric_view))

    range_file = st.file_uploader("Optional reference-ranges CSV (biomarker, lower, upper)", type=["csv"], key="reference_csv")
    if range_file is not None:
        try:
            ranges_frame = load_uploaded_csv(range_file)
            if not {"biomarker", "lower", "upper"}.issubset(ranges_frame.columns):
                st.error("Reference file must contain biomarker, lower, and upper columns.")
            else:
                lower_bounds = pd.to_numeric(ranges_frame["lower"], errors="coerce")
                upper_bounds = pd.to_numeric(ranges_frame["upper"], errors="coerce")
                if lower_bounds.isna().any() or upper_bounds.isna().any() or (lower_bounds > upper_bounds).any():
                    st.error("Reference bounds must be numeric, non-missing, and have lower ≤ upper.")
                    return
                range_map = dict(
                    zip(
                        ranges_frame["biomarker"].astype(str),
                        zip(lower_bounds.astype(float), upper_bounds.astype(float)),
                    )
                )
                row_position = st.selectbox("Dataset row for reference comparison", range(len(frame)), key="biomarker_row")
                values = patient_biomarker_summary(frame, row_position).set_index("biomarker")["value"].to_dict()
                comparison = compare_reference_ranges(values, range_map)
                st.dataframe(comparison, width="stretch", hide_index=True)
                st.caption("Comparisons use only the bounds you supplied. They are not a diagnosis and must be interpreted by a clinician.")
        except Exception as exc:
            st.error(f"Could not process the reference file: {exc}")


def render_disease_risk() -> None:
    st.title("Disease-Risk Analysis")
    st.warning("The application does not diagnose anemia, dengue, sepsis, or any other disease.")
    st.write("No condition-specific labeled dataset is present in this project. This page checks supplied data for named outcome columns; it does not train or imply a disease detector.")
    uploaded = st.file_uploader("Optional labeled dataset to inspect", type=["csv"], key="disease_csv")
    frame = read_uploaded_csv(uploaded)
    if frame is None:
        try:
            local = {path.name: load_csv(path) for path in csv_inventory(DATA_DIR)}
        except Exception as exc:
            st.error(f"A local dataset could not be loaded: {exc}")
            return
        if not local:
            st.info("No disease outcome labels are available. No disease-risk prediction can be supported from the current files.")
            return
        datasets = local
    else:
        datasets = {uploaded.name: frame}
    inventory = disease_label_inventory(datasets)
    if inventory.empty:
        st.info("No anemia_label, dengue_label, sepsis_label, or equivalent named outcome columns were found.")
    else:
        st.dataframe(inventory, width="stretch", hide_index=True)
        st.info("A label column alone does not establish a valid predictive model, adequate sample size, or clinical validity.")


def render_regional_trends() -> None:
    st.title("Regional Health Trends")
    st.write("Historical aggregation only. No outbreak detection or real-time monitoring is implemented.")
    uploaded = st.file_uploader("Regional dataset CSV", type=["csv"], key="regional_csv")
    frame = read_uploaded_csv(uploaded)
    if frame is None:
        st.info("No regional dataset is present in the project. Upload a table with an explicit region column to explore it.")
        return
    show_table_quality(frame)
    region_default = suggested_column(frame, REGION_CANDIDATES)
    date_default = suggested_column(frame, DATE_CANDIDATES)
    columns = list(frame.columns)
    region_index = columns.index(region_default) if region_default in columns else 0
    region_column = st.selectbox("Region column", columns, index=region_index, key="region_column")
    category_options = ["(none)"] + columns
    category_column = st.selectbox("Optional category/disease column", category_options, key="region_category")
    date_options = ["(none)"] + columns
    date_index = date_options.index(date_default) if date_default in date_options else 0
    date_column = st.selectbox("Optional date/time column", date_options, index=date_index, key="region_date")
    try:
        regional_result = RegionalTrendAgent().analyze(
            frame,
            region_column,
            category_column=None if category_column == "(none)" else category_column,
            date_column=None if date_column == "(none)" else date_column,
        )
        counts, ranked = regional_result["counts"], regional_result["ranked"]
        st.metric("Regions represented", int(counts[region_column].nunique()))
        st.write(regional_result["summary"])
        st.dataframe(ranked, width="stretch", hide_index=True)
        if "_period" in counts:
            st.subheader("Historical records over time")
            if category_column != "(none)":
                categories = counts[category_column].dropna().unique().tolist()
                if categories:
                    selected_category = st.selectbox("Category to chart", categories, key="regional_category_filter")
                    counts = counts[counts[category_column] == selected_category]
            trend = counts.pivot_table(index="_period", columns=region_column, values="records", aggfunc="sum").fillna(0)
            st.line_chart(trend)
        else:
            st.bar_chart(counts.groupby(region_column)["records"].sum().sort_values(ascending=False))
        st.caption("Counts reflect the uploaded rows only; without population denominators they are not regional rates.")
    except Exception as exc:
        st.error(f"Could not aggregate this dataset: {exc}")


def render_model_performance(artifact: dict | None) -> None:
    st.title("Model Performance & Training")
    st.write("Compare classifiers only on an owner-provided labeled CSV with all six supported features and a risk-category target.")
    if artifact and artifact.get("metadata", {}).get("holdout_metrics"):
        st.subheader("Metrics saved with the active trained model")
        st.dataframe(pd.DataFrame([artifact["metadata"]["holdout_metrics"]]), width="stretch", hide_index=True)
        st.caption("These are the recorded holdout metrics for the active artifact; the comparison table below is produced when its training CSV is supplied again.")
    uploaded = st.file_uploader("Labeled health-risk CSV", type=["csv"], key="model_csv")
    frame = read_uploaded_csv(uploaded)
    if frame is None:
        st.info("There is no labeled health-risk CSV in data/. The existing 12-row prototype is not used for evaluation because it cannot provide a credible holdout estimate.")
        return
    show_table_quality(frame)
    targets = [column for column in frame.columns if column not in HEALTH_FEATURES]
    if not targets:
        st.error("Choose a dataset that has a risk-category label column in addition to the six model features.")
        return
    target_column = st.selectbox("Risk target column", targets, key="risk_target")
    hypertension_history_target = str(target_column).strip().lower() == "hypertension_history"
    if hypertension_history_target:
        st.info(
            "This target records reported ever-diagnosed hypertension history in cross-sectional NHANES data. "
            "It is not a future outcome or a Low/Medium/High risk label, so this risk-category trainer cannot use it."
        )
    if st.button(
        "Compare models, evaluate, and save selected pipeline",
        type="primary",
        disabled=hypertension_history_target,
    ):
        try:
            features, target = prepare_health_dataset(frame, target_column)
            validate_risk_labels(target)
            evaluation = evaluate_models(features, target)
            fitted = fit_full_model(features, target, evaluation.best_model_name)
            output_path = Path(__file__).resolve().parent / "models" / "health_risk_pipeline.joblib"
            save_model_artifact(
                fitted,
                output_path,
                target_name=target_column,
                training_rows=len(features),
                model_name=evaluation.best_model_name,
                evaluation=evaluation,
                training_features=features,
            )
            st.cache_resource.clear()
            st.session_state["model_evaluation"] = evaluation
            st.session_state["model_artifact_path"] = str(output_path)
        except Exception as exc:
            st.error(f"Training/evaluation could not be completed: {exc}")

    evaluation = st.session_state.get("model_evaluation")
    if evaluation is not None:
        st.subheader("Model comparison (stratified holdout)")
        st.dataframe(evaluation.comparison, width="stretch", hide_index=True)
        st.caption(f"Selection method: {evaluation.selection_basis}. Selected: {evaluation.best_model_name}.")
        st.subheader("Selected model confusion matrix")
        st.dataframe(evaluation.confusion, width="stretch")
        st.success(f"Final pipeline fit on all uploaded labeled rows and saved to {st.session_state['model_artifact_path']}")
        st.warning("Metrics are dataset-specific and do not demonstrate clinical accuracy or medical validation.")


def render_about() -> None:
    st.title("About the Capstone")
    st.markdown(
        """
        **Project:** AI-Driven Multi-Agent Predictive Healthcare and Health Risk Scoring System  
        **Purpose:** demonstrate modular data analysis, supervised learning workflows, interpretable model outputs, and a Streamlit dashboard.

        ### Current data limitations
        - NHANES public survey data is available for a documented hypertension-history outcome; it is not an individual patient's longitudinal medical record.
        - The available `health_risk_model.pkl` is the original Random Forest trained from 12 hard-coded illustrative rows. It is not clinical data and is not validated.
        - No holdout performance claim is made for that legacy artifact. Model comparison becomes available after a real labeled risk CSV is provided.
        - CBC reference intervals are not included. Biomarker status is shown only if explicit ranges are supplied.
        - Disease prediction, regional rates, outbreak detection, assessment-history storage, and external LLM services are not implemented.

        ### Local agent architecture
        Patient Analysis → Risk Prediction → Explanation → Biomarker Analysis → Report/Summary.  
        A separate Regional Trend Agent aggregates uploaded historical data. Agents are local Python modules; no patient data is sent to an external AI API.

        ### Safety and privacy
        This project is an educational prototype, not a medical device or diagnosis system. Do not use it for treatment or emergency triage. Report files uploaded through Patient Reports are stored locally without encryption or sign-in controls; use test or de-identified reports only until appropriate protections are added. Assessment inputs and summaries remain in the current Streamlit session unless downloaded.
        """
    )
    st.subheader("Emergency links")
    emergency_panel()


artifact = get_model_artifact()
with st.sidebar:
    st.title("🩺 Predictive Health")
    st.caption("Academic capstone prototype")
    page = st.radio(
        "Navigate",
        ["Home", "Health Risk Assessment", "Patient Summary", "Patient Reports", "Data & EDA", "Biomarker Analysis", "Disease Risk", "Regional Trends", "Model Performance", "About Project"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("For emergencies, contact local emergency services. This app does not triage.")

if page == "Home":
    render_home(artifact)
elif page == "Health Risk Assessment":
    render_assessment(artifact)
elif page == "Patient Summary":
    render_patient_summary()
elif page == "Patient Reports":
    render_patient_records()
elif page == "Data & EDA":
    render_data_eda()
elif page == "Biomarker Analysis":
    render_biomarker_analysis()
elif page == "Disease Risk":
    render_disease_risk()
elif page == "Regional Trends":
    render_regional_trends()
elif page == "Model Performance":
    render_model_performance(artifact)
else:
    render_about()
