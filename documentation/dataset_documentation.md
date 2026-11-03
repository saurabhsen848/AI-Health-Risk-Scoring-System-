# Dataset Documentation

## Dataset inventory

The repository contains public-use U.S. CDC/NCHS NHANES August 2021–August 2023 data, not a private patient database. Seven SAS transport files are in `data/raw/`, with URLs, byte sizes, and SHA-256 hashes recorded in `data/raw/SOURCE_MANIFEST.json`:

| File | Component | Variables used in preparation |
|---|---|---|
| `DEMO_L.xpt` | Demographics and sample design | `SEQN`, `RIDAGEYR`, `SDMVSTRA`, `SDMVPSU` |
| `BMX_L.xpt` | Body measures | `BMXBMI` |
| `BPXO_L.xpt` | Oscillometric blood pressure | `BPXOSY1`–`BPXOSY3` |
| `GLU_L.xpt` | Fasting plasma glucose | `LBXGLU`, `WTSAF2YR` |
| `SMQ_L.xpt` | Smoking questionnaire | `SMQ020`, `SMQ040` |
| `PAQ_L.xpt` | Physical activity questionnaire | `PAD790Q/U`, `PAD810Q/U` |
| `BPQ_L.xpt` | Blood pressure/cholesterol questionnaire | `BPQ020` |

Derived files are in `data/processed/` and descriptive outputs are in `reports/nhanes_2021_2023/`. The code/config describes a merge of 11,933 demographic rows, with 2,923 final complete adult records. These are unweighted row counts; no model has been fit on them.

The checked-in `data_quality_report.json` records 1,058 positive (`Yes`, 36.20%) and 1,865 negative (`No`, 63.80%) target records in the prepared table. These proportions describe this filtered, unweighted complete-case sample and are not population estimates.

## Target definition

`BPQ_L.BPQ020` asks whether a doctor or other health professional has ever told the participant they had hypertension. Raw value `1` is mapped to `hypertension_history=1` (Yes), and raw value `2` to `0` (No). Refused (`7`), don't know (`9`), and missing are excluded. This target measures reported history, not onset, current measured diagnosis, future hypertension, or an individual clinical diagnosis.

## Selected fields and derivations

- `age`: `DEMO_L.RIDAGEYR`; age 18+, with public ages 80+ top-coded at 80.
- `bmi`: `BMX_L.BMXBMI`; finite and greater than zero.
- `systolic_bp`: row mean of positive `BPXOSY1`–`BPXOSY3` values no greater than 300 mmHg.
- `fasting_glucose`: finite positive `GLU_L.LBXGLU`; the workflow requires positive `WTSAF2YR`.
- `current_smoker`: derived from `SMQ020` and `SMQ040` according to questionnaire skip logic.
- `physical_activity_sessions_per_week`: moderate and vigorous session frequencies converted from days/week/month/year and summed. This is sessions per week, not unique days active.
- Retained survey design fields: fasting subsample weight `WTSAF2YR`, stratum `SDMVSTRA`, PSU `SDMVPSU`.

The actual prepared NHANES features are not identical to the six application-form fields. `src/nhanes_data.py` lists an initial proposed model feature set that excludes systolic blood pressure; no NHANES classifier is trained by the preparation workflow.

## Data handling and limitations

Source XPT files are retained unchanged. Duplicate `SEQN` values abort preparation; merges are checked one-to-one. The development table uses complete cases and excludes invalid/missing required values, nonpositive fasting weights, and unlabeled target codes. No imputation is done in this dataset preparation. Source missingness and quality counts are written to reports. Unusual valid measurements are generally retained and described rather than automatically trimmed.

NHANES is a complex, cross-sectional survey of the U.S. civilian noninstitutionalized population. Population estimates and uncertainty require survey weights and design-aware methods. The project's descriptive counts are unweighted; complete-case filtering and fasting-subsample restrictions limit representativeness. No claim of national inference or clinical representativeness is made from the simple row counts.

The legacy Random Forest uses only 12 illustrative/hard-coded rows according to the code/UI documentation, not NHANES. The tests use artificial fixture data for software mechanics. Neither should be described as real clinical observations.

## Reproduction

From the project root, the existing workflow documents:

```powershell
py -3.11 scripts\download_nhanes.py
py -3.11 scripts\prepare_nhanes.py
```

The preparation configuration and scripts should be checked before rerunning. Existing source, processed, or report outputs should be backed up if any future workflow change could overwrite them. This review did not rerun preprocessing because Python was unavailable.
