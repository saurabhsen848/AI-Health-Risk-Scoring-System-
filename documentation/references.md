# References

The following sources are authoritative references for the data and software discussed in this documentation. Accessed 2026-11-02. Project-local manifests and code should be used alongside external documentation to reproduce the exact files and transformations in this checkout.

## Dataset and survey documentation

1. Centers for Disease Control and Prevention, National Center for Health Statistics (CDC/NCHS). **NHANES August 2021–August 2023: Questionnaires, Datasets, and Related Documentation.** [Cycle index and analytic materials](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023).
2. CDC/NCHS. **Blood Pressure & Cholesterol (BPQ_L): Data Documentation, Codebook, and Frequencies.** [BPQ_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPQ_L.htm). Documents `BPQ020` and analytic notes.
3. CDC/NCHS. **Demographic Variables and Sample Weights (DEMO_L): Data Documentation, Codebook, and Frequencies.** [DEMO_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm). Documents respondent identifiers, age, survey design and weights.
4. CDC/NCHS. **Body Measures (BMX_L): Data Documentation, Codebook, and Frequencies.** [BMX_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.htm).
5. CDC/NCHS. **Blood Pressure – Oscillometric Measurements (BPXO_L): Data Documentation, Codebook, and Frequencies.** [BPXO_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPXO_L.htm).
6. CDC/NCHS. **Plasma Fasting Glucose (GLU_L): Data Documentation, Codebook, and Frequencies.** [GLU_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GLU_L.htm). Documents the fasting subsample and the need for the corresponding subsample weight.
7. CDC/NCHS. **NHANES 2021–2023 data file index**, including smoking and physical activity components. [Data index](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023). The project source manifest identifies local `SMQ_L.xpt` and `PAQ_L.xpt` URLs and hashes.
8. CDC/NCHS. **Brief Overview of Sample Design, Nonresponse Bias Assessment, and Analytic Guidelines for NHANES August 2021–August 2023.** [Overview and analytic guidance](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/overviewbrief.aspx?Cycle=2021-2023).
9. CDC/NCHS. **NHANES Survey Methods and Analytic Guidelines.** [Analytic guidelines](https://wwwn.cdc.gov/nchs/nhanes/analyticguidelines.aspx).

## Software and methods documentation

10. scikit-learn developers. **RandomForestClassifier.** [API reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
11. scikit-learn developers. **train_test_split.** [API reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
12. scikit-learn developers. **Classification metrics.** [Model evaluation reference](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).
13. Streamlit, Inc. **Streamlit documentation.** [Official documentation](https://docs.streamlit.io/).
14. pypdf contributors. **pypdf documentation.** [Official documentation](https://pypdf.readthedocs.io/).
15. Python Software Foundation. **Python `unittest` documentation.** [Official documentation](https://docs.python.org/3/library/unittest.html).

## Repository provenance records

- `data/raw/SOURCE_MANIFEST.json`: exact seven NHANES source URLs, sizes and SHA-256 checksums for this checkout.
- `config/nhanes_2021_2023_bpq020.json`: target definition, feature mappings, cohort filters, missingness, survey-design fields, and explicit statement that the workflow does not train a model.
- `reports/nhanes_2021_2023/`: generated quality and descriptive analysis outputs. Their presence does not imply model validation.
- `requirements.txt`: declared Python dependencies; most are not version-pinned.
