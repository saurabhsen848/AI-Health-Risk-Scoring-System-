# Testing Results

## Environment discovery

Review date: 2026-11-02.

- `py -0p` listed Python 3.14 at `C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe` and Python 3.11 under the WindowsApps package.
- Running `.venv\Scripts\python.exe --version` failed because the venv's configured base Python 3.14 executable returned **Access is denied**.
- Directly invoking the registered WindowsApps Python 3.11 executable also returned **Access is denied**.
- PowerShell could find `py.exe`, but not `python` or `python3` commands.
- The venv's `pyvenv.cfg` confirms its base points to the inaccessible Python 3.14 path.

## Results table

| Check | Result | Evidence / limitation |
|---|---|---|
| Static review of requested files and source layout | Completed | Reviewed `app.py`, `model.py`, `train_model.py`, `src/`, `utils/`, `tests/`, `requirements.txt`, README, config, and NHANES manifests. Static inspection does not prove runtime behavior. |
| Dependency installation | Not run | No usable Python/pip interpreter. `requirements.txt` names Streamlit, pandas, NumPy, scikit-learn, Joblib, Matplotlib, and pypdf>=5.0. Installed versions not verified. |
| `python -m unittest discover -s tests -v` | Not run | Interpreter could not start. There are three test modules plus Streamlit smoke tests; do not report these as passed. |
| Streamlit startup | Not run | No Streamlit process was started. |
| Sidebar/page navigation and interactive feature checks | Not run | AppTest test definitions exist but did not execute; no live browser verification. |
| CBC malformed/scanned PDF checks | Not run | Unit tests define these cases, but no executed result is available. |
| Model metrics | None available | No model evaluation was run on a suitable held-out real dataset. The tests' artificial fixtures are not project metrics. |
| NHANES preparation rerun | Not run | Existing files/reports were inspected; preprocessing was not rerun. |

## Test coverage present in source

`tests/test_data_pipeline.py` covers CSV validation, preprocessing, quality summaries, and chart builders. `tests/test_model_pipeline.py` covers metric functions, model comparison mechanics, synthetic fixture fitting, artifact persistence, prediction class ordering, explainability, and insufficient class support. `tests/test_health_modules.py` covers biomarker summaries/reference intervals, CBC parsing and PDF errors, disease-label inventory, regional aggregation, explainability, and orchestration. `tests/test_app_smoke.py` defines Streamlit AppTest checks for Home, assessment, and selected sidebar pages. Coverage definitions are not test results.

## Required next steps on a working environment

1. Install a standard CPython distribution (Python 3.11 is preferred) that is executable for this Windows user.
2. From the project root, create a separate environment so the inaccessible existing `.venv` is preserved: `py -3.11 -m venv .venv-review`.
3. Install dependencies: `.\.venv-review\Scripts\python.exe -m pip install -r requirements.txt`.
4. Run tests: `.\.venv-review\Scripts\python.exe -m unittest discover -s tests -v`.
5. Start the dashboard: `.\.venv-review\Scripts\python.exe -m streamlit run app.py`.
6. Record actual output, inspect each sidebar page, and exercise uploads using non-sensitive fixtures. In particular, test Patient Reports separately because the current sidebar smoke loop does not include that page.

Until those steps succeed, project-wide runtime verification remains incomplete. Do not describe the application or test suite as verified.
