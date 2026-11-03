"""CSV loading and dataset inspection helpers."""

from __future__ import annotations

from pathlib import Path
from io import BytesIO
import csv

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV and reject empty or malformed tables with useful errors."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    try:
        payload = csv_path.read_bytes()
        _validate_csv_header(payload, csv_path.name)
        frame = pd.read_csv(BytesIO(payload))
    except (pd.errors.ParserError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"Could not parse CSV file {csv_path}: {exc}") from exc
    return validate_dataframe(frame)


def load_uploaded_csv(uploaded) -> pd.DataFrame:
    """Apply the same header, parsing, and validation rules to an upload."""
    name = getattr(uploaded, "name", "uploaded file")
    try:
        payload = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
        _validate_csv_header(payload, name)
        return validate_dataframe(pd.read_csv(BytesIO(payload)))
    except (pd.errors.ParserError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"Could not parse CSV file {name}: {exc}") from exc


def _validate_csv_header(payload: bytes, name: str) -> None:
    try:
        header = next(csv.reader([payload.decode("utf-8-sig").splitlines()[0]]))
    except (UnicodeDecodeError, IndexError, csv.Error) as exc:
        raise ValueError(f"Could not read a CSV header from {name}.") from exc
    duplicates = sorted({value for value in header if header.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate column names found: {duplicates}")


def validate_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate structural properties without changing the input data."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if frame.empty:
        raise ValueError("The dataset has no data rows.")
    if len(frame.columns) == 0:
        raise ValueError("The dataset has no columns.")
    if frame.columns.has_duplicates:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {duplicates}")
    result = frame.copy()
    for name in [c for c in result.columns if str(c).strip().lower().startswith("unnamed:")]:
        values = result[name]
        nonmissing = values.dropna()
        if nonmissing.empty:
            result = result.drop(columns=[name])
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        expected = pd.Series(range(len(result)), dtype="float64").to_numpy()
        if numeric.notna().all() and (numeric.astype(float).to_numpy() == expected).all():
            result = result.drop(columns=[name])
        else:
            raise ValueError(
                f"Column '{name}' is not an empty or sequential saved index. Rename it to a meaningful column name before analysis."
            )
    if len(result.columns) == 0:
        raise ValueError("The dataset has no usable columns after removing an empty saved index.")
    return result


def csv_inventory(data_dir: str | Path = DATA_DIR) -> list[Path]:
    """List active CSV datasets, excluding archived backups and generated reports."""
    root = Path(data_dir)
    if not root.exists():
        return []
    excluded = {"backup", "backups", "report", "reports"}
    return sorted(
        path for path in root.rglob("*.csv")
        if path.is_file() and not (excluded & {part.lower() for part in path.relative_to(root).parts[:-1]})
    )


def dataset_quality(frame: pd.DataFrame) -> dict[str, object]:
    """Return non-destructive quality diagnostics for a table."""
    checked = validate_dataframe(frame)
    return {
        "rows": len(checked),
        "columns": len(checked.columns),
        "duplicate_rows": int(checked.duplicated().sum()),
        "missing_by_column": checked.isna().sum().astype(int).to_dict(),
        "dtypes": {str(column): str(dtype) for column, dtype in checked.dtypes.items()},
    }
