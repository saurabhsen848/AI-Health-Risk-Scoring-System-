"""Local-only storage for uploaded patient report files.

Uploads are stored as SQLite BLOBs in the current user's local application-data
directory. This module does not parse reports, run OCR, or transmit data.
"""

from __future__ import annotations

import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


MAX_REPORT_BYTES = 20 * 1024 * 1024
ALLOWED_REPORT_TYPES = {"pdf", "png", "jpg", "jpeg", "docx"}
REPORT_TYPE_LABELS = ["Lab report", "Discharge summary", "Prescription", "Imaging report", "Other"]
_REFERENCE_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,63}$")


def patient_data_directory() -> Path:
    """Return a local application-data path outside the project/OneDrive tree."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "AIHealthRiskScoring"


def database_path() -> Path:
    return patient_data_directory() / "patient_reports.sqlite3"


def normalize_patient_reference(value: str) -> str:
    """Validate a non-identifying reference; names and contact details are rejected by UI guidance."""
    reference = str(value).strip().upper()
    if not _REFERENCE_PATTERN.fullmatch(reference):
        raise ValueError("Use 1–64 letters, numbers, periods, underscores, or hyphens for the patient reference.")
    return reference


def new_patient_reference() -> str:
    return f"PT-{uuid.uuid4().hex[:10].upper()}"


@contextmanager
def _connection(path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    db_path = Path(path) if path is not None else database_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS patient_reports (
                report_id TEXT PRIMARY KEY,
                patient_reference TEXT NOT NULL,
                report_type TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                content_type TEXT NOT NULL,
                uploaded_at_utc TEXT NOT NULL,
                byte_count INTEGER NOT NULL,
                file_bytes BLOB NOT NULL
            )"""
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_patient_reports_reference_time "
            "ON patient_reports(patient_reference, uploaded_at_utc DESC)"
        )
        connection.commit()
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def save_patient_report(
    patient_reference: str,
    report_type: str,
    original_filename: str,
    content_type: str,
    file_bytes: bytes,
    *,
    path: str | Path | None = None,
) -> str:
    """Persist one approved-format report file and return its opaque record ID."""
    reference = normalize_patient_reference(patient_reference)
    suffix = Path(original_filename).suffix.lower().lstrip(".")
    if suffix not in ALLOWED_REPORT_TYPES:
        raise ValueError(f"Unsupported report type '.{suffix}'.")
    if not file_bytes:
        raise ValueError("The selected report is empty.")
    if len(file_bytes) > MAX_REPORT_BYTES:
        raise ValueError("The report exceeds the 20 MB upload limit.")
    filename = Path(original_filename.replace("\\", "/")).name
    filename = "".join(char for char in filename if char.isprintable())[:255]
    if not filename:
        filename = f"patient_report.{suffix}"
    mime = content_type or "application/octet-stream"
    record_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connection(path) as connection:
        connection.execute(
            """INSERT INTO patient_reports
            (report_id, patient_reference, report_type, original_filename,
             content_type, uploaded_at_utc, byte_count, file_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (record_id, reference, report_type, filename, mime, timestamp, len(file_bytes), sqlite3.Binary(file_bytes)),
        )
    return record_id


def list_patient_reports(
    patient_reference: str, *, path: str | Path | None = None
) -> list[dict[str, object]]:
    reference = normalize_patient_reference(patient_reference)
    with _connection(path) as connection:
        rows = connection.execute(
            """SELECT report_id, patient_reference, report_type, original_filename,
                      content_type, uploaded_at_utc, byte_count
               FROM patient_reports WHERE patient_reference = ?
               ORDER BY uploaded_at_utc DESC, report_id DESC""",
            (reference,),
        ).fetchall()
    return [dict(row) for row in rows]


def read_patient_report(
    report_id: str, *, path: str | Path | None = None
) -> dict[str, object] | None:
    with _connection(path) as connection:
        row = connection.execute(
            """SELECT report_id, patient_reference, report_type, original_filename,
                      content_type, uploaded_at_utc, byte_count, file_bytes
               FROM patient_reports WHERE report_id = ?""",
            (report_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def delete_patient_report(report_id: str, *, path: str | Path | None = None) -> bool:
    with _connection(path) as connection:
        cursor = connection.execute("DELETE FROM patient_reports WHERE report_id = ?", (report_id,))
    return cursor.rowcount > 0


def count_patient_reports(*, path: str | Path | None = None) -> int:
    db_path = Path(path) if path is not None else database_path()
    if not db_path.is_file():
        return 0
    with _connection(db_path) as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM patient_reports").fetchone()
    return int(row["count"])
