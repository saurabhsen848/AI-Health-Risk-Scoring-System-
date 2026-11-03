"""Conservative text extraction for common CBC values in selectable-text PDFs."""

from __future__ import annotations

from io import BytesIO
import math
import re

import pandas as pd
from pypdf import PdfReader


class PDFExtractionError(ValueError):
    """The PDF could not be read or contains no extractable text."""


_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_RANGE_RE = re.compile(
    rf"(?<![\w.])(?P<low>{_NUMBER})\s*(?:-|–|—|\bto\b)\s*(?P<high>{_NUMBER})(?![\w.])",
    re.IGNORECASE,
)
_RESULT_RE = re.compile(rf"^\s*[:=]?\s*(?P<result>{_NUMBER})(?=$|\s|[,(])")
_LEADING_UNIT_RESULT_RE = re.compile(
    rf"^\s*(?:\(|\[)?(?P<unit>(?:[xX×])?[A-Za-z0-9µμ/%^.*_-]+)(?:\)|\])?\s+"
    rf"(?P<result>{_NUMBER})(?=$|\s|[,(])"
)
_LABELS = (
    ("Hemoglobin", re.compile(r"\b(?:hemoglobin|haemoglobin|hgb|hb)\b", re.IGNORECASE)),
    ("WBC", re.compile(r"\b(?:wbc|white\s+blood\s+cell(?:\s+count)?)\b", re.IGNORECASE)),
    ("RBC", re.compile(r"\b(?:rbc|red\s+blood\s+cell(?:\s+count)?)\b", re.IGNORECASE)),
    ("Platelets", re.compile(r"\b(?:platelets?|plt)\b", re.IGNORECASE)),
    ("Hematocrit", re.compile(r"\b(?:hematocrit|haematocrit|hct)\b", re.IGNORECASE)),
)
_UNIT_RE = re.compile(r"(?:[x×])?[A-Za-z0-9µμ/%^.*_-]+", re.IGNORECASE)
_NON_UNIT_WORDS = {"h", "l", "high", "low", "ref", "reference", "range", "flag"}
_COLUMNS = ["Biomarker", "Result", "Unit", "Reference Range", "Status"]


def _parse_line(analyte: str, remainder: str) -> dict[str, str]:
    """Parse a result and optional unit/range from the text following a label."""
    result_match = _RESULT_RE.search(remainder)
    leading_unit = ""
    if result_match is None:
        result_match = _LEADING_UNIT_RESULT_RE.search(remainder)
        if result_match:
            leading_unit = result_match.group("unit")
    result = result_match.group("result") if result_match else "Result unavailable"
    rest = remainder[result_match.end():] if result_match else remainder

    range_match = _RANGE_RE.search(rest)
    if range_match:
        low = float(range_match.group("low").replace(",", ""))
        high = float(range_match.group("high").replace(",", ""))
        valid_range = math.isfinite(low) and math.isfinite(high) and low <= high
    else:
        low = high = None
        valid_range = False

    if range_match:
        unit_text = leading_unit + " " + rest[:range_match.start()] + " " + rest[range_match.end():]
    else:
        unit_text = rest
    unit = ""
    for candidate in _UNIT_RE.findall(unit_text):
        if (
            candidate.lower() not in _NON_UNIT_WORDS
            and (re.search(r"[0-9/%^µμ]", candidate) or candidate.lower() in {"fl", "pg"})
        ):
            unit = candidate
            break

    if result == "Result unavailable":
        status = "Result unavailable"
    elif not valid_range:
        status = "Range unavailable"
    else:
        numeric_result = float(result.replace(",", ""))
        status = (
            "Low" if numeric_result < low else
            "High" if numeric_result > high else
            "Within range"
        )

    reference = f"{low:g} - {high:g}" if valid_range else "Range unavailable"
    return {
        "Biomarker": analyte,
        "Result": result,
        "Unit": unit or "Unit unavailable",
        "Reference Range": reference,
        "Status": status,
    }


def parse_cbc_text(text: str) -> pd.DataFrame:
    """Extract only explicit common CBC labels and their same-line values."""
    results: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        for analyte, pattern in _LABELS:
            match = pattern.search(line)
            if match:
                results.setdefault(analyte, _parse_line(analyte, line[match.end():]))
                break
    return pd.DataFrame(list(results.values()), columns=_COLUMNS)


def extract_cbc_pdf(file_bytes: bytes) -> pd.DataFrame:
    """Read selectable text from a PDF and extract available CBC rows.

    No OCR is attempted. Missing or image-only PDF text is reported to the caller.
    """
    try:
        reader = PdfReader(BytesIO(file_bytes), strict=False)
        if reader.is_encrypted:
            raise PDFExtractionError("This PDF is password-protected. Upload an unlocked report.")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except PDFExtractionError:
        raise
    except Exception as exc:
        raise PDFExtractionError("This file could not be read as a valid PDF.") from exc

    if not text.strip():
        raise PDFExtractionError(
            "No selectable text was found. This may be a scanned PDF; OCR is not performed, so no values were extracted."
        )
    return parse_cbc_text(text)
