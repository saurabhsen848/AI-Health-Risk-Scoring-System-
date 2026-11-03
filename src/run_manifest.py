"""Build a provenance record for a completed NHANES preprocessing run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(
    *,
    project_root: Path,
    started_at: datetime,
    completed_at: datetime,
    source_paths: list[Path],
    script_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Describe inputs, code, runtime, and the output of a completed run."""
    return {
        "schema_version": 1,
        "cycle": "NHANES August 2021-August 2023",
        "started_at_utc": started_at.astimezone(timezone.utc).isoformat(),
        "completed_at_utc": completed_at.astimezone(timezone.utc).isoformat(),
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "dependencies": {
            package: _installed_version(package)
            for package in ("pandas", "numpy", "matplotlib")
        },
        "source_files": [
            {
                "path": str(path.relative_to(project_root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
        "script_files": [
            {
                "path": str(path.relative_to(project_root)),
                "sha256": sha256_file(path),
            }
            for path in script_paths
        ],
        "processed_output": {
            "path": str(output_path.relative_to(project_root)),
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
    }


def write_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a completed-run manifest as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
