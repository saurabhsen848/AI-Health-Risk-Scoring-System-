"""Build the NHANES BPQ020 complete-case dataset and descriptive EDA outputs."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nhanes_data import (  # noqa: E402
    FILES,
    load_merge_sources,
    prepare_dataset,
    run_eda,
    write_config,
)
from src.run_manifest import build_run_manifest, write_run_manifest  # noqa: E402


RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "nhanes_2021_2023"
CONFIG_PATH = PROJECT_ROOT / "config" / "nhanes_2021_2023_bpq020.json"


def main() -> None:
    started_at = datetime.now(timezone.utc)
    absent = [filename for filename in FILES.values() if not (RAW_DIR / filename).is_file()]
    if absent:
        print("Missing raw NHANES files:", ", ".join(absent))
        print("Run: python scripts/download_nhanes.py")
        raise SystemExit(2)

    merged, _ = load_merge_sources(RAW_DIR)
    processed, report = prepare_dataset(RAW_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DIR / "nhanes_2021_2023_bpq020.csv"
    processed.to_csv(processed_path, index=False)
    write_config(CONFIG_PATH)
    run_eda(merged, processed, report, REPORT_DIR)
    completed_at = datetime.now(timezone.utc)
    manifest_path = REPORT_DIR / "preprocessing_run_manifest.json"
    script_paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "nhanes_data.py",
        PROJECT_ROOT / "src" / "run_manifest.py",
    ]
    run_manifest = build_run_manifest(
        project_root=PROJECT_ROOT,
        started_at=started_at,
        completed_at=completed_at,
        source_paths=[RAW_DIR / filename for filename in FILES.values()],
        script_paths=script_paths,
        output_path=processed_path,
    )
    write_run_manifest(manifest_path, run_manifest)
    print(json.dumps(report, indent=2))
    print(f"Processed dataset: {processed_path}")
    print(f"Configuration: {CONFIG_PATH}")
    print(f"EDA outputs: {REPORT_DIR}")
    print(f"Run provenance: {manifest_path}")
    print("No ML model was trained.")


if __name__ == "__main__":
    main()
