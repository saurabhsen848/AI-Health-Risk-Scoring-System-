"""Download only official NHANES XPT files; existing raw files are never replaced."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nhanes_data import FILES, RAW_BASE_URL


RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        destination = RAW_DIR / filename
        if destination.exists():
            print(f"Keeping existing raw file unchanged: {destination}")
            continue
        url = f"{RAW_BASE_URL}/{filename}"
        print(f"Downloading {url}")
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        if not payload.startswith(b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!"):
            raise ValueError(f"Downloaded content does not look like a SAS XPORT file: {filename}")
        destination.write_bytes(payload)
        print(f"Saved {destination} ({len(payload):,} bytes)")

    manifest = {
        filename: {
            "source_url": f"{RAW_BASE_URL}/{filename}",
            "size_bytes": (RAW_DIR / filename).stat().st_size,
            "sha256": hashlib.sha256((RAW_DIR / filename).read_bytes()).hexdigest(),
        }
        for filename in FILES.values()
    }
    (RAW_DIR / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote source URL, size, and SHA-256 manifest to {RAW_DIR / 'SOURCE_MANIFEST.json'}")


if __name__ == "__main__":
    main()
