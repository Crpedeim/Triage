"""
rag/fetch_guidelines.py — download the WHO guideline PDFs listed in
rag/guidelines/sources.json from WHO IRIS and verify their sha256.

    python -m rag.fetch_guidelines            # download missing / mismatched files
    python -m rag.fetch_guidelines --check    # verify only, download nothing

Run at Docker build time. The PDFs are not committed: the two IMCI documents
are "All rights reserved", so this repository points to WHO instead of
redistributing them. A file whose hash does not match is never used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

GUIDELINES_DIR = Path(__file__).parent / "guidelines"
MANIFEST = GUIDELINES_DIR / "sources.json"
USER_AGENT = "TriageAI-guideline-fetch/1.0 (+https://github.com/Crpedeim/Triage)"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_valid(path: Path, expected_sha256: str) -> bool:
    return path.exists() and sha256_of(path) == expected_sha256


def download(url: str, dest: Path, attempts: int = 3) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            tmp.replace(dest)
            return
        except Exception as e:  # network errors: retry with backoff
            print(f"  attempt {attempt}/{attempts} failed: {e}")
            if attempt == attempts:
                raise
            time.sleep(5 * attempt)


def fetch_all(check_only: bool = False) -> bool:
    GUIDELINES_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for doc in load_manifest()["documents"]:
        dest = GUIDELINES_DIR / doc["file"]
        if is_valid(dest, doc["sha256"]):
            print(f"[fetch] OK       {doc['file']} (sha256 matches)")
            continue
        if check_only:
            print(f"[fetch] MISSING  {doc['file']}")
            ok = False
            continue
        print(f"[fetch] download {doc['file']} <- {doc['pdf_url']}")
        download(doc["pdf_url"], dest)
        if not is_valid(dest, doc["sha256"]):
            got = sha256_of(dest)
            dest.unlink(missing_ok=True)
            print(f"[fetch] HASH MISMATCH {doc['file']}: expected {doc['sha256']}, got {got}")
            ok = False
        else:
            print(f"[fetch] OK       {doc['file']} ({dest.stat().st_size} bytes)")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="verify only")
    args = parser.parse_args()
    sys.exit(0 if fetch_all(check_only=args.check) else 1)
