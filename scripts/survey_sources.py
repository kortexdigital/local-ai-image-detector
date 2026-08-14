"""Verify every dataset in the source registry against the HuggingFace rows API.

Run this before changing training/datasets/sources.py, and rerun it whenever a
download starts failing. It reports, per dataset: whether the rows endpoint
answers, how many rows exist, and which column holds the image.

Existing on the Hub is not sufficient. Several datasets that resolve fine with
`HfApi.dataset_info` return HTTP 500 or 501 from the rows endpoint and cannot
be sampled this way. Others were built around loading scripts, which recent
versions of `datasets` no longer execute.

Usage:
    .venv/bin/python scripts/survey_sources.py
"""
from __future__ import annotations

import concurrent.futures as futures
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.datasets.sources import SOURCES  # noqa: E402

ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
TIMEOUT_SECONDS = 40


def probe(source) -> tuple[str, str, str | None, int | None, list[str]]:
    query = urllib.parse.urlencode(
        {
            "dataset": source.repo_id,
            "config": "default",
            "split": source.split,
            "offset": 0,
            "length": 1,
        }
    )
    try:
        with urllib.request.urlopen(f"{ROWS_ENDPOINT}?{query}", timeout=TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        return source.key, f"HTTP{exc.code}", None, None, []
    except Exception as exc:  # noqa: BLE001 - a survey must not abort on one row
        return source.key, type(exc).__name__, None, None, []

    row = payload["rows"][0]["row"]
    image_columns = [k for k, v in row.items() if isinstance(v, dict) and "src" in v]
    return (
        source.key,
        "OK",
        source.repo_id,
        payload.get("num_rows_total"),
        image_columns,
    )


def main() -> int:
    problems = 0
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(probe, SOURCES))

    by_key = {s.key: s for s in SOURCES}
    print(f"{'status':<9} {'key':<14} {'generator':<20} {'rows':>9}  image column")
    for key, status, _repo, total, columns in sorted(results):
        source = by_key[key]
        column = columns[0] if columns else "-"
        print(f"{status:<9} {key:<14} {source.generator:<20} {str(total):>9}  {column}")

        if status != "OK":
            problems += 1
        elif source.image_column not in columns:
            print(f"          ^ MISMATCH: registry says {source.image_column!r}, API says {columns}")
            problems += 1
        elif total is not None and total < source.target_count:
            print(f"          ^ TOO SMALL: target_count {source.target_count} exceeds {total} rows")
            problems += 1

    print()
    print(f"{len(results) - problems}/{len(results)} sources verified")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
