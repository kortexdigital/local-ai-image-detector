"""Read and write the cache manifest.

The manifest is the contract between the download stage and every later
stage: one row per cached image, carrying the label and the generator that
the split logic partitions on.
"""
from __future__ import annotations

import csv
from dataclasses import astuple, dataclass
from pathlib import Path

FIELDNAMES = ("sha256", "relpath", "source_key", "generator", "label")


@dataclass(frozen=True)
class ManifestRow:
    sha256: str
    relpath: str
    source_key: str
    generator: str
    label: int


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDNAMES)
        for row in rows:
            writer.writerow(astuple(row))


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            ManifestRow(
                sha256=r["sha256"],
                relpath=r["relpath"],
                source_key=r["source_key"],
                generator=r["generator"],
                label=int(r["label"]),
            )
            for r in reader
        ]
