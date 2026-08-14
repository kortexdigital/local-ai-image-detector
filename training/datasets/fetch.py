"""Download source datasets and write them into the local cache.

Sampling happens at the parquet-shard level rather than row by row. The rows
endpoint returns one image per row and rate-limits unauthenticated callers,
so pulling twenty thousand images that way costs twenty thousand requests and
stops halfway with HTTP 429. Fetching a whole shard costs one request and
yields thousands of images, which keeps the whole download inside a few dozen
requests with no token required.

Shards are downloaded one at a time and deleted after extraction, so peak
disk usage is one shard rather than the whole dataset.
"""
from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from training.datasets.manifest import ManifestRow, write_manifest
from training.datasets.normalize import CacheError, normalize_for_cache
from training.datasets.sources import SOURCES, Source

log = logging.getLogger(__name__)

PARQUET_ENDPOINT = "https://datasets-server.huggingface.co/parquet"
USER_AGENT = "ai-image-detector/0.1"
HTTP_TIMEOUT = 120
MAX_SHARDS_PER_SOURCE = 6


def image_bytes_from_cell(cell) -> bytes | None:
    """Pull raw bytes out of one parquet cell holding an image."""
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, dict):
        value = cell.get("bytes")
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
    return None


def cache_image(raw: bytes, source: Source, cache_dir: Path) -> ManifestRow | None:
    """Normalize and store one image. Returns None if unusable or duplicate."""
    try:
        normalized = normalize_for_cache(raw)
    except CacheError:
        return None

    digest = hashlib.sha256(normalized).hexdigest()
    relpath = f"{source.generator}/{digest}.jpg"
    target = cache_dir / relpath
    if target.exists():
        return None

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(normalized)
    return ManifestRow(
        sha256=digest,
        relpath=relpath,
        source_key=source.key,
        generator=source.generator,
        label=source.label,
    )


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        return json.load(response)


def list_shards(source: Source) -> list[str]:
    """Return parquet shard URLs for the source's split."""
    query = urllib.parse.urlencode({"dataset": source.repo_id})
    payload = _get_json(f"{PARQUET_ENDPOINT}?{query}")
    return [
        entry["url"]
        for entry in payload.get("parquet_files", [])
        if entry.get("split") == source.split
    ]


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        with target.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)


def _harvest_shard(
    shard_path: Path, source: Source, cache_dir: Path, needed: int, rng
) -> list[ManifestRow]:
    """Pull up to `needed` images out of one shard.

    Reads a row group at a time rather than the whole column. A shard holds
    several hundred megabytes of encoded images, and materializing all of it
    as Python objects at once costs multiples of that in memory.
    """
    parquet = pq.ParquetFile(shard_path)
    group_order = rng.permutation(parquet.num_row_groups)

    rows: list[ManifestRow] = []
    for group_index in group_order:
        if len(rows) >= needed:
            break
        table = parquet.read_row_group(int(group_index), columns=[source.image_column])
        cells = table.column(source.image_column).to_pylist()
        for cell_index in rng.permutation(len(cells)):
            if len(rows) >= needed:
                break
            raw = image_bytes_from_cell(cells[int(cell_index)])
            if raw is None:
                continue
            row = cache_image(raw, source, cache_dir)
            if row is not None:
                rows.append(row)
        del cells, table
    return rows


def fetch_source(source: Source, cache_dir: Path, seed: int) -> list[ManifestRow]:
    # zlib.crc32 rather than hash(): str hashing is salted per process unless
    # PYTHONHASHSEED is pinned, which would make runs unreproducible.
    rng = np.random.default_rng(seed + zlib.crc32(source.key.encode("utf-8")))
    try:
        shards = list_shards(source)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        log.error("source %s: cannot list shards (%s)", source.key, exc)
        return []

    if not shards:
        log.error("source %s: no parquet shards for split %s", source.key, source.split)
        return []

    # A random shard order keeps us from always sampling the same slice of a
    # dataset, which would correlate every image we cache from it.
    shards_ordered = [shards[int(i)] for i in rng.permutation(len(shards))]

    collected: list[ManifestRow] = []
    for shard_url in shards_ordered[:MAX_SHARDS_PER_SOURCE]:
        if len(collected) >= source.target_count:
            break
        with tempfile.TemporaryDirectory() as tmp:
            shard_path = Path(tmp) / "shard.parquet"
            try:
                _download(shard_url, shard_path)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                log.warning("source %s: shard download failed (%s)", source.key, exc)
                continue
            try:
                collected.extend(
                    _harvest_shard(
                        shard_path,
                        source,
                        cache_dir,
                        source.target_count - len(collected),
                        rng,
                    )
                )
            except Exception:
                log.exception("source %s: shard unreadable", source.key)
                continue
        log.info(
            "source %s: %d/%d cached", source.key, len(collected), source.target_count
        )

    if len(collected) < source.target_count:
        log.warning(
            "source %s: only %d of %d requested images were cached",
            source.key,
            len(collected),
            source.target_count,
        )
    return collected


def fetch_all(cache_dir: Path, manifest_path: Path, seed: int) -> list[ManifestRow]:
    all_rows: list[ManifestRow] = []
    for source in SOURCES:
        try:
            all_rows.extend(fetch_source(source, cache_dir, seed))
        except Exception:
            log.exception("source %s failed, continuing", source.key)
        write_manifest(all_rows, manifest_path)
    log.info("cached %d images in total", len(all_rows))
    return all_rows
