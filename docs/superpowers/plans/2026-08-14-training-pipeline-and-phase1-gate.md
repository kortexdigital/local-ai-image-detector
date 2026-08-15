# Training Pipeline and Phase 1 Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline pipeline that turns public image datasets into a trained, calibrated detection head, and measure its balanced accuracy on generators the training never saw.

**Architecture:** A frozen vision backbone runs as an ONNX graph under onnxruntime. Preprocessing (resize plus normalization) is itself a small ONNX graph, so the exact same numerical path will later run in the browser. Features are extracted once, cached to disk, and a shallow head is trained on them with scikit-learn. Calibration maps the head's optimal decision threshold onto 0.65.

**Tech Stack:** Python 3.12, onnxruntime (CPU execution provider), numpy, Pillow, scikit-learn, onnx, huggingface_hub, datasets, pytest.

**Spec:** `docs/superpowers/specs/2026-08-14-ai-image-detector-design.md`

## Global Constraints

- Python 3.12 exactly. Create the venv with `/opt/homebrew/bin/python3.12`. Python 3.14 lacks wheels for onnxruntime and scikit-learn.
- onnxruntime always runs with `providers=["CPUExecutionProvider"]`. Never CoreML or any accelerated provider. Determinism across machines matters more than speed here, and the browser side has no CoreML.
- ONNX graphs we author target **opset 17**. Resize uses opset-13 semantics with `antialias` absent, because ONNX Runtime Web's coverage of `antialias=1` is unreliable.
- Every random operation takes an explicit seed. No bare `random` or `np.random` global state.
- Repository license is MIT. Before any model weight is downloaded or committed, its license is checked and recorded in `models/LICENSES.md`.
- All public-facing artifacts (README, code comments, commit messages, docstrings) are written in **English**. Design and plan documents stay in Portuguese.
- No public artifact in this repository discusses funding or compensation of any kind. This is a hard rule, not a style preference.
- Disk budget for Phase 1 is 15 GB total under `data/`. `data/` is gitignored.
- Cached images are always re-encoded through one single code path regardless of class. Real and synthetic images must carry identical re-encode signatures, or the head learns the cache format instead of the task.

---

### Task 1: Project scaffolding and test harness

**Files:**
- Create: `training/__init__.py`
- Create: `training/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `models/LICENSES.md`

**Interfaces:**
- Consumes: nothing.
- Produces: `training.config.Config` dataclass with fields `data_dir: Path`, `cache_dir: Path`, `features_dir: Path`, `models_dir: Path`, `seed: int`, `cache_max_side: int`, `cache_jpeg_quality: int`, `image_size: int`, `decision_confidence: float`. Module-level `CONFIG: Config` built from environment overrides with defaults. Also `training.config.HELD_OUT_GENERATORS: tuple[str, ...]`.

- [ ] **Step 1: Create the virtualenv and install dependencies**

```bash
cd ~/Pessoal/projetos/ai-image-detector
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install onnx==1.17.0 onnxruntime==1.20.1 numpy pillow scikit-learn \
  huggingface_hub datasets pytest
.venv/bin/python -c "import onnxruntime, sklearn, PIL, onnx; print(onnxruntime.__version__)"
```

Expected: prints `1.20.1` with no import error.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
from training.config import CONFIG, Config, HELD_OUT_GENERATORS


def test_config_defaults_are_absolute_paths():
    assert isinstance(CONFIG, Config)
    for p in (CONFIG.data_dir, CONFIG.cache_dir, CONFIG.features_dir, CONFIG.models_dir):
        assert isinstance(p, Path)
        assert p.is_absolute()


def test_decision_confidence_is_the_evaluation_threshold():
    assert CONFIG.decision_confidence == 0.65


def test_cache_settings_match_the_disk_budget():
    assert CONFIG.cache_max_side == 512
    assert CONFIG.cache_jpeg_quality == 95


def test_held_out_generators_are_declared_and_non_empty():
    assert len(HELD_OUT_GENERATORS) >= 3
    assert all(isinstance(g, str) and g for g in HELD_OUT_GENERATORS)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training'`

- [ ] **Step 4: Write minimal implementation**

```python
# training/config.py
"""Central configuration for the offline training pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Generators deliberately excluded from training. Balanced accuracy on these is
# the only honest estimate of performance on an unseen evaluation set.
HELD_OUT_GENERATORS: tuple[str, ...] = ("flux", "midjourney-v6", "stylegan3")


@dataclass(frozen=True)
class Config:
    data_dir: Path
    cache_dir: Path
    features_dir: Path
    models_dir: Path
    seed: int
    cache_max_side: int
    cache_jpeg_quality: int
    image_size: int
    decision_confidence: float


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


CONFIG = Config(
    data_dir=_path_from_env("AIID_DATA_DIR", REPO_ROOT / "data"),
    cache_dir=_path_from_env("AIID_CACHE_DIR", REPO_ROOT / "data" / "cache"),
    features_dir=_path_from_env("AIID_FEATURES_DIR", REPO_ROOT / "data" / "features"),
    models_dir=_path_from_env("AIID_MODELS_DIR", REPO_ROOT / "models"),
    seed=20260814,
    cache_max_side=512,
    cache_jpeg_quality=95,
    image_size=224,
    decision_confidence=0.65,
)
```

```python
# training/__init__.py
```

```python
# tests/__init__.py
```

```toml
# pyproject.toml
[project]
name = "ai-image-detector-training"
version = "0.1.0"
requires-python = ">=3.12,<3.13"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

```make
# Makefile
PY := .venv/bin/python
PYTEST := .venv/bin/pytest

.PHONY: test
test:
	$(PYTEST) -v

.PHONY: clean-features
clean-features:
	rm -rf data/features
```

```markdown
<!-- models/LICENSES.md -->
# Model weight licenses

Every weight file bundled or downloaded by this project is recorded here with
its upstream source and license. The project itself is MIT, so only weights
under a compatible permissive license may be used.

| Artifact | Source | License | Verified on |
|---|---|---|---|
| (none yet) | | | |
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml Makefile training/ tests/ models/LICENSES.md
git commit -m "chore: scaffold training package with config and test harness"
```

---

### Task 2: Cache normalizer

Every image entering the cache passes through one function. This is what stops the head from learning "PNG means synthetic" instead of learning the task.

**Files:**
- Create: `training/datasets/__init__.py`
- Create: `training/datasets/normalize.py`
- Create: `tests/test_normalize.py`

**Interfaces:**
- Consumes: `training.config.CONFIG`.
- Produces: `training.datasets.normalize.normalize_for_cache(raw: bytes) -> bytes` returning JPEG bytes. Also `training.datasets.normalize.CacheError(Exception)` raised for undecodable input.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_normalize.py
import io

import pytest
from PIL import Image

from training.config import CONFIG
from training.datasets.normalize import CacheError, normalize_for_cache


def _encode(im: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()


def test_png_input_becomes_jpeg():
    im = Image.new("RGB", (300, 200), (10, 120, 240))
    out = normalize_for_cache(_encode(im, "PNG"))
    assert Image.open(io.BytesIO(out)).format == "JPEG"


def test_large_image_is_downscaled_to_max_side():
    im = Image.new("RGB", (2000, 1000), (5, 5, 5))
    out = Image.open(io.BytesIO(normalize_for_cache(_encode(im, "JPEG"))))
    assert max(out.size) == CONFIG.cache_max_side
    assert out.size == (CONFIG.cache_max_side, CONFIG.cache_max_side // 2)


def test_small_image_is_never_upscaled():
    im = Image.new("RGB", (64, 48), (200, 100, 50))
    out = Image.open(io.BytesIO(normalize_for_cache(_encode(im, "PNG"))))
    assert out.size == (64, 48)


def test_grayscale_and_rgba_are_converted_to_rgb():
    for mode, fmt in (("L", "PNG"), ("RGBA", "PNG")):
        im = Image.new(mode, (50, 50))
        out = Image.open(io.BytesIO(normalize_for_cache(_encode(im, fmt))))
        assert out.mode == "RGB"


def test_normalizing_twice_is_stable_in_size():
    im = Image.new("RGB", (800, 600), (33, 66, 99))
    once = normalize_for_cache(_encode(im, "PNG"))
    twice = normalize_for_cache(once)
    assert Image.open(io.BytesIO(once)).size == Image.open(io.BytesIO(twice)).size


def test_undecodable_bytes_raise_cache_error():
    with pytest.raises(CacheError):
        normalize_for_cache(b"this is not an image")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.datasets'`

- [ ] **Step 3: Write minimal implementation**

```python
# training/datasets/normalize.py
"""Single entry point for writing an image into the local cache.

Both real and synthetic images pass through this exact function, so neither
class carries a re-encoding signature the classifier could exploit.
"""
from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

from training.config import CONFIG


class CacheError(Exception):
    """Raised when input bytes cannot be decoded as an image."""


def normalize_for_cache(raw: bytes) -> bytes:
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise CacheError("input bytes are not a decodable image") from exc

    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")

    width, height = im.size
    longest = max(width, height)
    if longest > CONFIG.cache_max_side:
        scale = CONFIG.cache_max_side / longest
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        im = im.resize(target, Image.BICUBIC)

    out = io.BytesIO()
    im.save(
        out,
        format="JPEG",
        quality=CONFIG.cache_jpeg_quality,
        subsampling=0,
        optimize=False,
    )
    return out.getvalue()
```

```python
# training/datasets/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_normalize.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add training/datasets/ tests/test_normalize.py
git commit -m "feat: add cache normalizer with a single re-encode path for all classes"
```

---

### Task 3: Source registry and manifest

Dataset repository identifiers on HuggingFace move and get renamed. This task resolves them against the live Hub and writes down what actually exists, instead of hardcoding identifiers that may be wrong.

**Files:**
- Create: `training/datasets/sources.py`
- Create: `training/datasets/manifest.py`
- Create: `tests/test_manifest.py`
- Create: `scripts/resolve_sources.py`

**Interfaces:**
- Consumes: `training.config.CONFIG`, `training.config.HELD_OUT_GENERATORS`.
- Produces:
  - `training.datasets.sources.Source` dataclass: `key: str`, `repo_id: str`, `label: int` (0 real, 1 synthetic), `generator: str` (`"real"` for real sources), `split: str`, `image_column: str`, `target_count: int`.
  - `training.datasets.sources.SOURCES: tuple[Source, ...]`.
  - `training.datasets.manifest.ManifestRow` dataclass: `sha256: str`, `relpath: str`, `source_key: str`, `generator: str`, `label: int`.
  - `training.datasets.manifest.write_manifest(rows: list[ManifestRow], path: Path) -> None`
  - `training.datasets.manifest.read_manifest(path: Path) -> list[ManifestRow]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py
from pathlib import Path

from training.config import HELD_OUT_GENERATORS
from training.datasets.manifest import ManifestRow, read_manifest, write_manifest
from training.datasets.sources import SOURCES


def test_manifest_roundtrips(tmp_path: Path):
    rows = [
        ManifestRow("a" * 64, "real/coco/aaa.jpg", "coco", "real", 0),
        ManifestRow("b" * 64, "fake/flux/bbb.jpg", "flux", "flux", 1),
    ]
    path = tmp_path / "manifest.csv"
    write_manifest(rows, path)
    assert read_manifest(path) == rows


def test_sources_cover_both_classes():
    assert any(s.label == 0 for s in SOURCES)
    assert any(s.label == 1 for s in SOURCES)


def test_source_keys_are_unique():
    keys = [s.key for s in SOURCES]
    assert len(keys) == len(set(keys))


def test_real_sources_use_the_real_generator_marker():
    for s in SOURCES:
        assert (s.generator == "real") == (s.label == 0)


def test_every_held_out_generator_has_a_source():
    available = {s.generator for s in SOURCES}
    for generator in HELD_OUT_GENERATORS:
        assert generator in available, f"no source provides held-out generator {generator}"


def test_synthetic_sources_span_multiple_generators():
    synthetic = {s.generator for s in SOURCES if s.label == 1}
    assert len(synthetic) >= 6, "cross-generator generalization needs generator diversity"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.datasets.sources'`

- [ ] **Step 3: Write the resolver script and run it to discover real repository identifiers**

```python
# scripts/resolve_sources.py
"""Probe candidate HuggingFace datasets and report which ones are usable.

Run this before editing training/datasets/sources.py. It prints a table of
candidates with availability, so the source registry records what exists
rather than what we assumed exists.
"""
from __future__ import annotations

import sys

from huggingface_hub import HfApi

CANDIDATES = [
    # (candidate repo_id, note)
    ("detection-datasets/coco", "real photos"),
    ("nateraw/open-images-subset", "real photos"),
    ("nielsr/CelebA-faces", "real faces"),
    ("eurecom-ds/ffhq", "real faces"),
    ("poloclub/diffusiondb", "SD 1.x synthetic"),
    ("elsaEU/ELSA_D3", "multi-generator synthetic"),
    ("InfImagine/FakeImageDataset", "multi-generator synthetic"),
    ("Hemg/AI-Generated-vs-Real-Images-Datasets", "mixed"),
    ("dragonintelligence/CIFAKE-image-dataset", "mixed, low resolution"),
    ("yuvalkirstain/midjourney", "midjourney synthetic"),
    ("k-mktr/improved-flux-prompts", "flux synthetic"),
    ("OpenDatasets/dalle-3-dataset", "dalle-3 synthetic"),
]


def main() -> int:
    api = HfApi()
    print(f"{'repo_id':<50} {'ok':<5} note")
    for repo_id, note in CANDIDATES:
        try:
            api.dataset_info(repo_id)
            ok = "yes"
        except Exception:
            ok = "no"
        print(f"{repo_id:<50} {ok:<5} {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Run: `.venv/bin/python scripts/resolve_sources.py`

Expected: a table. Record which repository identifiers report `yes`. Search the Hub for replacements for any that report `no`, using `HfApi().list_datasets(search=...)`. Do not proceed to Step 4 with an unverified identifier.

**Acceptance criteria for the resolved set, all of which the tests in Step 1 enforce:** at least six distinct synthetic generators, at least three distinct real sources, and a working source for each of the three held-out generators in `HELD_OUT_GENERATORS`. If a held-out generator has no available source, change `HELD_OUT_GENERATORS` in `training/config.py` to name generators that do have sources, and keep at least three of them.

- [ ] **Step 4: Write the implementation using the verified identifiers**

```python
# training/datasets/sources.py
"""Registry of dataset sources.

Every repo_id here was verified against the HuggingFace Hub by
scripts/resolve_sources.py. Do not add an entry without running that script.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    key: str
    repo_id: str
    label: int  # 0 real, 1 synthetic
    generator: str  # "real" for real sources, otherwise the generator name
    split: str
    image_column: str
    target_count: int


# NOTE: replace the repo_id values below with the identifiers verified in
# Step 3. The structure and the invariants stay the same.
SOURCES: tuple[Source, ...] = (
    Source("coco", "detection-datasets/coco", 0, "real", "val", "image", 3000),
    Source("openimages", "nateraw/open-images-subset", 0, "real", "train", "image", 3000),
    Source("faces_real", "nielsr/CelebA-faces", 0, "real", "train", "image", 2000),
    Source("diffusiondb", "poloclub/diffusiondb", 1, "stable-diffusion-1", "train", "image", 3000),
    Source("elsa_d3", "elsaEU/ELSA_D3", 1, "stable-diffusion-xl", "train", "image", 3000),
    Source("fakeimage_glide", "InfImagine/FakeImageDataset", 1, "glide", "train", "image", 1500),
    Source("fakeimage_sd21", "InfImagine/FakeImageDataset", 1, "stable-diffusion-2", "train", "image", 1500),
    Source("flux", "k-mktr/improved-flux-prompts", 1, "flux", "train", "image", 1500),
    Source("midjourney_v6", "yuvalkirstain/midjourney", 1, "midjourney-v6", "train", "image", 1500),
    Source("stylegan3", "InfImagine/FakeImageDataset", 1, "stylegan3", "train", "image", 1500),
)
```

```python
# training/datasets/manifest.py
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_manifest.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add training/datasets/sources.py training/datasets/manifest.py \
        tests/test_manifest.py scripts/resolve_sources.py
git commit -m "feat: add verified dataset source registry and cache manifest"
```

---

### Task 4: Dataset downloader

**Files:**
- Create: `training/datasets/fetch.py`
- Create: `tests/test_fetch.py`

**Interfaces:**
- Consumes: `Source`, `SOURCES`, `ManifestRow`, `write_manifest`, `normalize_for_cache`, `CacheError`.
- Produces: `training.datasets.fetch.fetch_source(source: Source, cache_dir: Path, seed: int) -> list[ManifestRow]` and `training.datasets.fetch.fetch_all(cache_dir: Path, manifest_path: Path, seed: int) -> list[ManifestRow]`. Also `training.datasets.fetch.cache_image(raw: bytes, source: Source, cache_dir: Path) -> ManifestRow | None`, which returns `None` when the image is undecodable or already cached.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetch.py
import hashlib
import io
from pathlib import Path

from PIL import Image

from training.datasets.fetch import cache_image
from training.datasets.sources import Source

SRC = Source("t", "some/repo", 1, "flux", "train", "image", 10)


def _png(color=(1, 2, 3), size=(40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_cache_image_writes_a_file_and_returns_a_row(tmp_path: Path):
    row = cache_image(_png(), SRC, tmp_path)
    assert row is not None
    assert (tmp_path / row.relpath).exists()
    assert row.label == 1
    assert row.generator == "flux"
    assert row.source_key == "t"


def test_relpath_is_partitioned_by_generator(tmp_path: Path):
    row = cache_image(_png(), SRC, tmp_path)
    assert row.relpath.startswith("flux/")
    assert row.relpath.endswith(".jpg")


def test_sha256_is_of_the_normalized_bytes_not_the_input(tmp_path: Path):
    raw = _png()
    row = cache_image(raw, SRC, tmp_path)
    stored = (tmp_path / row.relpath).read_bytes()
    assert row.sha256 == hashlib.sha256(stored).hexdigest()
    assert row.sha256 != hashlib.sha256(raw).hexdigest()


def test_duplicate_image_is_skipped(tmp_path: Path):
    raw = _png()
    assert cache_image(raw, SRC, tmp_path) is not None
    assert cache_image(raw, SRC, tmp_path) is None


def test_undecodable_image_is_skipped_not_raised(tmp_path: Path):
    assert cache_image(b"garbage", SRC, tmp_path) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.datasets.fetch'`

- [ ] **Step 3: Write minimal implementation**

```python
# training/datasets/fetch.py
"""Download source datasets and write them into the local cache."""
from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

from PIL import Image

from training.datasets.manifest import ManifestRow, write_manifest
from training.datasets.normalize import CacheError, normalize_for_cache
from training.datasets.sources import SOURCES, Source

log = logging.getLogger(__name__)


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


def _to_bytes(value) -> bytes | None:
    """Coerce a datasets image column value into raw bytes."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict) and value.get("bytes"):
        return value["bytes"]
    if isinstance(value, Image.Image):
        buf = io.BytesIO()
        value.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    return None


def fetch_source(source: Source, cache_dir: Path, seed: int) -> list[ManifestRow]:
    from datasets import load_dataset

    stream = load_dataset(
        source.repo_id, split=source.split, streaming=True
    ).shuffle(seed=seed, buffer_size=2000)

    rows: list[ManifestRow] = []
    for record in stream:
        if len(rows) >= source.target_count:
            break
        raw = _to_bytes(record.get(source.image_column))
        if raw is None:
            continue
        row = cache_image(raw, source, cache_dir)
        if row is not None:
            rows.append(row)
    log.info("cached %d images for source %s", len(rows), source.key)
    return rows


def fetch_all(cache_dir: Path, manifest_path: Path, seed: int) -> list[ManifestRow]:
    all_rows: list[ManifestRow] = []
    for source in SOURCES:
        try:
            all_rows.extend(fetch_source(source, cache_dir, seed))
        except Exception:
            log.exception("source %s failed, continuing", source.key)
    write_manifest(all_rows, manifest_path)
    return all_rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_fetch.py -v`
Expected: 5 passed

- [ ] **Step 5: Download the Phase 1 cache and check the disk budget**

```bash
.venv/bin/python -c "
import logging; logging.basicConfig(level=logging.INFO)
from training.config import CONFIG
from training.datasets.fetch import fetch_all
rows = fetch_all(CONFIG.cache_dir, CONFIG.data_dir / 'manifest.csv', CONFIG.seed)
print('cached rows:', len(rows))
"
du -sh data/cache
```

Expected: at least 15000 rows total, and `data/cache` under 15 GB. If a source failed, the log names it; fix its entry in `sources.py` and rerun. Reruns are safe because duplicates are skipped.

- [ ] **Step 6: Commit**

```bash
git add training/datasets/fetch.py tests/test_fetch.py
git commit -m "feat: add streaming dataset downloader with deduplicated cache"
```

---

### Task 5: Preprocessing ONNX graph

The resize and normalization live inside an ONNX graph so that Python and the browser execute bit-identical arithmetic. This is the single highest-value decision in the pipeline.

**Files:**
- Create: `training/features/__init__.py`
- Create: `training/features/preprocess_graph.py`
- Create: `tests/test_preprocess_graph.py`

**Interfaces:**
- Consumes: `training.config.CONFIG`.
- Produces:
  - `training.features.preprocess_graph.PreprocessSpec` dataclass: `mean: tuple[float, float, float]`, `std: tuple[float, float, float]`, `size: int`, `square_crop: bool`.
  - `training.features.preprocess_graph.CLIP_SPEC` and `SIGLIP_SPEC` constants.
  - `training.features.preprocess_graph.build_preprocess_graph(spec: PreprocessSpec, path: Path) -> Path` writing an ONNX file.
  - `training.features.preprocess_graph.center_square_crop(im: Image.Image) -> Image.Image`.

Graph contract: input `pixels`, uint8, shape `[1, H, W, 3]` with dynamic `H` and `W`, channel order RGB. Output `pixel_values`, float32, shape `[1, 3, size, size]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocess_graph.py
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from training.features.preprocess_graph import (
    CLIP_SPEC,
    SIGLIP_SPEC,
    build_preprocess_graph,
    center_square_crop,
)


def _session(spec, tmp_path: Path) -> ort.InferenceSession:
    path = build_preprocess_graph(spec, tmp_path / "pre.onnx")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_output_shape_and_dtype(tmp_path: Path):
    sess = _session(CLIP_SPEC, tmp_path)
    pixels = np.zeros((1, 137, 91, 3), dtype=np.uint8)
    out = sess.run(None, {"pixels": pixels})[0]
    assert out.shape == (1, 3, CLIP_SPEC.size, CLIP_SPEC.size)
    assert out.dtype == np.float32


def test_normalization_matches_a_numpy_reference(tmp_path: Path):
    sess = _session(CLIP_SPEC, tmp_path)
    # A constant image isolates normalization from resampling.
    pixels = np.full((1, 300, 300, 3), 128, dtype=np.uint8)
    out = sess.run(None, {"pixels": pixels})[0]
    expected = [(128 / 255.0 - m) / s for m, s in zip(CLIP_SPEC.mean, CLIP_SPEC.std)]
    for channel, value in enumerate(expected):
        assert np.allclose(out[0, channel], value, atol=1e-5)


def test_arbitrary_input_sizes_are_accepted(tmp_path: Path):
    sess = _session(SIGLIP_SPEC, tmp_path)
    for h, w in ((32, 512), (999, 17), (224, 224)):
        out = sess.run(None, {"pixels": np.zeros((1, h, w, 3), dtype=np.uint8)})[0]
        assert out.shape == (1, 3, SIGLIP_SPEC.size, SIGLIP_SPEC.size)


def test_channel_order_is_preserved(tmp_path: Path):
    sess = _session(CLIP_SPEC, tmp_path)
    pixels = np.zeros((1, 64, 64, 3), dtype=np.uint8)
    pixels[..., 0] = 255  # pure red in RGB
    out = sess.run(None, {"pixels": pixels})[0]
    red = (1.0 - CLIP_SPEC.mean[0]) / CLIP_SPEC.std[0]
    green = (0.0 - CLIP_SPEC.mean[1]) / CLIP_SPEC.std[1]
    assert abs(float(out[0, 0].mean()) - red) < 1e-3
    assert abs(float(out[0, 1].mean()) - green) < 1e-3


def test_center_square_crop_produces_a_square_of_the_shorter_side():
    im = Image.new("RGB", (300, 200))
    cropped = center_square_crop(im)
    assert cropped.size == (200, 200)


def test_center_square_crop_is_centered():
    im = Image.new("RGB", (5, 1), (0, 0, 0))
    im.putpixel((2, 0), (255, 255, 255))
    assert center_square_crop(im).getpixel((0, 0)) == (255, 255, 255)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_preprocess_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.features'`

- [ ] **Step 3: Write minimal implementation**

```python
# training/features/preprocess_graph.py
"""Build the preprocessing ONNX graph shared by training and the browser.

Keeping resize and normalization inside ONNX means onnxruntime in Python and
onnxruntime-web in the extension execute the same arithmetic. Doing this in
host code instead would leave resampling to PIL on one side and to canvas on
the other, which do not agree.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

OPSET = 17


@dataclass(frozen=True)
class PreprocessSpec:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    size: int
    square_crop: bool


# OpenAI CLIP normalization constants.
CLIP_SPEC = PreprocessSpec(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
    size=224,
    square_crop=True,
)

# SigLIP squashes to a square without cropping and normalizes to [-1, 1].
SIGLIP_SPEC = PreprocessSpec(
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    size=224,
    square_crop=False,
)


def center_square_crop(im: Image.Image) -> Image.Image:
    """Crop to a centered square of the shorter side, in integer arithmetic."""
    width, height = im.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return im.crop((left, top, left + side, top + side))


def _const(name: str, array: np.ndarray):
    return numpy_helper.from_array(array, name=name)


def build_preprocess_graph(spec: PreprocessSpec, path: Path) -> Path:
    mean = np.array(spec.mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(spec.std, dtype=np.float32).reshape(1, 3, 1, 1)

    initializers = [
        _const("scale255", np.array([255.0], dtype=np.float32)),
        _const("mean", mean),
        _const("std", std),
        _const("roi", np.array([], dtype=np.float32)),
        _const("scales", np.array([], dtype=np.float32)),
        _const("sizes", np.array([1, 3, spec.size, spec.size], dtype=np.int64)),
    ]

    nodes = [
        helper.make_node("Cast", ["pixels"], ["as_float"], to=TensorProto.FLOAT),
        helper.make_node("Transpose", ["as_float"], ["nchw"], perm=[0, 3, 1, 2]),
        helper.make_node(
            "Resize",
            ["nchw", "roi", "scales", "sizes"],
            ["resized"],
            mode="linear",
            coordinate_transformation_mode="half_pixel",
            nearest_mode="floor",
        ),
        helper.make_node("Div", ["resized", "scale255"], ["unit"]),
        helper.make_node("Sub", ["unit", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "std"], ["pixel_values"]),
    ]

    graph = helper.make_graph(
        nodes,
        "preprocess",
        inputs=[
            helper.make_tensor_value_info(
                "pixels", TensorProto.UINT8, [1, "H", "W", 3]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "pixel_values", TensorProto.FLOAT, [1, 3, spec.size, spec.size]
            )
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)]
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path
```

```python
# training/features/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_preprocess_graph.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add training/features/ tests/test_preprocess_graph.py
git commit -m "feat: build preprocessing as an ONNX graph shared with the browser runtime"
```

---

### Task 6: Robustness augmentation

**Files:**
- Create: `training/features/augment.py`
- Create: `tests/test_augment.py`

**Interfaces:**
- Consumes: nothing beyond Pillow and numpy.
- Produces: `training.features.augment.augment(im: Image.Image, rng: np.random.Generator) -> Image.Image` and `training.features.augment.AUG_PROBABILITIES: dict[str, float]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_augment.py
import numpy as np
from PIL import Image

from training.features.augment import AUG_PROBABILITIES, augment


def _photo(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))


def test_output_is_rgb_pil_image():
    out = augment(_photo(), np.random.default_rng(1))
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"


def test_same_seed_gives_identical_output():
    a = augment(_photo(), np.random.default_rng(7))
    b = augment(_photo(), np.random.default_rng(7))
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_different_seeds_give_different_output():
    a = augment(_photo(), np.random.default_rng(1))
    b = augment(_photo(), np.random.default_rng(2))
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_augmentation_actually_changes_pixels_over_many_seeds():
    original = np.asarray(_photo())
    changed = sum(
        not np.array_equal(np.asarray(augment(_photo(), np.random.default_rng(s))), original)
        for s in range(20)
    )
    assert changed >= 18, "augmentation is firing too rarely to build robustness"


def test_all_declared_probabilities_are_in_range():
    assert AUG_PROBABILITIES
    assert all(0.0 <= p <= 1.0 for p in AUG_PROBABILITIES.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_augment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.features.augment'`

- [ ] **Step 3: Write minimal implementation**

```python
# training/features/augment.py
"""Degradations that mimic what the web does to images.

Evaluation samples are described as web-realistic, which in practice means
recompressed, resized and rescaled. A detector trained only on pristine files
learns artifacts that these operations destroy.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

AUG_PROBABILITIES: dict[str, float] = {
    "jpeg": 0.60,
    "webp": 0.15,
    "downscale": 0.45,
    "crop": 0.35,
    "blur": 0.15,
    "noise": 0.15,
}


def _recompress(im: Image.Image, fmt: str, quality: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format=fmt, quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def augment(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    im = im.convert("RGB")

    if rng.random() < AUG_PROBABILITIES["crop"]:
        width, height = im.size
        keep = float(rng.uniform(0.70, 1.0))
        new_w, new_h = max(16, int(width * keep)), max(16, int(height * keep))
        left = int(rng.integers(0, max(1, width - new_w + 1)))
        top = int(rng.integers(0, max(1, height - new_h + 1)))
        im = im.crop((left, top, left + new_w, top + new_h))

    if rng.random() < AUG_PROBABILITIES["downscale"]:
        width, height = im.size
        factor = float(rng.uniform(0.40, 1.0))
        small = (max(16, int(width * factor)), max(16, int(height * factor)))
        im = im.resize(small, Image.BICUBIC).resize((width, height), Image.BICUBIC)

    if rng.random() < AUG_PROBABILITIES["blur"]:
        im = im.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.9))))

    if rng.random() < AUG_PROBABILITIES["noise"]:
        arr = np.asarray(im, dtype=np.float32)
        arr += rng.normal(0.0, float(rng.uniform(1.0, 5.0)), arr.shape)
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if rng.random() < AUG_PROBABILITIES["webp"]:
        im = _recompress(im, "WEBP", int(rng.integers(40, 96)))

    if rng.random() < AUG_PROBABILITIES["jpeg"]:
        im = _recompress(im, "JPEG", int(rng.integers(30, 96)))

    return im
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_augment.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add training/features/augment.py tests/test_augment.py
git commit -m "feat: add web-realistic degradation augmentation"
```

---

### Task 7: Backbone acquisition and feature extraction

**Files:**
- Create: `training/features/backbone.py`
- Create: `training/features/extract.py`
- Create: `tests/test_backbone.py`
- Modify: `models/LICENSES.md`

**Interfaces:**
- Consumes: `PreprocessSpec`, `CLIP_SPEC`, `SIGLIP_SPEC`, `build_preprocess_graph`, `center_square_crop`, `augment`, `read_manifest`, `ManifestRow`, `CONFIG`.
- Produces:
  - `training.features.backbone.Backbone` dataclass: `key: str`, `repo_id: str`, `filename: str`, `spec: PreprocessSpec`, `license_name: str`.
  - `training.features.backbone.BACKBONES: tuple[Backbone, ...]`.
  - `training.features.backbone.download_backbone(backbone: Backbone, models_dir: Path) -> Path`.
  - `training.features.backbone.FeatureExtractor` class with `__init__(self, backbone: Backbone, models_dir: Path)` and `embed(self, im: Image.Image) -> np.ndarray` returning a 1-D float32 vector.
  - `training.features.extract.extract_all(backbone_key: str, manifest_path: Path, cache_dir: Path, out_dir: Path, seed: int, augment_fraction: float) -> Path` writing `features.npy`, `labels.npy` and `rows.csv` into `out_dir / backbone_key`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backbone.py
import numpy as np
import pytest
from PIL import Image

from training.config import CONFIG
from training.features.backbone import BACKBONES, FeatureExtractor, download_backbone


def test_every_backbone_declares_a_permissive_license():
    permissive = {"MIT", "Apache-2.0", "BSD-3-Clause"}
    assert BACKBONES
    for b in BACKBONES:
        assert b.license_name in permissive, f"{b.key} license is not MIT-compatible"


def test_backbone_keys_are_unique():
    keys = [b.key for b in BACKBONES]
    assert len(keys) == len(set(keys))


@pytest.mark.slow
def test_extractor_produces_a_deterministic_fixed_length_vector():
    backbone = BACKBONES[0]
    download_backbone(backbone, CONFIG.models_dir)
    extractor = FeatureExtractor(backbone, CONFIG.models_dir)

    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 256, (321, 456, 3), dtype=np.uint8))

    first = extractor.embed(im)
    second = extractor.embed(im)

    assert first.ndim == 1
    assert first.shape[0] >= 256
    assert first.dtype == np.float32
    assert np.allclose(first, second, atol=0)


@pytest.mark.slow
def test_extractor_accepts_images_of_any_aspect_ratio():
    backbone = BACKBONES[0]
    download_backbone(backbone, CONFIG.models_dir)
    extractor = FeatureExtractor(backbone, CONFIG.models_dir)
    dim = None
    for size in ((64, 512), (512, 64), (224, 224)):
        vec = extractor.embed(Image.new("RGB", size, (120, 30, 200)))
        dim = dim or vec.shape[0]
        assert vec.shape[0] == dim
```

Register the marker so the slow tests can be skipped during fast iteration by appending to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
markers = ["slow: needs a downloaded backbone"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_backbone.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.features.backbone'`

- [ ] **Step 3: Write minimal implementation**

```python
# training/features/backbone.py
"""Frozen vision backbones, downloaded as ONNX and run under onnxruntime."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

from training.features.preprocess_graph import (
    CLIP_SPEC,
    SIGLIP_SPEC,
    PreprocessSpec,
    build_preprocess_graph,
    center_square_crop,
)


@dataclass(frozen=True)
class Backbone:
    key: str
    repo_id: str
    filename: str
    spec: PreprocessSpec
    license_name: str


BACKBONES: tuple[Backbone, ...] = (
    Backbone(
        key="clip-vit-b32",
        repo_id="Xenova/clip-vit-base-patch32",
        filename="onnx/vision_model.onnx",
        spec=CLIP_SPEC,
        license_name="MIT",
    ),
    Backbone(
        key="siglip-base-p16",
        repo_id="Xenova/siglip-base-patch16-224",
        filename="onnx/vision_model.onnx",
        spec=SIGLIP_SPEC,
        license_name="Apache-2.0",
    ),
)


def download_backbone(backbone: Backbone, models_dir: Path) -> Path:
    target_dir = models_dir / "backbones" / backbone.key
    target_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=backbone.repo_id,
            filename=backbone.filename,
            local_dir=str(target_dir),
        )
    )


class FeatureExtractor:
    """Runs preprocessing and the frozen backbone, returning one vector."""

    def __init__(self, backbone: Backbone, models_dir: Path) -> None:
        self.backbone = backbone
        model_path = download_backbone(backbone, models_dir)
        pre_path = build_preprocess_graph(
            backbone.spec, models_dir / "backbones" / backbone.key / "preprocess.onnx"
        )
        providers = ["CPUExecutionProvider"]
        self._pre = ort.InferenceSession(str(pre_path), providers=providers)
        self._net = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._net.get_inputs()[0].name
        self._output_name = self._pick_output()

    def _pick_output(self) -> str:
        """Prefer a pooled embedding over the token sequence."""
        names = [o.name for o in self._net.get_outputs()]
        for preferred in ("image_embeds", "pooler_output"):
            if preferred in names:
                return preferred
        return names[0]

    def embed(self, im: Image.Image) -> np.ndarray:
        im = im.convert("RGB")
        if self.backbone.spec.square_crop:
            im = center_square_crop(im)
        pixels = np.asarray(im, dtype=np.uint8)[None, ...]
        pixel_values = self._pre.run(None, {"pixels": pixels})[0]
        raw = self._net.run([self._output_name], {self._input_name: pixel_values})[0]
        vec = np.asarray(raw, dtype=np.float32).reshape(-1)
        if vec.size > 4096:  # token sequence fallback: mean-pool it
            tokens = np.asarray(raw, dtype=np.float32).reshape(-1, raw.shape[-1])
            vec = tokens.mean(axis=0)
        return vec.astype(np.float32)
```

```python
# training/features/extract.py
"""Extract and cache features for every image in the manifest."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from training.datasets.manifest import read_manifest
from training.features.augment import augment
from training.features.backbone import BACKBONES, FeatureExtractor

log = logging.getLogger(__name__)


def _backbone_by_key(key: str):
    for backbone in BACKBONES:
        if backbone.key == key:
            return backbone
    raise KeyError(f"unknown backbone {key!r}")


def extract_all(
    backbone_key: str,
    manifest_path: Path,
    cache_dir: Path,
    out_dir: Path,
    seed: int,
    augment_fraction: float,
) -> Path:
    backbone = _backbone_by_key(backbone_key)
    extractor = FeatureExtractor(backbone, Path("models"))
    rows = read_manifest(manifest_path)
    rng = np.random.default_rng(seed)

    vectors: list[np.ndarray] = []
    labels: list[int] = []
    kept: list[tuple[str, str, int, int]] = []

    for index, row in enumerate(rows):
        path = cache_dir / row.relpath
        if not path.exists():
            continue
        try:
            im = Image.open(path)
            im.load()
        except Exception:
            log.warning("undecodable cached file %s", path)
            continue

        augmented = rng.random() < augment_fraction
        if augmented:
            im = augment(im, np.random.default_rng(seed + index))

        vectors.append(extractor.embed(im))
        labels.append(row.label)
        kept.append((row.relpath, row.generator, row.label, int(augmented)))

        if index % 500 == 0:
            log.info("extracted %d/%d", index, len(rows))

    target = out_dir / backbone_key
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "features.npy", np.stack(vectors).astype(np.float32))
    np.save(target / "labels.npy", np.asarray(labels, dtype=np.int64))
    with (target / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relpath", "generator", "label", "augmented"))
        writer.writerows(kept)
    log.info("wrote %d feature vectors to %s", len(vectors), target)
    return target
```

- [ ] **Step 4: Run the fast tests**

Run: `.venv/bin/pytest tests/test_backbone.py -v -m "not slow"`
Expected: 2 passed, 2 deselected

- [ ] **Step 5: Run the slow tests, which download the backbone**

Run: `.venv/bin/pytest tests/test_backbone.py -v -m slow`
Expected: 2 passed. If `onnx/vision_model.onnx` is not present in the repository, list the repository files with `huggingface_hub.list_repo_files(repo_id)` and correct `filename` to the actual vision-encoder ONNX path.

- [ ] **Step 6: Record the licenses**

Edit `models/LICENSES.md`, replacing the empty row with one row per backbone actually downloaded, filling in artifact path, source repository, license and today's date.

- [ ] **Step 7: Extract features for both backbones**

```bash
.venv/bin/python -c "
import logging; logging.basicConfig(level=logging.INFO)
from training.config import CONFIG
from training.features.extract import extract_all
for key in ('clip-vit-b32', 'siglip-base-p16'):
    extract_all(key, CONFIG.data_dir/'manifest.csv', CONFIG.cache_dir,
                CONFIG.features_dir, CONFIG.seed, augment_fraction=0.5)
"
```

Expected: `data/features/<key>/features.npy` exists for both keys with matching row counts.

- [ ] **Step 8: Commit**

```bash
git add training/features/backbone.py training/features/extract.py \
        tests/test_backbone.py models/LICENSES.md pyproject.toml
git commit -m "feat: add frozen backbones and cached feature extraction"
```

---

### Task 8: Source-aware splits

**Files:**
- Create: `training/head/__init__.py`
- Create: `training/head/splits.py`
- Create: `tests/test_splits.py`

**Interfaces:**
- Consumes: `HELD_OUT_GENERATORS`.
- Produces: `training.head.splits.Split` dataclass with `train: np.ndarray`, `val_seen: np.ndarray`, `val_unseen: np.ndarray` (index arrays), and `training.head.splits.make_splits(generators: list[str], labels: np.ndarray, held_out: tuple[str, ...], seed: int, val_fraction: float = 0.2) -> Split`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_splits.py
import numpy as np

from training.head.splits import make_splits

GENS = (
    ["real"] * 40 + ["stable-diffusion-1"] * 20 + ["glide"] * 20 + ["flux"] * 20
)
LABELS = np.array([0] * 40 + [1] * 60)
HELD = ("flux",)


def test_held_out_generator_never_appears_in_train():
    split = make_splits(GENS, LABELS, HELD, seed=1)
    assert all(GENS[i] != "flux" for i in split.train)


def test_held_out_generator_appears_in_val_unseen():
    split = make_splits(GENS, LABELS, HELD, seed=1)
    assert any(GENS[i] == "flux" for i in split.val_unseen)


def test_val_unseen_contains_real_images_too():
    split = make_splits(GENS, LABELS, HELD, seed=1)
    assert any(LABELS[i] == 0 for i in split.val_unseen)
    assert any(LABELS[i] == 1 for i in split.val_unseen)


def test_splits_are_disjoint_and_cover_everything():
    split = make_splits(GENS, LABELS, HELD, seed=1)
    joined = np.concatenate([split.train, split.val_seen, split.val_unseen])
    assert len(joined) == len(set(joined.tolist())) == len(GENS)


def test_splits_are_deterministic_for_a_seed():
    a = make_splits(GENS, LABELS, HELD, seed=42)
    b = make_splits(GENS, LABELS, HELD, seed=42)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.val_unseen, b.val_unseen)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_splits.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.head'`

- [ ] **Step 3: Write minimal implementation**

```python
# training/head/splits.py
"""Partition data so validation measures generalization, not memorization.

Random splits leak: the same scene, prompt or capture session lands on both
sides. Held-out generators are stronger still, because they estimate what
happens on a generator the training never saw.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val_seen: np.ndarray
    val_unseen: np.ndarray


def make_splits(
    generators: list[str],
    labels: np.ndarray,
    held_out: tuple[str, ...],
    seed: int,
    val_fraction: float = 0.2,
) -> Split:
    generators_arr = np.asarray(generators)
    rng = np.random.default_rng(seed)

    unseen_mask = np.isin(generators_arr, held_out)
    seen_indices = np.flatnonzero(~unseen_mask)
    unseen_indices = np.flatnonzero(unseen_mask)

    # Real images are reserved for val_unseen too, so that split has both
    # classes and a balanced accuracy computed on it is meaningful.
    real_seen = seen_indices[labels[seen_indices] == 0]
    reserved_real = rng.permutation(real_seen)[: max(1, len(unseen_indices))]

    remaining = np.setdiff1d(seen_indices, reserved_real, assume_unique=False)
    shuffled = rng.permutation(remaining)
    cut = int(len(shuffled) * val_fraction)

    return Split(
        train=np.sort(shuffled[cut:]),
        val_seen=np.sort(shuffled[:cut]),
        val_unseen=np.sort(np.concatenate([unseen_indices, reserved_real])),
    )
```

```python
# training/head/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_splits.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add training/head/ tests/test_splits.py
git commit -m "feat: add source-aware splits with held-out generators"
```

---

### Task 9: Head training

**Files:**
- Create: `training/head/train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: `Split`.
- Produces:
  - `training.head.train.TrainedHead` dataclass: `weights: np.ndarray` (shape `[dim]`), `bias: float`, `dim: int`, `C: float`.
  - `training.head.train.l2_normalize(x: np.ndarray) -> np.ndarray`.
  - `training.head.train.raw_scores(head: TrainedHead, features: np.ndarray) -> np.ndarray` returning logits.
  - `training.head.train.train_head(features: np.ndarray, labels: np.ndarray, split: Split, seed: int) -> TrainedHead`.
  - `training.head.train.balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
import numpy as np

from training.head.splits import Split
from training.head.train import (
    TrainedHead,
    balanced_accuracy,
    l2_normalize,
    raw_scores,
    train_head,
)


def test_l2_normalize_gives_unit_rows():
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = l2_normalize(x)
    assert abs(float(np.linalg.norm(out[0])) - 1.0) < 1e-6
    assert np.all(np.isfinite(out[1]))  # zero row must not produce NaN


def test_balanced_accuracy_ignores_class_imbalance():
    labels = np.array([0] * 90 + [1] * 10)
    always_zero = np.zeros(100, dtype=int)
    assert balanced_accuracy(labels, always_zero) == 0.5


def test_balanced_accuracy_is_one_for_perfect_predictions():
    labels = np.array([0, 0, 1, 1])
    assert balanced_accuracy(labels, labels) == 1.0


def _separable(n: int = 400, dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    offset = np.zeros(dim, dtype=np.float32)
    offset[0] = 2.0
    features = rng.normal(0, 1, (n, dim)).astype(np.float32)
    features[labels == 1] += offset
    return features, labels


def test_train_head_learns_a_separable_problem():
    features, labels = _separable()
    idx = np.arange(len(labels))
    split = Split(train=idx[::2], val_seen=idx[1::4], val_unseen=idx[3::4])
    head = train_head(features, labels, split, seed=0)

    assert isinstance(head, TrainedHead)
    assert head.dim == features.shape[1]

    scores = raw_scores(head, l2_normalize(features[split.val_unseen]))
    predictions = (scores > 0).astype(int)
    assert balanced_accuracy(labels[split.val_unseen], predictions) > 0.9


def test_raw_scores_are_logits_not_probabilities():
    features, labels = _separable()
    idx = np.arange(len(labels))
    split = Split(train=idx[::2], val_seen=idx[1::4], val_unseen=idx[3::4])
    head = train_head(features, labels, split, seed=0)
    scores = raw_scores(head, l2_normalize(features))
    assert scores.min() < 0.0 < scores.max()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.head.train'`

- [ ] **Step 3: Write minimal implementation**

```python
# training/head/train.py
"""Train a shallow head on frozen features.

The head stays linear on purpose. A large head on frozen features overfits
the generators present in training, which is exactly the failure mode this
project is built to avoid.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from training.head.splits import Split

log = logging.getLogger(__name__)

C_GRID = (0.01, 0.1, 1.0, 10.0)


@dataclass(frozen=True)
class TrainedHead:
    weights: np.ndarray
    bias: float
    dim: int
    C: float


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def raw_scores(head: TrainedHead, features: np.ndarray) -> np.ndarray:
    return features @ head.weights + head.bias


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for cls in (0, 1):
        mask = labels == cls
        if not mask.any():
            continue
        scores.append(float((predictions[mask] == cls).mean()))
    return float(np.mean(scores))


def train_head(
    features: np.ndarray, labels: np.ndarray, split: Split, seed: int
) -> TrainedHead:
    normalized = l2_normalize(features.astype(np.float32))
    x_train, y_train = normalized[split.train], labels[split.train]
    x_val, y_val = normalized[split.val_seen], labels[split.val_seen]

    best: TrainedHead | None = None
    best_score = -1.0
    for c in C_GRID:
        model = LogisticRegression(
            C=c, class_weight="balanced", max_iter=3000, random_state=seed
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)
        score = balanced_accuracy(y_val, predictions)
        log.info("C=%s val_seen balanced accuracy=%.4f", c, score)
        if score > best_score:
            best_score = score
            best = TrainedHead(
                weights=model.coef_.reshape(-1).astype(np.float32),
                bias=float(model.intercept_[0]),
                dim=int(normalized.shape[1]),
                C=c,
            )
    assert best is not None
    return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_train.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add training/head/train.py tests/test_train.py
git commit -m "feat: train a linear head on frozen features with balanced accuracy selection"
```

---

### Task 10: Calibration onto the 0.65 decision point

This is the task that turns a competent model into a passing one. The evaluation cuts at 0.65, so the model's optimal boundary must land exactly there.

**Files:**
- Create: `training/head/calibrate.py`
- Create: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `balanced_accuracy`.
- Produces:
  - `training.head.calibrate.Calibration` dataclass: `a: float`, `b: float`, `t_star: float`, `decision_confidence: float`.
  - `training.head.calibrate.apply(cal: Calibration, scores: np.ndarray) -> np.ndarray` returning confidences in `[0, 1]`.
  - `training.head.calibrate.best_threshold(scores: np.ndarray, labels: np.ndarray) -> float`.
  - `training.head.calibrate.fit_calibration(scores: np.ndarray, labels: np.ndarray, decision_confidence: float, sharpness: float = 1.0) -> Calibration`.
  - `training.head.calibrate.dead_zone_fraction(confidences: np.ndarray, low: float = 0.35, high: float = 0.65) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibrate.py
import numpy as np

from training.head.calibrate import (
    apply,
    best_threshold,
    dead_zone_fraction,
    fit_calibration,
)
from training.head.train import balanced_accuracy


def _scores_and_labels(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    scores = rng.normal(-1.0, 1.0, n)
    scores[labels == 1] = rng.normal(1.5, 1.0, n // 2)
    return scores, labels


def test_best_threshold_lands_between_the_two_modes():
    scores, labels = _scores_and_labels()
    t = best_threshold(scores, labels)
    assert scores[labels == 0].mean() < t < scores[labels == 1].mean()


def test_calibration_maps_the_optimal_threshold_onto_the_decision_confidence():
    scores, labels = _scores_and_labels()
    cal = fit_calibration(scores, labels, decision_confidence=0.65)
    mapped = apply(cal, np.array([cal.t_star]))[0]
    assert abs(mapped - 0.65) < 1e-6


def test_calibration_is_monotonic():
    scores, labels = _scores_and_labels()
    cal = fit_calibration(scores, labels, decision_confidence=0.65)
    grid = np.linspace(-8, 8, 400)
    out = apply(cal, grid)
    assert np.all(np.diff(out) >= -1e-12)


def test_confidences_stay_inside_the_unit_interval():
    scores, labels = _scores_and_labels()
    cal = fit_calibration(scores, labels, decision_confidence=0.65)
    out = apply(cal, np.linspace(-100, 100, 1000))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_thresholding_at_065_matches_thresholding_the_raw_score_at_t_star():
    scores, labels = _scores_and_labels()
    cal = fit_calibration(scores, labels, decision_confidence=0.65)
    from_raw = (scores >= cal.t_star).astype(int)
    from_confidence = (apply(cal, scores) >= 0.65).astype(int)
    assert np.array_equal(from_raw, from_confidence)


def test_sharpness_shrinks_the_dead_zone_without_moving_the_boundary():
    scores, labels = _scores_and_labels()
    soft = fit_calibration(scores, labels, 0.65, sharpness=1.0)
    sharp = fit_calibration(scores, labels, 0.65, sharpness=3.0)

    soft_predictions = (apply(soft, scores) >= 0.65).astype(int)
    sharp_predictions = (apply(sharp, scores) >= 0.65).astype(int)
    assert np.array_equal(soft_predictions, sharp_predictions)
    assert balanced_accuracy(labels, soft_predictions) == balanced_accuracy(
        labels, sharp_predictions
    )

    assert dead_zone_fraction(apply(sharp, scores)) < dead_zone_fraction(
        apply(soft, scores)
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.head.calibrate'`

- [ ] **Step 3: Write minimal implementation**

```python
# training/head/calibrate.py
"""Map raw scores onto a confidence whose decision point is 0.65.

Evaluation thresholds confidence at 0.65 rather than 0.5. A conventionally
calibrated model puts its optimal boundary at 0.5, so thresholding it at 0.65
shifts predictions toward the negative class and costs balanced accuracy.

The fix is a monotonic affine-in-logit map constrained so that the raw
threshold maximizing balanced accuracy lands exactly on 0.65. The extension
flags images at that same 0.65, so the product's operating point and the
classifier's optimal operating point are the same number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Calibration:
    a: float
    b: float
    t_star: float
    decision_confidence: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    parts = []
    for cls in (0, 1):
        mask = labels == cls
        if mask.any():
            parts.append(float((predictions[mask] == cls).mean()))
    return float(np.mean(parts))


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(scores)
    if candidates.size > 2000:
        candidates = np.quantile(scores, np.linspace(0.0, 1.0, 2000))
    best_t, best_score = float(candidates[0]), -1.0
    for t in candidates:
        score = _balanced_accuracy(labels, (scores >= t).astype(int))
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def fit_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    decision_confidence: float,
    sharpness: float = 1.0,
) -> Calibration:
    t_star = best_threshold(scores, labels)

    # Slope from the spread of the scores, then scaled by the sharpness knob.
    spread = float(np.std(scores)) or 1.0
    a = sharpness * (2.0 / spread)

    # Constrain the map so that t_star lands on the decision confidence.
    b = _logit(decision_confidence) - a * t_star

    return Calibration(
        a=a, b=b, t_star=t_star, decision_confidence=decision_confidence
    )


def apply(cal: Calibration, scores: np.ndarray) -> np.ndarray:
    return _sigmoid(cal.a * np.asarray(scores, dtype=np.float64) + cal.b)


def dead_zone_fraction(
    confidences: np.ndarray, low: float = 0.35, high: float = 0.65
) -> float:
    inside = (confidences >= low) & (confidences < high)
    return float(inside.mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_calibrate.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add training/head/calibrate.py tests/test_calibrate.py
git commit -m "feat: calibrate confidence so the optimal boundary lands on 0.65"
```

---

### Task 11: Export head and calibration as runtime artifacts

**Files:**
- Create: `training/export/__init__.py`
- Create: `training/export/head_onnx.py`
- Create: `tests/test_export.py`

**Interfaces:**
- Consumes: `TrainedHead`, `Calibration`.
- Produces:
  - `training.export.head_onnx.export_head(head: TrainedHead, path: Path) -> Path` writing an ONNX graph with input `features` float32 `[1, dim]` and output `score` float32 `[1]` (the raw logit, before calibration).
  - `training.export.head_onnx.export_calibration(cal: Calibration, head: TrainedHead, backbone_key: str, path: Path) -> Path` writing JSON with keys `a`, `b`, `t_star`, `decision_confidence`, `dim`, `backbone_key`, `l2_normalize`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export.py
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from training.export.head_onnx import export_calibration, export_head
from training.head.calibrate import Calibration
from training.head.train import TrainedHead, raw_scores

HEAD = TrainedHead(
    weights=np.array([0.5, -1.5, 2.0], dtype=np.float32), bias=0.25, dim=3, C=1.0
)
CAL = Calibration(a=1.7, b=-0.4, t_star=0.35, decision_confidence=0.65)


def test_exported_graph_matches_the_python_scores(tmp_path: Path):
    path = export_head(HEAD, tmp_path / "head.onnx")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    for _ in range(20):
        features = rng.normal(0, 1, (1, 3)).astype(np.float32)
        onnx_score = sess.run(None, {"features": features})[0].reshape(-1)[0]
        python_score = raw_scores(HEAD, features)[0]
        assert abs(float(onnx_score) - float(python_score)) < 1e-5


def test_exported_graph_declares_the_expected_io(tmp_path: Path):
    path = export_head(HEAD, tmp_path / "head.onnx")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["features"]
    assert [o.name for o in sess.get_outputs()] == ["score"]


def test_calibration_json_carries_everything_the_runtime_needs(tmp_path: Path):
    path = export_calibration(CAL, HEAD, "clip-vit-b32", tmp_path / "calibration.json")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["a"] == CAL.a
    assert payload["b"] == CAL.b
    assert payload["t_star"] == CAL.t_star
    assert payload["decision_confidence"] == 0.65
    assert payload["dim"] == 3
    assert payload["backbone_key"] == "clip-vit-b32"
    assert payload["l2_normalize"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.export'`

- [ ] **Step 3: Write minimal implementation**

```python
# training/export/head_onnx.py
"""Write the runtime artifacts the browser extension consumes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from training.head.calibrate import Calibration
from training.head.train import TrainedHead

OPSET = 17


def export_head(head: TrainedHead, path: Path) -> Path:
    weights = head.weights.astype(np.float32).reshape(head.dim, 1)
    bias = np.array([head.bias], dtype=np.float32)

    nodes = [
        helper.make_node("MatMul", ["features", "W"], ["projected"]),
        helper.make_node("Add", ["projected", "B"], ["biased"]),
        helper.make_node("Flatten", ["biased"], ["score"], axis=0),
    ]
    graph = helper.make_graph(
        nodes,
        "head",
        inputs=[
            helper.make_tensor_value_info(
                "features", TensorProto.FLOAT, [1, head.dim]
            )
        ],
        outputs=[helper.make_tensor_value_info("score", TensorProto.FLOAT, [1, 1])],
        initializer=[
            numpy_helper.from_array(weights, name="W"),
            numpy_helper.from_array(bias, name="B"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def export_calibration(
    cal: Calibration, head: TrainedHead, backbone_key: str, path: Path
) -> Path:
    payload = {
        "a": cal.a,
        "b": cal.b,
        "t_star": cal.t_star,
        "decision_confidence": cal.decision_confidence,
        "dim": head.dim,
        "backbone_key": backbone_key,
        "l2_normalize": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
```

```python
# training/export/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_export.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add training/export/ tests/test_export.py
git commit -m "feat: export head as ONNX and calibration as JSON for the runtime"
```

---

### Task 12: Evaluation report and the Phase 1 gate

**Files:**
- Create: `training/report.py`
- Create: `training/cli.py`
- Create: `tests/test_report.py`
- Create: `docs/superpowers/reports/2026-08-14-phase1-gate.md`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `training.report.EvalResult` dataclass: `balanced_accuracy: float`, `tpr: float`, `tnr: float`, `dead_zone: float`, `per_generator: dict[str, float]`, `threshold_curve: list[tuple[float, float]]`, `n: int`.
  - `training.report.evaluate(confidences: np.ndarray, labels: np.ndarray, generators: list[str], decision_confidence: float) -> EvalResult`.
  - `training.report.render_markdown(result: EvalResult, title: str) -> str`.
  - `training.cli.main(argv: list[str] | None = None) -> int` exposing subcommands `fetch`, `extract`, `train`, `report`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
import numpy as np

from training.report import evaluate, render_markdown


def test_evaluate_reports_balanced_accuracy_at_the_decision_confidence():
    labels = np.array([0, 0, 1, 1])
    confidences = np.array([0.10, 0.20, 0.90, 0.80])
    generators = ["real", "real", "flux", "flux"]
    result = evaluate(confidences, labels, generators, 0.65)
    assert result.balanced_accuracy == 1.0
    assert result.tpr == 1.0
    assert result.tnr == 1.0
    assert result.n == 4


def test_confidence_just_below_the_threshold_counts_as_real():
    labels = np.array([1, 0])
    confidences = np.array([0.6499, 0.10])
    result = evaluate(confidences, labels, ["flux", "real"], 0.65)
    assert result.tpr == 0.0
    assert result.tnr == 1.0
    assert result.balanced_accuracy == 0.5


def test_per_generator_breakdown_uses_recall_within_each_generator():
    labels = np.array([1, 1, 1, 1, 0, 0])
    confidences = np.array([0.9, 0.9, 0.2, 0.2, 0.1, 0.1])
    generators = ["flux", "flux", "sd", "sd", "real", "real"]
    result = evaluate(confidences, labels, generators, 0.65)
    assert result.per_generator["flux"] == 1.0
    assert result.per_generator["sd"] == 0.0
    assert result.per_generator["real"] == 1.0


def test_threshold_curve_covers_the_unit_interval():
    labels = np.array([0, 1])
    result = evaluate(np.array([0.2, 0.8]), labels, ["real", "flux"], 0.65)
    xs = [t for t, _ in result.threshold_curve]
    assert min(xs) <= 0.05 and max(xs) >= 0.95
    assert all(0.0 <= ba <= 1.0 for _, ba in result.threshold_curve)


def test_render_markdown_states_the_headline_number():
    labels = np.array([0, 1])
    result = evaluate(np.array([0.2, 0.8]), labels, ["real", "flux"], 0.65)
    text = render_markdown(result, "Phase 1")
    assert "Phase 1" in text
    assert "1.0000" in text or "1.000" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.report'`

- [ ] **Step 3: Write minimal implementation**

```python
# training/report.py
"""Turn confidences into the numbers that decide whether this approach works."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training.head.calibrate import dead_zone_fraction


@dataclass(frozen=True)
class EvalResult:
    balanced_accuracy: float
    tpr: float
    tnr: float
    dead_zone: float
    per_generator: dict[str, float]
    threshold_curve: list[tuple[float, float]]
    n: int


def _rates(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    positive, negative = labels == 1, labels == 0
    tpr = float((predictions[positive] == 1).mean()) if positive.any() else 0.0
    tnr = float((predictions[negative] == 0).mean()) if negative.any() else 0.0
    return tpr, tnr


def evaluate(
    confidences: np.ndarray,
    labels: np.ndarray,
    generators: list[str],
    decision_confidence: float,
) -> EvalResult:
    confidences = np.asarray(confidences, dtype=np.float64)
    labels = np.asarray(labels)
    predictions = (confidences >= decision_confidence).astype(int)
    tpr, tnr = _rates(labels, predictions)

    generators_arr = np.asarray(generators)
    per_generator: dict[str, float] = {}
    for name in sorted(set(generators)):
        mask = generators_arr == name
        expected = int(labels[mask][0])
        per_generator[name] = float((predictions[mask] == expected).mean())

    curve = []
    for t in np.linspace(0.02, 0.98, 49):
        t_predictions = (confidences >= t).astype(int)
        a, b = _rates(labels, t_predictions)
        curve.append((round(float(t), 4), round((a + b) / 2.0, 4)))

    return EvalResult(
        balanced_accuracy=(tpr + tnr) / 2.0,
        tpr=tpr,
        tnr=tnr,
        dead_zone=dead_zone_fraction(confidences, 0.35, decision_confidence),
        per_generator=per_generator,
        threshold_curve=curve,
        n=int(labels.size),
    )


def render_markdown(result: EvalResult, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Images evaluated: {result.n}",
        f"- **Balanced accuracy at the decision threshold: {result.balanced_accuracy:.4f}**",
        f"- True positive rate (synthetic detected): {result.tpr:.4f}",
        f"- True negative rate (real kept): {result.tnr:.4f}",
        f"- Fraction in the low-confidence dead zone: {result.dead_zone:.4f}",
        "",
        "## Per generator",
        "",
        "| Generator | Accuracy |",
        "|---|---|",
    ]
    for name, value in sorted(result.per_generator.items(), key=lambda kv: kv[1]):
        lines.append(f"| {name} | {value:.4f} |")
    lines += ["", "## Balanced accuracy versus threshold", "", "| Threshold | BA |", "|---|---|"]
    lines += [f"| {t:.2f} | {ba:.4f} |" for t, ba in result.threshold_curve]
    return "\n".join(lines) + "\n"
```

```python
# training/cli.py
"""Command line entry point for the offline pipeline."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

from training.config import CONFIG, HELD_OUT_GENERATORS
from training.datasets.fetch import fetch_all
from training.export.head_onnx import export_calibration, export_head
from training.features.extract import extract_all
from training.head.calibrate import apply, fit_calibration
from training.head.splits import make_splits
from training.head.train import l2_normalize, raw_scores, train_head
from training.report import evaluate, render_markdown


def _load(backbone_key: str):
    base = CONFIG.features_dir / backbone_key
    features = np.load(base / "features.npy")
    labels = np.load(base / "labels.npy")
    with (base / "rows.csv").open(newline="", encoding="utf-8") as handle:
        generators = [r["generator"] for r in csv.DictReader(handle)]
    return features, labels, generators


def _train_and_report(backbone_key: str, sharpness: float) -> int:
    features, labels, generators = _load(backbone_key)
    split = make_splits(generators, labels, HELD_OUT_GENERATORS, CONFIG.seed)

    head = train_head(features, labels, split, CONFIG.seed)
    normalized = l2_normalize(features.astype(np.float32))
    scores = raw_scores(head, normalized)

    cal = fit_calibration(
        scores[split.val_unseen],
        labels[split.val_unseen],
        CONFIG.decision_confidence,
        sharpness=sharpness,
    )

    confidences = apply(cal, scores[split.val_unseen])
    result = evaluate(
        confidences,
        labels[split.val_unseen],
        [generators[i] for i in split.val_unseen],
        CONFIG.decision_confidence,
    )

    out_dir = CONFIG.models_dir / backbone_key
    export_head(head, out_dir / "head.onnx")
    export_calibration(cal, head, backbone_key, out_dir / "calibration.json")

    report_path = (
        Path("docs/superpowers/reports") / f"phase1-{backbone_key}.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_markdown(result, f"Phase 1 gate: {backbone_key}"), encoding="utf-8"
    )
    print(json.dumps({"backbone": backbone_key, "ba": result.balanced_accuracy}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="training")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch")

    extract = sub.add_parser("extract")
    extract.add_argument("backbone")
    extract.add_argument("--augment-fraction", type=float, default=0.5)

    train = sub.add_parser("train")
    train.add_argument("backbone")
    train.add_argument("--sharpness", type=float, default=1.0)

    args = parser.parse_args(argv)

    if args.command == "fetch":
        fetch_all(CONFIG.cache_dir, CONFIG.data_dir / "manifest.csv", CONFIG.seed)
        return 0
    if args.command == "extract":
        extract_all(
            args.backbone,
            CONFIG.data_dir / "manifest.csv",
            CONFIG.cache_dir,
            CONFIG.features_dir,
            CONFIG.seed,
            args.augment_fraction,
        )
        return 0
    return _train_and_report(args.backbone, args.sharpness)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_report.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/pytest -v -m "not slow"`
Expected: all pass

- [ ] **Step 6: Run the gate for both backbones**

```bash
.venv/bin/python -m training.cli train clip-vit-b32
.venv/bin/python -m training.cli train siglip-base-p16
```

Expected: two JSON lines with a `ba` value each, and two report files under `docs/superpowers/reports/`.

- [ ] **Step 7: Record the gate decision**

Create `docs/superpowers/reports/2026-08-14-phase1-gate.md` containing: the balanced accuracy of each backbone on `val_unseen`, which backbone won, the per-generator table for the winner, and the decision drawn from the spec's thresholds.

Decision rule, copied from spec section 11:
- At or above 0.85: comfortable margin, proceed to Plan B and scale the dataset.
- Between 0.75 and 0.85: viable, but the full augmentation program and more data are required before submission.
- Below 0.75: stop. Do not start Plan B. Reassess the approach, considering CLIP ViT-L/14 as the backbone and a high-frequency residual second head.

- [ ] **Step 8: Commit**

```bash
git add training/report.py training/cli.py tests/test_report.py \
        docs/superpowers/reports/ models/
git commit -m "feat: add evaluation report and phase 1 gate CLI"
```

---

## Plan Self-Review

**Spec coverage.**

| Spec section | Covered by |
|---|---|
| 1 Success criterion | Task 12 (balanced accuracy at 0.65) |
| 3 Frozen backbone plus head | Tasks 7, 9 |
| 4 Preprocessing contract | Task 5 |
| 5 Threshold 0.65 and calibration | Task 10 |
| 6 Data plan, held-out generators, augmentation, source splits | Tasks 3, 4, 6, 8 |
| 7 Model selection between candidates | Tasks 7, 12 (both backbones trained and compared) |
| 11 Phase 1 gate | Task 12 |
| 12 MIT and license compatibility | Tasks 1, 7 (`models/LICENSES.md`) |
| 13 Repository layout | Tasks 1 through 12 build `training/`, `models/`, `scripts/` |

Sections 8, 9, 10 and 14 of the spec (extension runtime, metadata fusion, Puppeteer harness) are Plan B, deliberately out of scope here.

**Placeholder scan.** One item needs attention at execution time and is explicitly structured, not vague: Task 3 Step 4 says to replace `repo_id` values with those verified in Step 3. That is a real procedure with acceptance criteria enforced by tests in Step 1, not a TODO. Similarly Task 7 Step 5 names the exact fallback (`list_repo_files`) if the ONNX filename differs.

**Type consistency.** `TrainedHead.dim` is used identically in Tasks 9 and 11. `Calibration.a/b/t_star/decision_confidence` match between Task 10 and Task 11's JSON payload. `Split.train/val_seen/val_unseen` match between Tasks 8, 9 and 12. `ManifestRow` fields match between Tasks 3, 4 and 7. `balanced_accuracy` is defined once in `training/head/train.py` and reimplemented privately inside `calibrate.py` as `_balanced_accuracy` to avoid a circular import; both are covered by tests.
