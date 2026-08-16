"""Command line entry point for the offline pipeline.

    python -m training.cli fetch
    python -m training.cli extract clip-vit-b32
    python -m training.cli train clip-vit-b32
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

from training.config import CONFIG, HELD_OUT_GENERATORS
from training.datasets.fetch import fetch_all
from training.export.head_onnx import export_calibration, export_head
from training.features.extract import extract_all
from training.head.calibrate import apply, fit_calibration
from training.head.splits import make_splits, split_for_calibration
from training.head.train import build_features, raw_scores, train_head
from training.report import evaluate, render_markdown

REPORT_DIR = Path("docs/superpowers/reports")


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
    logging.info(
        "%s: %d train, %d val_seen, %d val_unseen",
        backbone_key,
        len(split.train),
        len(split.val_seen),
        len(split.val_unseen),
    )

    # The held-out generators are halved first: one half chooses the model and
    # the threshold, the other is reported. Fitting and reporting on the same
    # images would pick both knowing the answers.
    generators_arr = np.asarray(generators)
    calib_idx, report_idx = split_for_calibration(
        split.val_unseen,
        generators_arr[split.val_unseen],
        labels[split.val_unseen],
        CONFIG.seed,
    )

    inputs = build_features(features)
    # Selecting on generators the training never saw picks the head that
    # generalizes, rather than the one that fits the familiar ones best.
    head = train_head(inputs, labels, split, CONFIG.seed, selection_indices=calib_idx)
    scores = raw_scores(head, inputs)

    cal = fit_calibration(
        scores[calib_idx],
        labels[calib_idx],
        CONFIG.decision_confidence,
        sharpness=sharpness,
        families=generators_arr[calib_idx],
    )

    result = evaluate(
        apply(cal, scores[report_idx]),
        labels[report_idx],
        [generators[i] for i in report_idx],
        CONFIG.decision_confidence,
    )
    seen_result = evaluate(
        apply(cal, scores[split.val_seen]),
        labels[split.val_seen],
        [generators[i] for i in split.val_seen],
        CONFIG.decision_confidence,
    )

    out_dir = CONFIG.models_dir / backbone_key
    export_head(head, out_dir / "head.onnx")
    export_calibration(cal, head, backbone_key, out_dir / "calibration.json")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"phase1-{backbone_key}.md").write_text(
        render_markdown(result, f"Phase 1 gate: {backbone_key} (held-out generators)"),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "backbone": backbone_key,
                "balanced_accuracy_unseen": round(result.balanced_accuracy, 4),
                "balanced_accuracy_seen": round(seen_result.balanced_accuracy, 4),
                "tpr_unseen": round(result.tpr, 4),
                "tnr_unseen": round(result.tnr, 4),
                "dead_zone_unseen": round(result.dead_zone, 4),
                "per_generator": {k: round(v, 4) for k, v in result.per_generator.items()},
            },
            indent=2,
        )
    )
    return 0


def _export_benchmark(backbone_key: str, out_dir: Path) -> int:
    """Copy the reporting half of the held-out split into real/ and ai/ folders.

    The browser harness scores whatever is in those folders. Exporting exactly
    the images the Python gate reported on is what makes the two numbers
    comparable: a difference between them is then a difference in the code
    path, not a difference in the sample.
    """
    features, labels, generators = _load(backbone_key)
    split = make_splits(generators, labels, HELD_OUT_GENERATORS, CONFIG.seed)
    generators_arr = np.asarray(generators)
    _, report_idx = split_for_calibration(
        split.val_unseen,
        generators_arr[split.val_unseen],
        labels[split.val_unseen],
        CONFIG.seed,
    )

    base = CONFIG.features_dir / backbone_key
    with (base / "rows.csv").open(newline="", encoding="utf-8") as handle:
        relpaths = [r["relpath"] for r in csv.DictReader(handle)]

    real_dir, ai_dir = out_dir / "real", out_dir / "ai"
    for directory in (real_dir, ai_dir):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)

    counts = {"real": 0, "ai": 0}
    for index in report_idx:
        source = CONFIG.cache_dir / relpaths[index]
        if not source.exists():
            continue
        target_dir = ai_dir if labels[index] == 1 else real_dir
        shutil.copy2(source, target_dir / f"{generators[index]}__{source.name}")
        counts["ai" if labels[index] == 1 else "real"] += 1

    print(json.dumps({"exported": counts, "out": str(out_dir)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(prog="training")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch")

    extract = sub.add_parser("extract")
    extract.add_argument("backbone")
    extract.add_argument("--augment-fraction", type=float, default=0.5)

    train = sub.add_parser("train")
    train.add_argument("backbone")
    train.add_argument("--sharpness", type=float, default=3.0)

    export = sub.add_parser("export-benchmark")
    export.add_argument("backbone")
    export.add_argument("--out", default="data/benchmark")

    args = parser.parse_args(argv)

    if args.command == "fetch":
        fetch_all(CONFIG.cache_dir, CONFIG.data_dir / "manifest.csv", CONFIG.seed)
        return 0
    if args.command == "export-benchmark":
        return _export_benchmark(args.backbone, Path(args.out))
    if args.command == "extract":
        extract_all(
            args.backbone,
            CONFIG.data_dir / "manifest.csv",
            CONFIG.cache_dir,
            CONFIG.features_dir,
            CONFIG.models_dir,
            CONFIG.seed,
            args.augment_fraction,
        )
        return 0
    return _train_and_report(args.backbone, args.sharpness)


if __name__ == "__main__":
    sys.exit(main())
