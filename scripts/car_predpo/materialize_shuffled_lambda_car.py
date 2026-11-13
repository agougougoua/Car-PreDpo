#!/usr/bin/env python3
"""Create an isolated shuffled-lambda CAR Stage 2 control split."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty JSON list: {path}")
    return rows


def lambda_summary(lambdas: list[float]) -> dict[str, float]:
    return {"min": min(lambdas), "mean": sum(lambdas) / len(lambdas), "max": max(lambdas)}


def shuffled_train_rows(rows: list[dict], permutation_seed: int) -> tuple[list[dict], list[int]]:
    if len(rows) < 2:
        raise ValueError("Shuffled-lambda control requires at least two train rows")
    original_lambdas = [float(row["lambda_i"]) for row in rows]
    if not all(math.isfinite(value) and 0.0 < value < 1.0 for value in original_lambdas):
        raise ValueError("Every train lambda_i must be finite and strictly between 0 and 1")
    permutation = list(range(len(rows)))
    random.Random(permutation_seed).shuffle(permutation)
    if permutation == list(range(len(rows))):
        permutation[0], permutation[1] = permutation[1], permutation[0]

    rewritten = []
    for index, row in enumerate(rows):
        shuffled_lambda = original_lambdas[permutation[index]]
        item = dict(row)
        item["lambda_i_original"] = original_lambdas[index]
        item["margin_original"] = float(item.get("margin", item["lambda_i"]))
        item["lambda_source_row_index"] = permutation[index]
        item["lambda_i"] = shuffled_lambda
        item["lambda_base"] = shuffled_lambda
        item["lambda_raw"] = shuffled_lambda
        item["margin"] = shuffled_lambda * float(item.get("delta_h", 1.0))
        item["shuffled_lambda_car"] = True
        rewritten.append(item)
    return rewritten, permutation


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a shuffled-lambda CAR control split.")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--permutation-seed", type=int, default=42)
    parser.add_argument("--expected-train", type=int, default=5131)
    parser.add_argument("--expected-dev", type=int, default=570)
    args = parser.parse_args()

    train = load_rows(args.train)
    dev = load_rows(args.dev)
    if len(train) != args.expected_train or len(dev) != args.expected_dev:
        raise ValueError(f"Unexpected split sizes: train={len(train)}, dev={len(dev)}")
    shuffled_train, permutation = shuffled_train_rows(train, args.permutation_seed)
    original_lambdas = [float(row["lambda_i"]) for row in train]

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / "car_predpo_train_shuffled_lambda.json").write_text(
        json.dumps(shuffled_train, ensure_ascii=False), encoding="utf-8"
    )
    (output / "car_predpo_dev_shuffled_lambda.json").write_text(
        json.dumps(dev, ensure_ascii=False), encoding="utf-8"
    )
    info = {
        "car_predpo_train_shuffled_lambda": {
            "file_name": "car_predpo_train_shuffled_lambda.json",
            "formatting": "alpaca",
            "ranking": True,
            "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected", "images": "images", "lambda_i": "lambda_i", "delta_h": "delta_h", "vdi": "vdi"},
        },
        "car_predpo_dev_shuffled_lambda": {
            "file_name": "car_predpo_dev_shuffled_lambda.json",
            "formatting": "alpaca",
            "ranking": True,
            "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected", "images": "images", "lambda_i": "lambda_i", "delta_h": "delta_h", "vdi": "vdi"},
        },
    }
    (output / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "control": "shuffled_lambda_car",
        "permutation_seed": args.permutation_seed,
        "source": {"train": str(args.train.resolve()), "dev": str(args.dev.resolve())},
        "counts": {"train": len(train), "dev": len(dev)},
        "original_train_lambda": lambda_summary(original_lambdas),
        "permutation": {"fixed_points": sum(index == source for index, source in enumerate(permutation))},
        "invariant": "Train examples, RVDI/VDI diagnostics, delta_h and split are preserved; train lambda_i values are permuted within the train set and margin is recomputed. Dev rows remain Full CAR values.",
    }
    (output / "shuffled_lambda_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
