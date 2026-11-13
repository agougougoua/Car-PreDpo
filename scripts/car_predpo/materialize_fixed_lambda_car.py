#!/usr/bin/env python3
"""Create an isolated fixed-lambda CAR Stage 2 split without changing examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty JSON list: {path}")
    return rows


def rewrite(rows: list[dict], fixed_lambda: float) -> list[dict]:
    rewritten = []
    for row in rows:
        item = dict(row)
        item["lambda_i_original"] = float(item["lambda_i"])
        item["margin_original"] = float(item.get("margin", item["lambda_i"]))
        item["lambda_i"] = fixed_lambda
        item["lambda_base"] = fixed_lambda
        item["lambda_raw"] = fixed_lambda
        item["margin"] = fixed_lambda * float(item.get("delta_h", 1.0))
        item["fixed_lambda_car"] = True
        rewritten.append(item)
    return rewritten


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a fixed-lambda CAR control split.")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lambda", dest="fixed_lambda", type=float)
    parser.add_argument("--expected-train", type=int, default=5131)
    parser.add_argument("--expected-dev", type=int, default=570)
    args = parser.parse_args()

    train = load_rows(args.train)
    dev = load_rows(args.dev)
    if len(train) != args.expected_train or len(dev) != args.expected_dev:
        raise ValueError(f"Unexpected split sizes: train={len(train)}, dev={len(dev)}")
    original_lambdas = [float(row["lambda_i"]) for row in train]
    fixed_lambda = args.fixed_lambda if args.fixed_lambda is not None else sum(original_lambdas) / len(original_lambdas)
    if not 0.0 < fixed_lambda < 1.0:
        raise ValueError(f"Invalid fixed lambda: {fixed_lambda}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    with (output / "car_predpo_train_fixed_lambda.json").open("w", encoding="utf-8") as handle:
        json.dump(rewrite(train, fixed_lambda), handle, ensure_ascii=False)
    with (output / "car_predpo_dev_fixed_lambda.json").open("w", encoding="utf-8") as handle:
        json.dump(rewrite(dev, fixed_lambda), handle, ensure_ascii=False)
    info = {
        "car_predpo_train_fixed_lambda": {
            "file_name": "car_predpo_train_fixed_lambda.json",
            "formatting": "alpaca",
            "ranking": True,
            "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected", "images": "images", "lambda_i": "lambda_i", "delta_h": "delta_h", "vdi": "vdi"},
        },
        "car_predpo_dev_fixed_lambda": {
            "file_name": "car_predpo_dev_fixed_lambda.json",
            "formatting": "alpaca",
            "ranking": True,
            "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected", "images": "images", "lambda_i": "lambda_i", "delta_h": "delta_h", "vdi": "vdi"},
        },
    }
    (output / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "control": "fixed_lambda_car",
        "fixed_lambda": fixed_lambda,
        "original_train_lambda": {"min": min(original_lambdas), "mean": sum(original_lambdas) / len(original_lambdas), "max": max(original_lambdas)},
        "source": {"train": str(args.train.resolve()), "dev": str(args.dev.resolve())},
        "counts": {"train": len(train), "dev": len(dev)},
        "invariant": "Examples, RVDI/VDI diagnostics, delta_h and split are preserved; only loss-facing lambda_i is made constant.",
    }
    (output / "fixed_lambda_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
