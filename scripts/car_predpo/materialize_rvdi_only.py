#!/usr/bin/env python3
"""Materialize the RVDI-only margin control from existing scored splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from car_predpo.margin import build_margins


def load_rows(path: Path, expected: int) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f"Unexpected row count in {path}: expected {expected}")
    ids = [row["record_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate record_id in {path}")
    for row in rows:
        if not 0 <= float(row["percentile"]) <= 1 or not math.isfinite(float(row["rvdi_lcb"])):
            raise ValueError(f"Invalid RVDI or percentile for {row['record_id']}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-train", type=int, default=5131)
    parser.add_argument("--expected-dev", type=int, default=570)
    args = parser.parse_args()

    train = load_rows(args.train, args.expected_train)
    dev = load_rows(args.dev, args.expected_dev)
    if set(row["record_id"] for row in train) & set(row["record_id"] for row in dev):
        raise ValueError("Train and dev IDs overlap")
    result = {}
    reports = {}
    for label, source in (("train", train), ("dev", dev)):
        rows = [{**row, "lambda_i_original": row.get("lambda_i"), "margin_original": row.get("margin")} for row in source]
        result[label], reports[label] = build_margins(rows, lambda_min=0.03, lambda_max=0.20,
                                                       alpha=0.0, rho=0.0, eta=0.0)
        if not all(math.isfinite(float(row["lambda_i"])) and math.isfinite(float(row["margin"])) for row in result[label]):
            raise ValueError(f"Non-finite margin in {label}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    columns = {key: key for key in ("prompt", "chosen", "rejected", "images", "lambda_i", "delta_h", "vdi")}
    info = {}
    for label in ("train", "dev"):
        name = f"car_predpo_{label}_rvdi_only"
        filename = f"{name}.json"
        (out / filename).write_text(json.dumps(result[label], ensure_ascii=False) + "\n", encoding="utf-8")
        info[name] = {"file_name": filename, "formatting": "alpaca", "ranking": True, "columns": columns}
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "method": "rvdi_only_margin", "gamma_of_existing_rvdi": 0.5,
        "margin_parameters": {"lambda_min": 0.03, "lambda_max": 0.20, "alpha": 0, "rho": 0, "eta": 0},
        "source": {label: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for label, path in (("train", args.train), ("dev", args.dev))},
        "count": {label: len(result[label]) for label in ("train", "dev")},
        "reports": reports,
    }
    (out / "rvdi_only_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
