#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from car_predpo.audit import file_sha256
from car_predpo.margin import build_margins
from car_predpo.rvdi import compute_rvdi, empirical_percentiles


ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCORER = ROOT / "LLaMA-Factory/scripts/compute_vdi_for_dpo.py"


def load_legacy_module():
    spec = importlib.util.spec_from_file_location("car_legacy_vdi", LEGACY_SCORER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score multi-perturbation robust VDI with a Stage-1 adapter.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--reference-train", type=Path, help="Scored train JSON used to fit dev percentiles.")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--lambda-min", type=float, default=0.03)
    parser.add_argument("--lambda-max", type=float, default=0.20)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--allow-invalid-margin-distribution", action="store_true")
    return parser.parse_args()


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    legacy = load_legacy_module()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("no input records")
    if any(len(row.get("degraded_images", [])) != 4 for row in rows):
        raise ValueError("every input record must contain exactly four degraded_images")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.output.with_suffix(args.output.suffix + ".progress.json")
    start = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("input_sha256") == file_sha256(args.input) and progress.get("adapter") == str(args.adapter.resolve()):
            rows = progress["rows"]
            start = int(progress["completed"])

    model, processor = legacy.load_model_and_processor(
        str(args.base_model),
        str(args.adapter),
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    failures = []
    for index in tqdm(range(start, len(rows)), desc="CAR-RVDI"):
        try:
            scored = legacy.compute_one_sample_vdi(model, processor, rows[index], device)
            values = [item["vdi_raw"] for item in scored["vdi_per_degrade"] if finite(item.get("vdi_raw"))]
            if len(values) != 4:
                raise ValueError(f"expected four finite VDI scores, got {len(values)}")
            robust = compute_rvdi(values, gamma=args.gamma)
            rows[index].update(scored, **robust)
        except Exception as exc:
            failures.append({"index": index, "record_id": rows[index].get("record_id"), "error": repr(exc)})
            rows[index]["rvdi_error"] = repr(exc)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if (index + 1) % args.save_every == 0:
            progress_path.write_text(
                json.dumps(
                    {
                        "input_sha256": file_sha256(args.input),
                        "adapter": str(args.adapter.resolve()),
                        "completed": index + 1,
                        "rows": rows,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    valid_rows = [row for row in rows if finite(row.get("rvdi_lcb"))]
    if len(valid_rows) != len(rows):
        failure_path = args.output.with_suffix(args.output.suffix + ".failures.json")
        failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(rows) - len(valid_rows)} RVDI failures; see {failure_path}")

    if args.reference_train:
        reference_rows = json.loads(args.reference_train.read_text(encoding="utf-8"))
        reference_values = [float(row["rvdi_lcb"]) for row in reference_rows if finite(row.get("rvdi_lcb"))]
        reference_uncertainty = [float(row["rvdi_std"]) for row in reference_rows if finite(row.get("rvdi_std"))]
    else:
        reference_values = [float(row["rvdi_lcb"]) for row in rows]
        reference_uncertainty = [float(row["rvdi_std"]) for row in rows]
    if not reference_values:
        raise RuntimeError("no finite train reference values")
    uncertainty_scale = float(np.percentile(reference_uncertainty, 95)) or 1.0
    percentiles = empirical_percentiles(reference_values, [float(row["rvdi_lcb"]) for row in rows])
    for row, percentile in zip(rows, percentiles):
        row["percentile"] = percentile
        row["s_perc"] = percentile
        row["vdi"] = float(row["rvdi_lcb"])
        row["rvdi_std_normalized"] = min(1.0, float(row["rvdi_std"]) / uncertainty_scale)
        row.setdefault("delta_h", 1.0)
    enriched, margin_report = build_margins(
        rows,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        alpha=args.alpha,
        rho=args.rho,
        eta=args.eta,
    )
    if not margin_report["valid_for_full_training"] and not args.allow_invalid_margin_distribution:
        raise RuntimeError(f"margin clip fraction exceeds 10%: {margin_report}")

    args.output.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta = {
        "input": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "output_sha256": file_sha256(args.output),
        "base_model": str(args.base_model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "adapter_config_sha256": file_sha256(args.adapter / "adapter_config.json"),
        "reference_train": str(args.reference_train.resolve()) if args.reference_train else None,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "rho": args.rho,
        "eta": args.eta,
        "lambda_range": [args.lambda_min, args.lambda_max],
        "uncertainty_p95_train": uncertainty_scale,
        "total_count": len(rows),
        "valid_count": len(valid_rows),
        "margin_report": margin_report,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
        },
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if progress_path.exists():
        progress_path.unlink()
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
