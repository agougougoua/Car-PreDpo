#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from car_predpo.rvdi_human_validation import prepare_annotation_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a blinded, RVDI-stratified human-validation package.")
    parser.add_argument("--input", required=True, type=Path, help="Scored dev JSON containing record_id and rvdi_lcb.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strata", type=int, default=5)
    args = parser.parse_args()
    report = prepare_annotation_package(args.input, args.output_dir, args.sample_size, args.seed, args.strata)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
