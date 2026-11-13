#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from car_predpo.rvdi_human_validation import analyze_annotation_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze blinded human visual-dependence labels against RVDI.")
    parser.add_argument("--annotation-key", required=True, type=Path)
    parser.add_argument("--annotator-a", required=True, type=Path)
    parser.add_argument("--annotator-b", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--final-labels", type=Path, help="Completed adjudication CSV when reviewers disagree.")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    summary = analyze_annotation_package(
        args.annotation_key,
        args.annotator_a,
        args.annotator_b,
        args.output_dir,
        final_labels=args.final_labels,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
