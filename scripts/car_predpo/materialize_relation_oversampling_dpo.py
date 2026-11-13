#!/usr/bin/env python3
"""Materialize strict paired Full CAR and Relation-Oversampling DPO datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from car_predpo.dataset import materialize_relation_oversampling_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a strict Full CAR / DPO+RO paired dataset.")
    parser.add_argument("--source-dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = materialize_relation_oversampling_pair(args.source_dataset_dir, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
