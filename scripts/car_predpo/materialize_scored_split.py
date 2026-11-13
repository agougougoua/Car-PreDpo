#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from car_predpo.dataset import materialize_scored_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize isolated scored CAR train/dev data.")
    parser.add_argument("--scored", required=True, type=Path)
    parser.add_argument("--split-index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-train", type=int)
    parser.add_argument("--expected-dev", type=int)
    args = parser.parse_args()
    report = materialize_scored_split(
        args.scored,
        args.split_index,
        args.output_dir,
        expected_train=args.expected_train,
        expected_dev=args.expected_dev,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
