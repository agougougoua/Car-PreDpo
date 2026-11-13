#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_predpo.mdpo import materialize_mdpo_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize isolated mDPO pairwise data.")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-train", required=True, type=int)
    parser.add_argument("--expected-dev", required=True, type=int)
    parser.add_argument("--source-lock", type=Path)
    args = parser.parse_args()
    manifest = materialize_mdpo_data(
        args.train,
        args.dev,
        args.output_dir,
        args.expected_train,
        args.expected_dev,
        args.source_lock,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

