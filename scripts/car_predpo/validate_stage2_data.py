#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from car_predpo.dataset import validate_split_rows


def resolve_dataset_file(dataset_dir: Path, name: str) -> Path:
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.is_file():
        raise ValueError(f"dataset_info.json does not exist: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if name not in info or not isinstance(info[name].get("file_name"), str):
        raise ValueError(f"dataset is not registered: {name}")
    return dataset_dir / info[name]["file_name"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate isolated CAR Stage 2 datasets.")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--dev-dataset", required=True)
    parser.add_argument("--expected-train", required=True, type=int)
    parser.add_argument("--expected-dev", required=True, type=int)
    args = parser.parse_args()
    try:
        train_path = resolve_dataset_file(args.dataset_dir, args.train_dataset)
        dev_path = resolve_dataset_file(args.dataset_dir, args.dev_dataset)
        if not train_path.is_file():
            raise ValueError(f"train dataset file does not exist: {train_path}")
        if not dev_path.is_file():
            raise ValueError(f"dev dataset file does not exist: {dev_path}")
        train = json.loads(train_path.read_text(encoding="utf-8"))
        dev = json.loads(dev_path.read_text(encoding="utf-8"))
        validate_split_rows(train, dev, args.expected_train, args.expected_dev)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"[CAR preflight] train={len(train)} dev={len(dev)}")


if __name__ == "__main__":
    main()
