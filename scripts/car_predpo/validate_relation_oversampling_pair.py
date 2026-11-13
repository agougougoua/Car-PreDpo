#!/usr/bin/env python3
"""Validate that Full CAR and DPO+RO use the same relation-oversampling data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from car_predpo.dataset import (
    RELATION_OVERSAMPLING_DEV_DATASET,
    RELATION_OVERSAMPLING_DPO_DATASET,
    RELATION_OVERSAMPLING_FULL_DATASET,
    RELATION_OVERSAMPLING_SEQUENCE_FILE,
    DATASET_COLUMNS,
    _MARGIN_FIELDS,
    _load_registered_relation_source_splits,
    _validate_relation_source_train,
)


def _validate_paired_dataset_registration(paired_dataset_dir: Path) -> None:
    info_path = paired_dataset_dir / "dataset_info.json"
    if not info_path.is_file():
        raise ValueError("paired dataset registration mismatch: dataset_info.json is missing")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(info, dict):
        raise ValueError("paired dataset registration mismatch: dataset_info.json must be an object")
    for name in (
        RELATION_OVERSAMPLING_FULL_DATASET,
        RELATION_OVERSAMPLING_DPO_DATASET,
        RELATION_OVERSAMPLING_DEV_DATASET,
    ):
        expected = {
            "file_name": f"{name}.json",
            "formatting": "alpaca",
            "ranking": True,
            "columns": DATASET_COLUMNS,
        }
        if info.get(name) != expected:
            raise ValueError(f"paired dataset registration mismatch: {name}")


def _read_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"paired dataset must be a JSON list of objects: {path}")
    return rows


def validate_relation_oversampling_pair(source_dataset_dir: str | Path, paired_dataset_dir: str | Path) -> dict:
    source_dataset_dir = Path(source_dataset_dir)
    paired_dataset_dir = Path(paired_dataset_dir)
    source_train, source_dev, _ = _load_registered_relation_source_splits(source_dataset_dir)
    by_id = _validate_relation_source_train(source_train)
    expected_sequence = [
        row["record_id"]
        for row in source_train
        for _ in range(2 if row["relation_confidence"] == 1.0 else 1)
    ]
    _validate_paired_dataset_registration(paired_dataset_dir)
    sequence = json.loads((paired_dataset_dir / RELATION_OVERSAMPLING_SEQUENCE_FILE).read_text(encoding="utf-8"))
    if sequence != expected_sequence:
        raise ValueError("sampled index sequence does not match source relation expansion")

    full = _read_rows(paired_dataset_dir / f"{RELATION_OVERSAMPLING_FULL_DATASET}.json")
    dpo = _read_rows(paired_dataset_dir / f"{RELATION_OVERSAMPLING_DPO_DATASET}.json")
    dev = _read_rows(paired_dataset_dir / f"{RELATION_OVERSAMPLING_DEV_DATASET}.json")
    if [row.get("record_id") for row in full] != sequence:
        raise ValueError("Full CAR record_id sequence does not match sampled index sequence")
    if [row.get("record_id") for row in dpo] != sequence:
        raise ValueError("DPO+RO record_id sequence does not match sampled index sequence")
    if dev != source_dev:
        raise ValueError("DPO+RO dev dataset differs from source dev dataset")

    for index, (full_row, dpo_row) in enumerate(zip(full, dpo)):
        record_id = sequence[index]
        if full_row != by_id[record_id]:
            raise ValueError("Full CAR row differs from source train row")
        restored = dict(dpo_row)
        for field in _MARGIN_FIELDS:
            if dpo_row.get(field) != 0.0:
                raise ValueError(f"DPO+RO margin field must be zero: {field}")
            original = f"{field}_original"
            if dpo_row.get(original) != full_row.get(field):
                raise ValueError(f"DPO+RO original margin field differs from Full CAR: {field}")
            restored[field] = restored.pop(original)
        if restored != full_row:
            raise ValueError("DPO+RO row differs from Full CAR outside approved margin fields")
    if len(full) != len(dpo):
        raise ValueError("Full CAR and DPO+RO train lengths differ")
    return {"source_train_count": len(source_train), "sequence_count": len(sequence), "dev_count": len(dev)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a strict Full CAR / DPO+RO paired dataset.")
    parser.add_argument("--source-dataset-dir", required=True, type=Path)
    parser.add_argument("--paired-dataset-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = validate_relation_oversampling_pair(args.source_dataset_dir, args.paired_dataset_dir)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"[DPO+RO preflight] source_train={report['source_train_count']} sequence={report['sequence_count']} dev={report['dev_count']}")


if __name__ == "__main__":
    main()
