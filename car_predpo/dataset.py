from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


DATASET_COLUMNS = {
    "prompt": "prompt",
    "chosen": "chosen",
    "rejected": "rejected",
    "images": "images",
    "lambda_i": "lambda_i",
    "delta_h": "delta_h",
    "vdi": "vdi",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def validate_split_rows(
    train: list[dict],
    dev: list[dict],
    expected_train: int | None = None,
    expected_dev: int | None = None,
) -> None:
    if not train or not dev:
        raise ValueError("train and dev must both be non-empty")
    if expected_train is not None and len(train) != expected_train:
        raise ValueError(f"expected {expected_train} train rows, found {len(train)}")
    if expected_dev is not None and len(dev) != expected_dev:
        raise ValueError(f"expected {expected_dev} dev rows, found {len(dev)}")
    train_ids = {row.get("record_id") for row in train}
    dev_ids = {row.get("record_id") for row in dev}
    if None in train_ids | dev_ids:
        raise ValueError("record_id missing")
    if train_ids & dev_ids:
        raise ValueError("record_id overlap between train and dev")
    train_images = {row.get("image_key") for row in train}
    dev_images = {row.get("image_key") for row in dev}
    if None in train_images | dev_images:
        raise ValueError("image_key missing")
    if train_images & dev_images:
        raise ValueError("image overlap between train and dev")


def materialize_scored_split(
    scored_path: str | Path,
    split_index_path: str | Path,
    output_dir: str | Path,
    expected_train: int | None = None,
    expected_dev: int | None = None,
) -> dict:
    scored_path = Path(scored_path)
    split_index_path = Path(split_index_path)
    output_dir = Path(output_dir)
    rows = json.loads(scored_path.read_text(encoding="utf-8"))
    split_index = json.loads(split_index_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not isinstance(split_index, dict):
        raise ValueError("scored data must be a list and split index must be an object")

    by_id: dict[str, dict] = {}
    for row in rows:
        record_id = row.get("record_id")
        if record_id in by_id:
            raise ValueError(f"duplicate record_id: {record_id}")
        by_id[record_id] = row
    unknown = sorted(set(split_index) - set(by_id))
    missing = sorted(set(by_id) - set(split_index))
    if unknown:
        raise ValueError(f"unknown record_id in split index: {unknown[0]}")
    if missing:
        raise ValueError(f"record_id missing from split index: {missing[0]}")
    invalid = sorted({value for value in split_index.values() if value not in {"train", "dev"}})
    if invalid:
        raise ValueError(f"invalid split label: {invalid[0]}")

    train = [row for row in rows if split_index[row["record_id"]] == "train"]
    dev = [row for row in rows if split_index[row["record_id"]] == "dev"]
    validate_split_rows(train, dev, expected_train, expected_dev)

    train_path = output_dir / "car_predpo_train_scored.json"
    dev_path = output_dir / "car_predpo_dev_scored.json"
    dataset_info_path = output_dir / "dataset_info.json"
    _atomic_json(train_path, train)
    _atomic_json(dev_path, dev)
    dataset_info = {
        name: {
            "file_name": f"{name}.json",
            "formatting": "alpaca",
            "ranking": True,
            "columns": DATASET_COLUMNS,
        }
        for name in ("car_predpo_train_scored", "car_predpo_dev_scored")
    }
    _atomic_json(dataset_info_path, dataset_info)
    report = {
        "scored_source": str(scored_path.resolve()),
        "scored_sha256": _sha256(scored_path),
        "split_index": str(split_index_path.resolve()),
        "split_index_sha256": _sha256(split_index_path),
        "train_count": len(train),
        "dev_count": len(dev),
        "train_sha256": _sha256(train_path),
        "dev_sha256": _sha256(dev_path),
    }
    _atomic_json(output_dir / "split_manifest.json", report)
    return report



RELATION_OVERSAMPLING_FULL_DATASET = "car_predpo_train_relation_oversampled_full"
RELATION_OVERSAMPLING_DPO_DATASET = "car_predpo_train_relation_oversampled_dpo"
RELATION_OVERSAMPLING_DEV_DATASET = "car_predpo_dev_relation_oversampled"
RELATION_OVERSAMPLING_SEQUENCE_FILE = "sampled_index_sequence.json"
RELATION_OVERSAMPLING_MANIFEST_FILE = "relation_oversampling_manifest.json"
_MARGIN_FIELDS = ("lambda_i", "lambda_raw", "margin")


def _load_registered_relation_source_splits(dataset_dir: Path) -> tuple[list[dict], list[dict], dict]:
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.is_file():
        raise ValueError(f"dataset_info.json does not exist: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(info, dict):
        raise ValueError("dataset_info.json must contain an object")

    def resolve(kind: str) -> tuple[list[dict], str]:
        preferred = f"car_predpo_{kind}_scored"
        candidates = [preferred] if preferred in info else [name for name in info if kind in name]
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one registered {kind} dataset")
        entry = info[candidates[0]]
        filename = entry.get("file_name") if isinstance(entry, dict) else None
        if not isinstance(filename, str):
            raise ValueError(f"registered {kind} dataset has no file_name")
        path = dataset_dir / filename
        if not path.is_file():
            raise ValueError(f"registered {kind} dataset file does not exist: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"registered {kind} dataset must be a JSON list of objects")
        return rows, str(path.resolve())

    train, train_path = resolve("train")
    dev, dev_path = resolve("dev")
    return train, dev, {"train_path": train_path, "dev_path": dev_path}


def _validate_relation_source_train(train: list[dict]) -> dict[str, dict]:
    if not train:
        raise ValueError("source train dataset must be non-empty")
    by_id: dict[str, dict] = {}
    for row in train:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("source train record_id must be a non-empty string")
        if record_id in by_id:
            raise ValueError(f"duplicate source train record_id: {record_id}")
        relation = row.get("relation_confidence")
        if isinstance(relation, bool) or relation not in (0, 0.0, 1, 1.0):
            raise ValueError("relation_confidence must be 0.0 or 1.0")
        missing_margin = [field for field in _MARGIN_FIELDS if field not in row]
        if missing_margin:
            raise ValueError(f"source train row missing margin field: {missing_margin[0]}")
        reserved_audit = [f"{field}_original" for field in _MARGIN_FIELDS if f"{field}_original" in row]
        if reserved_audit:
            raise ValueError(f"source train row has reserved DPO+RO audit field: {reserved_audit[0]}")
        by_id[record_id] = row
    return by_id


def materialize_relation_oversampling_pair(source_dataset_dir: str | Path, output_dir: str | Path) -> dict:
    """Materialize sequence-identical Full CAR and zero-margin DPO+RO train data."""
    source_dataset_dir = Path(source_dataset_dir)
    output_dir = Path(output_dir)
    train, dev, source_paths = _load_registered_relation_source_splits(source_dataset_dir)
    by_id = _validate_relation_source_train(train)
    sequence = [
        row["record_id"]
        for row in train
        for _ in range(2 if row["relation_confidence"] == 1.0 else 1)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_path = output_dir / RELATION_OVERSAMPLING_SEQUENCE_FILE
    if sequence_path.exists():
        persisted_sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
        if persisted_sequence != sequence:
            raise ValueError("existing sampled index sequence does not match source train data")
    else:
        _atomic_json(sequence_path, sequence)

    full = [dict(by_id[record_id]) for record_id in sequence]
    dpo: list[dict] = []
    for row in full:
        control = dict(row)
        for field in _MARGIN_FIELDS:
            control[f"{field}_original"] = control[field]
            control[field] = 0.0
        dpo.append(control)

    full_path = output_dir / f"{RELATION_OVERSAMPLING_FULL_DATASET}.json"
    dpo_path = output_dir / f"{RELATION_OVERSAMPLING_DPO_DATASET}.json"
    dev_path = output_dir / f"{RELATION_OVERSAMPLING_DEV_DATASET}.json"
    _atomic_json(full_path, full)
    _atomic_json(dpo_path, dpo)
    _atomic_json(dev_path, dev)
    dataset_info = {
        name: {
            "file_name": f"{name}.json",
            "formatting": "alpaca",
            "ranking": True,
            "columns": DATASET_COLUMNS,
        }
        for name in (
            RELATION_OVERSAMPLING_FULL_DATASET,
            RELATION_OVERSAMPLING_DPO_DATASET,
            RELATION_OVERSAMPLING_DEV_DATASET,
        )
    }
    _atomic_json(output_dir / "dataset_info.json", dataset_info)
    relation_count = sum(row["relation_confidence"] == 1.0 for row in train)
    report = {
        "control": "relation_oversampling_dpo",
        "source_dataset_dir": str(source_dataset_dir.resolve()),
        "source_train": source_paths["train_path"],
        "source_train_sha256": _sha256(Path(source_paths["train_path"])),
        "source_dev": source_paths["dev_path"],
        "source_dev_sha256": _sha256(Path(source_paths["dev_path"])),
        "sequence_sha256": _sha256(sequence_path),
        "source_train_count": len(train),
        "relation_train_count": relation_count,
        "sequence_count": len(sequence),
        "dev_count": len(dev),
        "relation_sampling_factor": 2,
        "dpo_margin_zero": all(row[field] == 0.0 for row in dpo for field in _MARGIN_FIELDS),
        "full_train_sha256": _sha256(full_path),
        "dpo_train_sha256": _sha256(dpo_path),
        "dev_sha256": _sha256(dev_path),
    }
    _atomic_json(output_dir / RELATION_OVERSAMPLING_MANIFEST_FILE, report)
    return report
