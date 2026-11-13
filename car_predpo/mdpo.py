from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {"record_id", "image_key", "prompt", "chosen", "rejected", "images"}
OUTPUT_FIELDS = ("record_id", "image_key", "prompt", "chosen", "rejected", "images")
SOURCE_LOCK = Path(__file__).resolve().parents[1] / "docs/superpowers/work/2026-08-03-mdpo-source-lock.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
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


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected non-empty JSON list: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected object rows: {path}")
    return rows


def validate_mdpo_rows(rows: list[dict[str, Any]], split_name: str) -> None:
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    for index, row in enumerate(rows):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"{split_name}[{index}] missing fields: {sorted(missing)}")
        record_id = row["record_id"]
        image_key = row["image_key"]
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{split_name}[{index}] record_id must be non-empty")
        if not isinstance(image_key, str) or not image_key:
            raise ValueError(f"{split_name}[{index}] image_key must be non-empty")
        if record_id in seen_ids:
            raise ValueError(f"{split_name} duplicate record_id: {record_id}")
        seen_ids.add(record_id)
        seen_images.add(image_key)
        if not isinstance(row["prompt"], str) or not row["prompt"]:
            raise ValueError(f"{split_name}[{index}] prompt must be non-empty")
        if not isinstance(row["chosen"], str) or not row["chosen"]:
            raise ValueError(f"{split_name}[{index}] chosen must be non-empty")
        if not isinstance(row["rejected"], str) or not row["rejected"]:
            raise ValueError(f"{split_name}[{index}] rejected must be non-empty")
        if row["chosen"] == row["rejected"]:
            raise ValueError(f"{split_name}[{index}] chosen and rejected are equal")
        images = row["images"]
        if not isinstance(images, list) or not images or not all(isinstance(item, str) and item for item in images):
            raise ValueError(f"{split_name}[{index}] images must be a non-empty path list")
        for image_path in images:
            if not Path(image_path).is_file():
                raise ValueError(f"{split_name}[{index}] image does not exist: {image_path}")


def _project_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{field: row[field] for field in OUTPUT_FIELDS} for row in rows]


def materialize_mdpo_data(
    train_path: str | Path,
    dev_path: str | Path,
    output_dir: str | Path,
    expected_train: int,
    expected_dev: int,
    source_lock: str | Path | None = None,
) -> dict[str, Any]:
    train_path, dev_path, output_dir = Path(train_path), Path(dev_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"protected output already exists: {output_dir}")
    train, dev = _load_rows(train_path), _load_rows(dev_path)
    if len(train) != expected_train or len(dev) != expected_dev:
        raise ValueError(f"Unexpected split sizes: train={len(train)}, dev={len(dev)}")
    validate_mdpo_rows(train, "train")
    validate_mdpo_rows(dev, "dev")
    train_ids = {row["record_id"] for row in train}
    dev_ids = {row["record_id"] for row in dev}
    train_images = {row["image_key"] for row in train}
    dev_images = {row["image_key"] for row in dev}
    if train_ids & dev_ids:
        raise ValueError("record_id overlap between train and dev")
    if train_images & dev_images:
        raise ValueError("image_key overlap between train and dev")
    output_dir.mkdir(parents=True)
    train_out = output_dir / "mdpo_train.json"
    dev_out = output_dir / "mdpo_dev.json"
    _atomic_json(train_out, _project_rows(train))
    _atomic_json(dev_out, _project_rows(dev))
    dataset_info = {
        "mdpo_train": {
            "file_name": "mdpo_train.json",
            "formatting": "alpaca",
            "ranking": True,
            "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected", "images": "images"},
        },
        "mdpo_dev": {
            "file_name": "mdpo_dev.json",
            "formatting": "alpaca",
            "ranking": True,
            "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected", "images": "images"},
        },
    }
    info_path = output_dir / "dataset_info.json"
    _atomic_json(info_path, dataset_info)
    lock_path = Path(source_lock) if source_lock else SOURCE_LOCK
    manifest = {
        "method": "mDPO",
        "source_lock": str(lock_path.resolve()),
        "source_lock_sha256": _sha256(lock_path),
        "source": {
            "train": str(train_path.resolve()),
            "dev": str(dev_path.resolve()),
            "train_sha256": _sha256(train_path),
            "dev_sha256": _sha256(dev_path),
        },
        "outputs": {
            "train_sha256": _sha256(train_out),
            "dev_sha256": _sha256(dev_out),
            "dataset_info_sha256": _sha256(info_path),
        },
        "counts": {"train": len(train), "dev": len(dev)},
        "fields": list(OUTPUT_FIELDS),
        "car_fields_excluded": ["lambda_i", "delta_h", "vdi", "margin", "relation_factor", "uncertainty_factor", "length_factor"],
    }
    _atomic_json(output_dir / "mdpo_manifest.json", manifest)
    return manifest

