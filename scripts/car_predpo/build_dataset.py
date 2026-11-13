#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from car_predpo.audit import (
    RULE_VERSION,
    annotate_capabilities,
    file_sha256,
    length_features,
    stratified_group_split,
    validate_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit, annotate and split CAR-PreDPO preference data.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=20260723)
    parser.add_argument("--test-image-root", action="append", default=[])
    return parser.parse_args()


def image_digest(path: Path) -> str:
    return file_sha256(path)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("input must be a non-empty JSON list")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokenize = lambda text: len(tokenizer.encode(text, add_special_tokens=False))

    test_hashes: set[str] = set()
    test_files = 0
    for root_value in args.test_image_root:
        root = Path(root_value)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                try:
                    test_hashes.add(image_digest(path))
                    test_files += 1
                except OSError:
                    pass

    audited: list[dict] = []
    errors: dict[str, list[str]] = {}
    duplicate_pairs = Counter()
    image_counts = Counter()
    leakage_ids: list[str] = []
    for index, source in enumerate(rows):
        record_id = f"car-{index:06d}"
        row_errors = validate_record(source)
        if row_errors:
            errors[record_id] = row_errors
            continue
        image = Path(source["images"][0]).resolve()
        digest = image_digest(image)
        pair_key = hashlib.sha256(
            json.dumps(
                [source["prompt"], source["chosen"], source["rejected"], [str(Path(x).resolve()) for x in source["images"]]],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        duplicate_pairs[pair_key] += 1
        image_counts[digest] += 1
        if digest in test_hashes:
            leakage_ids.append(record_id)
        labels, evidence = annotate_capabilities(source["prompt"], source["chosen"], source["rejected"])
        lengths = length_features(source["chosen"], source["rejected"], tokenize)
        enriched = dict(source)
        enriched.update(
            record_id=record_id,
            image_key=digest,
            capabilities=labels,
            capability_evidence=evidence,
            capability_rule_version=RULE_VERSION,
            relation_confidence=float(any(label.startswith("relation_") for label in labels)),
            verbosity_risk=bool(lengths["length_risk"] >= 0.5),
            **lengths,
        )
        audited.append(enriched)

    if errors:
        write_json(args.output_dir / "invalid_records.json", errors)
        raise ValueError(f"{len(errors)} invalid records; see invalid_records.json")
    if leakage_ids:
        write_json(args.output_dir / "test_leakage.json", leakage_ids)
        leakage_set = set(leakage_ids)
        audited = [row for row in audited if row["record_id"] not in leakage_set]
    if not audited:
        raise ValueError("no records remain after benchmark leakage exclusion")

    split = stratified_group_split(audited, args.dev_fraction, args.split_seed)
    train = [row for row in audited if split[row["record_id"]] == "train"]
    dev = [row for row in audited if split[row["record_id"]] == "dev"]
    train_images = {row["image_key"] for row in train}
    dev_images = {row["image_key"] for row in dev}
    if train_images & dev_images:
        raise AssertionError("image leakage between train and dev")

    train_path = args.output_dir / "train.json"
    dev_path = args.output_dir / "dev.json"
    write_json(train_path, train)
    write_json(dev_path, dev)
    write_json(args.output_dir / "split_index.json", split)
    report = {
        "source": str(args.input.resolve()),
        "source_sha256": file_sha256(args.input),
        "rule_version": RULE_VERSION,
        "split_seed": args.split_seed,
        "dev_fraction": args.dev_fraction,
        "input_count": len(rows),
        "excluded_test_overlap_count": len(leakage_ids),
        "eligible_count": len(audited),
        "train_count": len(train),
        "dev_count": len(dev),
        "unique_image_count": len(image_counts),
        "duplicate_image_count": sum(count > 1 for count in image_counts.values()),
        "duplicate_pair_count": sum(count - 1 for count in duplicate_pairs.values() if count > 1),
        "test_hash_roots_requested": args.test_image_root,
        "test_hash_files_resolved": test_files,
        "test_hash_unique": len(test_hashes),
        "test_hash_overlap_count": len(leakage_ids),
        "leakage_guarantee": "content-hash checked only for resolved test-image roots",
        "train_capabilities": Counter(label for row in train for label in row["capabilities"]),
        "dev_capabilities": Counter(label for row in dev for label in row["capabilities"]),
        "train_sha256": file_sha256(train_path),
        "dev_sha256": file_sha256(dev_path),
    }
    write_json(args.output_dir / "audit_manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
