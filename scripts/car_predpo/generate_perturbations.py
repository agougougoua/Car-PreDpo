#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from car_predpo.audit import file_sha256
from car_predpo.perturb import DEFAULT_CONFIG, make_perturbations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic semantic-preserving CAR perturbations.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    args.image_dir.mkdir(parents=True, exist_ok=True)
    enriched = []
    for row in rows:
        image_path = Path(row["images"][0])
        with Image.open(image_path) as image:
            outputs, metadata = make_perturbations(image)
        paths = []
        for name, changed in outputs.items():
            path = args.image_dir / f'{row["record_id"]}_{name}.jpg'
            changed.save(path, "JPEG", quality=95)
            paths.append(str(path.resolve()))
        item = dict(row)
        item["degraded_images"] = paths
        item["degradation_names"] = list(outputs)
        item["degradation_config"] = metadata
        enriched.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "input": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": file_sha256(args.output),
        "count": len(enriched),
        "k": 4,
        "config": DEFAULT_CONFIG,
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
