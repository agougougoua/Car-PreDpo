from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from PIL import Image


RULE_VERSION = "car-capability-rules-v1"
RULES = {
    "relation_spatial": (
        "left of", "right of", "above", "below", "inside", "outside",
        "in front of", "behind", "beside", "next to", "under", "over",
    ),
    "relation_action": (
        "holding", "riding", "wearing", "sitting on", "standing beside",
        "carrying", "eating", "drinking", "playing", "driving",
    ),
    "relation_comparison": (
        "larger", "smaller", "closer", "farther", "more than", "fewer",
        "taller", "shorter", "bigger",
    ),
    "attribute": (
        "color", "red", "blue", "green", "black", "white", "yellow",
        "number", "material", "wooden", "metal", "striped",
    ),
    "existence": (
        "there is", "there are", "no ", "not ", "missing", "absent",
        "present", "exist", "without", "has ", "have ",
    ),
}


def validate_record(row: dict) -> list[str]:
    errors: list[str] = []
    for key in ("prompt", "chosen", "rejected"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            errors.append(f"{key}_invalid")
    if isinstance(row.get("chosen"), str) and row.get("chosen") == row.get("rejected"):
        errors.append("chosen_equals_rejected")
    images = row.get("images")
    if not isinstance(images, list) or not images:
        errors.append("images_empty")
        return errors
    for value in images:
        if not isinstance(value, str):
            errors.append("image_path_invalid")
            continue
        path = Path(value)
        if not path.is_file():
            errors.append("image_missing")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            errors.append("image_unreadable")
    return sorted(set(errors))


def annotate_capabilities(prompt: str, chosen: str, rejected: str) -> tuple[list[str], dict[str, list[str]]]:
    text = " ".join((prompt, chosen, rejected)).lower()
    evidence = {label: [term for term in terms if term in text] for label, terms in RULES.items()}
    labels = [label for label, hits in evidence.items() if hits]
    if not labels:
        labels = ["general"]
        evidence["general"] = ["no_rule_match"]
    return labels, {key: value for key, value in evidence.items() if value}


def length_features(chosen: str, rejected: str, tokenize: Callable[[str], int]) -> dict[str, float | int]:
    chosen_chars, rejected_chars = len(chosen), len(rejected)
    chosen_words, rejected_words = len(chosen.split()), len(rejected.split())
    chosen_tokens, rejected_tokens = tokenize(chosen), tokenize(rejected)
    risk = abs(chosen_tokens - rejected_tokens) / max(chosen_tokens, rejected_tokens, 1)
    return {
        "chosen_char_length": chosen_chars,
        "rejected_char_length": rejected_chars,
        "chosen_word_length": chosen_words,
        "rejected_word_length": rejected_words,
        "chosen_token_length": chosen_tokens,
        "rejected_token_length": rejected_tokens,
        "length_risk": risk,
    }


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stratified_group_split(rows: list[dict], dev_fraction: float, seed: int) -> dict[str, str]:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["image_key"]].append(row)
    rng = random.Random(seed)
    ordered = list(groups)
    rng.shuffle(ordered)
    ordered.sort(key=lambda key: (-len(groups[key]), key))
    target_rows = max(1, round(len(rows) * dev_fraction))
    overall = Counter(label for row in rows for label in row.get("capabilities", ["general"]))
    dev_counts: Counter[str] = Counter()
    dev_keys: set[str] = set()
    dev_rows = 0
    remaining = set(ordered)
    while remaining and dev_rows < target_rows:
        best_key, best_score = None, None
        for key in remaining:
            trial = dev_counts + Counter(label for row in groups[key] for label in row.get("capabilities", ["general"]))
            new_size = dev_rows + len(groups[key])
            balance = sum(
                abs((trial[label] / max(new_size, 1)) - (count / len(rows)))
                for label, count in overall.items()
            )
            size_error = abs(new_size - target_rows) / len(rows)
            score = (balance + size_error, key)
            if best_score is None or score < best_score:
                best_key, best_score = key, score
        dev_keys.add(best_key)
        remaining.remove(best_key)
        dev_rows += len(groups[best_key])
        dev_counts.update(label for row in groups[best_key] for label in row.get("capabilities", ["general"]))
    return {
        row["record_id"]: ("dev" if row["image_key"] in dev_keys else "train")
        for row in rows
    }

