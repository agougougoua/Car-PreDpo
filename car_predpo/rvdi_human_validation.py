from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path


BLIND_FIELDS = [
    "record_id",
    "image_path",
    "prompt",
    "chosen_answer",
    "visual_dependence",
    "notes",
]
KEY_FIELDS = ["record_id", "rvdi_lcb", "rvdi_quintile"]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_scored_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("scored input must be a non-empty JSON list")
    if len({row.get("record_id") for row in rows}) != len(rows):
        raise ValueError("scored input contains duplicate or missing record_id values")
    for row in rows:
        try:
            score = float(row["rvdi_lcb"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"record {row.get('record_id')} has no numeric rvdi_lcb") from exc
        if not math.isfinite(score):
            raise ValueError(f"record {row.get('record_id')} has non-finite rvdi_lcb")
        if not row.get("images") or not row["images"][0]:
            raise ValueError(f"record {row.get('record_id')} has no image")
    return rows


def prepare_annotation_package(
    scored_input: str | Path,
    output_dir: str | Path,
    sample_size: int = 150,
    seed: int = 42,
    strata: int = 5,
) -> dict:
    """Create deterministic, RVDI-stratified and blinded reviewer packets."""
    source = Path(scored_input)
    destination = Path(output_dir)
    rows = _load_scored_rows(source)
    if not 1 <= strata <= len(rows):
        raise ValueError("strata must be between 1 and the number of scored rows")
    if not 1 <= sample_size <= len(rows):
        raise ValueError("sample_size must be between 1 and the number of scored rows")

    ordered = sorted(rows, key=lambda row: (float(row["rvdi_lcb"]), row["record_id"]))
    buckets: list[list[dict]] = [[] for _ in range(strata)]
    for rank, row in enumerate(ordered):
        buckets[min(strata - 1, rank * strata // len(ordered))].append(row)

    allocation = [sample_size // strata + int(index < sample_size % strata) for index in range(strata)]
    if any(count > len(bucket) for count, bucket in zip(allocation, buckets)):
        raise ValueError("requested sample_size cannot be allocated across RVDI strata")
    rng = random.Random(seed)
    selected: list[tuple[int, dict]] = []
    for index, (bucket, count) in enumerate(zip(buckets, allocation), start=1):
        selected.extend((index, row) for row in rng.sample(bucket, count))
    rng.shuffle(selected)

    reviewer_dir = destination / "reviewer"
    control_dir = destination / "control"
    reviewer_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)
    blind_rows = [
        {
            "record_id": row["record_id"],
            "image_path": row["images"][0],
            "prompt": row.get("prompt", ""),
            "chosen_answer": row.get("chosen", ""),
            "visual_dependence": "",
            "notes": "",
        }
        for _, row in selected
    ]
    key_rows = [
        {
            "record_id": row["record_id"],
            "rvdi_lcb": float(row["rvdi_lcb"]),
            "rvdi_quintile": quintile,
        }
        for quintile, row in selected
    ]
    _write_csv(reviewer_dir / "annotator_a.csv", BLIND_FIELDS, blind_rows)
    _write_csv(reviewer_dir / "annotator_b.csv", BLIND_FIELDS, blind_rows)
    _write_csv(control_dir / "annotation_key.csv", KEY_FIELDS, key_rows)
    manifest = {
        "source": str(source.resolve()),
        "candidate_count": len(rows),
        "sample_size": sample_size,
        "seed": seed,
        "strata": strata,
        "allocation": allocation,
        "annotation_scale": {
            "1": "low: the question can be answered without visual evidence",
            "2": "medium: the image helps but is not decisive",
            "3": "high: a correct answer requires visual evidence",
        },
    }
    (control_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (reviewer_dir / "review_guide.md").write_text(
        "# RVDI visual-dependence annotation\n\n"
        "Independently assign exactly one label for each row. Do not infer the label from answer length.\n\n"
        "- 1: low — the question can be answered without visual evidence.\n"
        "- 2: medium — the image helps but is not decisive.\n"
        "- 3: high — a correct answer requires visual evidence from the image.\n",
        encoding="utf-8",
    )
    return manifest


def _read_annotation_labels(path: str | Path, label_field: str = "visual_dependence") -> dict[str, int]:
    labels: dict[str, int] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record_id = row.get("record_id", "")
            if not record_id or record_id in labels:
                raise ValueError(f"annotation file {path} has duplicate or missing record_id")
            try:
                label = int(row[label_field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"record {record_id} has no valid {label_field}") from exc
            if label not in (1, 2, 3):
                raise ValueError(f"record {record_id} has {label_field} outside [1, 3]")
            labels[record_id] = label
    return labels


def _read_key(path: str | Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record_id = row.get("record_id", "")
            if not record_id or record_id in scores:
                raise ValueError(f"annotation key {path} has duplicate or missing record_id")
            try:
                score = float(row["rvdi_lcb"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"record {record_id} has no numeric rvdi_lcb") from exc
            if not math.isfinite(score):
                raise ValueError(f"record {record_id} has non-finite rvdi_lcb")
            scores[record_id] = score
    return scores


def _weighted_cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    import numpy as np

    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("weighted kappa requires equally sized non-empty label lists")
    observed = np.zeros((3, 3), dtype=float)
    for a, b in zip(labels_a, labels_b):
        observed[a - 1, b - 1] += 1
    observed /= len(labels_a)
    marginal_a = observed.sum(axis=1)
    marginal_b = observed.sum(axis=0)
    expected = np.outer(marginal_a, marginal_b)
    positions = np.arange(3)
    weights = ((positions[:, None] - positions[None, :]) / 2.0) ** 2
    expected_disagreement = float((weights * expected).sum())
    if expected_disagreement == 0:
        return 1.0 if labels_a == labels_b else float("nan")
    return 1.0 - float((weights * observed).sum()) / expected_disagreement


def _bootstrap_spearman(
    scores: list[float],
    labels: list[int],
    samples: int,
    seed: int,
) -> list[float]:
    import numpy as np
    from scipy.stats import spearmanr

    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(scores), size=len(scores))
        rho = float(spearmanr(np.asarray(scores)[indices], np.asarray(labels)[indices]).statistic)
        if math.isfinite(rho):
            values.append(rho)
    if not values:
        raise ValueError("bootstrap samples have undefined Spearman correlations")
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def analyze_annotation_package(
    annotation_key: str | Path,
    annotator_a: str | Path,
    annotator_b: str | Path,
    output_dir: str | Path,
    *,
    final_labels: str | Path | None = None,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 42,
) -> dict:
    """Summarize blinded ordinal labels against their withheld RVDI scores."""
    from scipy.stats import spearmanr

    scores_by_id = _read_key(annotation_key)
    labels_a = _read_annotation_labels(annotator_a)
    labels_b = _read_annotation_labels(annotator_b)
    expected_ids = set(scores_by_id)
    if set(labels_a) != expected_ids or set(labels_b) != expected_ids:
        raise ValueError("annotation IDs must exactly match annotation_key IDs")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    ordered_ids = sorted(expected_ids)
    disagreements = [
        record_id for record_id in ordered_ids if labels_a[record_id] != labels_b[record_id]
    ]
    adjudication_rows = [
        {
            "record_id": record_id,
            "annotator_a": labels_a[record_id],
            "annotator_b": labels_b[record_id],
            "final_visual_dependence": "" if record_id in disagreements else labels_a[record_id],
        }
        for record_id in ordered_ids
    ]
    _write_csv(
        destination / "adjudication.csv",
        ["record_id", "annotator_a", "annotator_b", "final_visual_dependence"],
        adjudication_rows,
    )

    if final_labels is None:
        if disagreements:
            raise ValueError(
                f"{len(disagreements)} reviewer disagreements; complete "
                f"{destination / 'adjudication.csv'} and pass it as final_labels"
            )
        labels_final = labels_a
    else:
        labels_final = _read_annotation_labels(final_labels, "final_visual_dependence")
        if set(labels_final) != expected_ids:
            raise ValueError("final-label IDs must exactly match annotation_key IDs")

    scores = [scores_by_id[record_id] for record_id in ordered_ids]
    final_values = [labels_final[record_id] for record_id in ordered_ids]
    rho_result = spearmanr(scores, final_values)
    rho = float(rho_result.statistic)
    if not math.isfinite(rho):
        raise ValueError("Spearman correlation is undefined for constant inputs")
    ci_low, ci_high = _bootstrap_spearman(scores, final_values, bootstrap_samples, bootstrap_seed)
    kappa = _weighted_cohen_kappa(
        [labels_a[record_id] for record_id in ordered_ids],
        [labels_b[record_id] for record_id in ordered_ids],
    )
    summary = {
        "sample_size": len(ordered_ids),
        "reviewer_disagreement_count": len(disagreements),
        "weighted_cohen_kappa": kappa,
        "spearman_rho": rho,
        "spearman_p_value": float(rho_result.pvalue),
        "spearman_bootstrap_95ci": [ci_low, ci_high],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }
    (destination / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grouped = [[score for score, label in zip(scores, final_values) if label == level] for level in (1, 2, 3)]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    axes[0].boxplot(grouped, tick_labels=["Low", "Medium", "High"], showfliers=True)
    axes[0].set_xlabel("Human visual-dependence label")
    axes[0].set_ylabel("RVDI")
    rng = np.random.default_rng(bootstrap_seed)
    axes[1].scatter(
        np.asarray(final_values) + rng.uniform(-0.07, 0.07, size=len(final_values)),
        scores,
        s=18,
        alpha=0.8,
    )
    axes[1].set_xticks([1, 2, 3], ["Low", "Medium", "High"])
    axes[1].set_xlabel("Human visual-dependence label")
    axes[1].set_ylabel("RVDI")
    axes[1].set_title(f"Spearman $\\rho$={rho:.3f}, p={rho_result.pvalue:.3g}")
    figure.savefig(destination / "rvdi_human_validation.png", dpi=300)
    plt.close(figure)
    return summary
