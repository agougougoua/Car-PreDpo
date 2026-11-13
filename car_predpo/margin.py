from __future__ import annotations

import math
from collections import defaultdict


def build_margins(
    rows: list[dict],
    lambda_min: float = 0.03,
    lambda_max: float = 0.20,
    alpha: float = 0.25,
    rho: float = 0.25,
    eta: float = 0.5,
    clip_limit: float = 0.10,
) -> tuple[list[dict], dict]:
    enriched = []
    lower = upper = 0
    category_values = defaultdict(list)
    for source in rows:
        row = dict(source)
        base = lambda_min + (lambda_max - lambda_min) * float(row["percentile"])
        relation = float(row.get("relation_confidence", 0.0))
        uncertainty = min(1.0, max(0.0, float(row.get("rvdi_std_normalized", 0.0))))
        length_factor = math.exp(-eta * float(row.get("length_risk", 0.0)))
        raw = base * (1.0 + alpha * relation) * (1.0 - rho * uncertainty) * length_factor
        clipped = min(lambda_max, max(lambda_min, raw))
        lower += raw < lambda_min
        upper += raw > lambda_max
        row.update(
            lambda_base=base,
            relation_factor=1.0 + alpha * relation,
            uncertainty_factor=1.0 - rho * uncertainty,
            length_factor=length_factor,
            lambda_raw=raw,
            lambda_i=clipped,
            margin=clipped * float(row.get("delta_h", 1.0)),
        )
        enriched.append(row)
        for label in row.get("capabilities", ["general"]):
            category_values[label].append(clipped)
    count = max(len(rows), 1)
    report = {
        "count": len(rows),
        "clip_fraction_lower": lower / count,
        "clip_fraction_upper": upper / count,
        "clip_fraction_any": (lower + upper) / count,
        "category_lambda_mean": {
            key: sum(values) / len(values) for key, values in sorted(category_values.items())
        },
    }
    report["valid_for_full_training"] = report["clip_fraction_any"] <= clip_limit
    return enriched, report
