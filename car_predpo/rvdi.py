from __future__ import annotations

import numpy as np


def compute_rvdi(vdi_values: list[float], gamma: float = 0.5) -> dict[str, float]:
    if not vdi_values:
        raise ValueError("at least one VDI value is required")
    values = np.asarray(vdi_values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return {"rvdi_mean": mean, "rvdi_std": std, "rvdi_lcb": mean - gamma * std}


def empirical_percentiles(train_values: list[float], values: list[float]) -> list[float]:
    if not train_values:
        raise ValueError("train_values cannot be empty")
    reference = np.sort(np.asarray(train_values, dtype=np.float64))
    return [float(np.searchsorted(reference, value, side="right") / len(reference)) for value in values]

