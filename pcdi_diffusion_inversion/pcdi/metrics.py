from __future__ import annotations

import numpy as np


def regression_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = pred - true
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    rel = float(np.linalg.norm(err.ravel()) / (np.linalg.norm(true.ravel()) + 1e-12))
    data_range = float(np.max(true) - np.min(true) + 1e-12)
    nrms = rmse / data_range
    return {"mse": mse, "rmse": rmse, "mae": mae, "rel_error": rel, "nrms": nrms}

