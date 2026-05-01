from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_comparison(true: np.ndarray, low: np.ndarray, pred: np.ndarray, seismic: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), constrained_layout=True)
    model_stack = np.concatenate([true.ravel(), low.ravel(), pred.ravel()])
    vmin, vmax = np.percentile(model_stack, [1, 99])
    panels = [
        (true, "True model", "jet", vmin, vmax),
        (low, "Low-frequency", "jet", vmin, vmax),
        (pred, "PCDI prediction", "jet", vmin, vmax),
        (np.abs(pred - true), "Absolute error", "hot", None, None),
        (seismic, "Observed seismic", "seismic", None, None),
    ]
    for ax, (data, title, cmap, lo, hi) in zip(axes, panels):
        if lo is None and cmap == "seismic":
            amp = np.percentile(np.abs(data), 99)
            lo, hi = -amp, amp
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

