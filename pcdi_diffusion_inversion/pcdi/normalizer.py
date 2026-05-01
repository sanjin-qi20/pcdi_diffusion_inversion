from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class MinMaxNormalizer:
    model_min: float
    model_max: float
    cond_min: float
    cond_max: float
    seismic_min: float
    seismic_max: float

    @classmethod
    def fit(cls, model_arrays: list[np.ndarray], cond_arrays: list[np.ndarray], seismic_arrays: list[np.ndarray]):
        return cls(
            model_min=float(min(np.nanmin(x) for x in model_arrays)),
            model_max=float(max(np.nanmax(x) for x in model_arrays)),
            cond_min=float(min(np.nanmin(x) for x in cond_arrays)),
            cond_max=float(max(np.nanmax(x) for x in cond_arrays)),
            seismic_min=float(min(np.nanmin(x) for x in seismic_arrays)),
            seismic_max=float(max(np.nanmax(x) for x in seismic_arrays)),
        )

    @staticmethod
    def _to_norm_np(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        scale = max(vmax - vmin, 1e-8)
        return ((x - vmin) / scale * 2.0 - 1.0).astype(np.float32)

    @staticmethod
    def _from_norm_np(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        return (((x + 1.0) * 0.5) * (vmax - vmin) + vmin).astype(np.float32)

    @staticmethod
    def _to_norm_torch(x: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
        scale = max(vmax - vmin, 1e-8)
        return (x - vmin) / scale * 2.0 - 1.0

    @staticmethod
    def _from_norm_torch(x: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
        return ((x + 1.0) * 0.5) * (vmax - vmin) + vmin

    @property
    def model_norm_scale(self) -> float:
        return max(self.model_max - self.model_min, 1e-8) * 0.5

    def model_to_norm_np(self, x: np.ndarray) -> np.ndarray:
        return self._to_norm_np(x, self.model_min, self.model_max)

    def cond_to_norm_np(self, x: np.ndarray) -> np.ndarray:
        return self._to_norm_np(x, self.cond_min, self.cond_max)

    def seismic_to_norm_np(self, x: np.ndarray) -> np.ndarray:
        return self._to_norm_np(x, self.seismic_min, self.seismic_max)

    def norm_to_model_np(self, x: np.ndarray) -> np.ndarray:
        return self._from_norm_np(x, self.model_min, self.model_max)

    def norm_to_seismic_np(self, x: np.ndarray) -> np.ndarray:
        return self._from_norm_np(x, self.seismic_min, self.seismic_max)

    def model_to_norm(self, x: torch.Tensor) -> torch.Tensor:
        return self._to_norm_torch(x, self.model_min, self.model_max)

    def cond_to_norm(self, x: torch.Tensor) -> torch.Tensor:
        return self._to_norm_torch(x, self.cond_min, self.cond_max)

    def seismic_to_norm(self, x: torch.Tensor) -> torch.Tensor:
        return self._to_norm_torch(x, self.seismic_min, self.seismic_max)

    def norm_to_model(self, x: torch.Tensor) -> torch.Tensor:
        return self._from_norm_torch(x, self.model_min, self.model_max)

    def norm_to_seismic(self, x: torch.Tensor) -> torch.Tensor:
        return self._from_norm_torch(x, self.seismic_min, self.seismic_max)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MinMaxNormalizer":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

