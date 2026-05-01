from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalizer import MinMaxNormalizer


@dataclass
class Section:
    name: str
    model: np.ndarray
    low: np.ndarray
    seismic: np.ndarray


@dataclass(frozen=True)
class SectionSpec:
    name: str
    model_path: str
    low_path: str
    seismic_path: str
    shape: tuple[int, int] | None = None
    model_scale: float = 1.0
    low_scale: float = 1.0
    seismic_scale: float = 1.0
    transpose: bool = True


DEFAULT_SPECS = {
    "marmousi": SectionSpec(
        name="marmousi",
        model_path="data/Marmousi2/Imp_1ms_Sample1701_CMP1001.bin",
        low_path="data/Marmousi2/Imp_1ms_S_Sample1701_CMP1001.bin",
        seismic_path="data/Marmousi2/Seismic_1ms_Sample1701_CMP1001.bin",
        shape=(1001, 1701),
        model_scale=1000.0,
        low_scale=1000.0,
        transpose=True,
    ),
    "overthrust": SectionSpec(
        name="overthrust",
        model_path="data/Overthrust/model_il100.npy",
        low_path="data/Overthrust/model_low_il100.npy",
        seismic_path="data/Overthrust/s_syn_il100.npy",
        transpose=True,
    ),
}


def _load_array(path: Path, shape: tuple[int, int] | None, scale: float) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".bin":
        if shape is None:
            raise ValueError(f"Binary file needs a shape: {path}")
        arr = np.fromfile(path, dtype=np.float32).reshape(shape)
    else:
        raise ValueError(f"Unsupported data file extension: {path}")
    arr = arr.astype(np.float32, copy=False)
    if scale != 1.0:
        arr = arr / np.float32(scale)
    return np.ascontiguousarray(arr)


def load_section(project_root: Path, name: str) -> Section:
    if name not in DEFAULT_SPECS:
        raise KeyError(f"Unknown section '{name}'. Available: {sorted(DEFAULT_SPECS)}")
    spec = DEFAULT_SPECS[name]
    model = _load_array(project_root / spec.model_path, spec.shape, spec.model_scale)
    low = _load_array(project_root / spec.low_path, spec.shape, spec.low_scale)
    seismic = _load_array(project_root / spec.seismic_path, spec.shape, spec.seismic_scale)
    if spec.transpose:
        model = model.T
        low = low.T
        seismic = seismic.T
    if not (model.shape == low.shape == seismic.shape):
        raise ValueError(
            f"{name} shape mismatch: model={model.shape}, low={low.shape}, seismic={seismic.shape}"
        )
    return Section(name=name, model=model, low=low, seismic=seismic)


def load_sections(project_root: Path, names: list[str]) -> list[Section]:
    return [load_section(project_root, name) for name in names]


def fit_normalizer(sections: list[Section]) -> MinMaxNormalizer:
    return MinMaxNormalizer.fit(
        [s.model for s in sections],
        [s.low for s in sections],
        [s.seismic for s in sections],
    )


def grid_starts(length: int, patch: int, stride: int) -> list[int]:
    if length < patch:
        raise ValueError(f"Section dimension {length} is smaller than patch size {patch}")
    starts = list(range(0, length - patch + 1, stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def extract_patch(section: Section, y: int, x: int, patch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy = slice(y, y + patch_size)
    xx = slice(x, x + patch_size)
    return section.model[yy, xx], section.low[yy, xx], section.seismic[yy, xx]


class SeismicPatchDataset(Dataset):
    def __init__(
        self,
        sections: list[Section],
        normalizer: MinMaxNormalizer,
        patch_size: int = 64,
        stride: int = 32,
        max_patches_per_section: int | None = 256,
        seed: int = 1234,
    ) -> None:
        self.sections = sections
        self.normalizer = normalizer
        self.patch_size = patch_size
        rng = np.random.default_rng(seed)
        indices: list[tuple[int, int, int]] = []
        for section_id, section in enumerate(sections):
            h, w = section.model.shape
            coords = [(y, x) for y in grid_starts(h, patch_size, stride) for x in grid_starts(w, patch_size, stride)]
            if max_patches_per_section is not None and len(coords) > max_patches_per_section:
                chosen = rng.choice(len(coords), size=max_patches_per_section, replace=False)
                coords = [coords[int(i)] for i in chosen]
            indices.extend((section_id, y, x) for y, x in coords)
        rng.shuffle(indices)
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        section_id, y, x = self.indices[idx]
        model, low, seismic = extract_patch(self.sections[section_id], y, x, self.patch_size)
        model_n = self.normalizer.model_to_norm_np(model)[None, ...]
        low_n = self.normalizer.cond_to_norm_np(low)[None, ...]
        seismic_n = self.normalizer.seismic_to_norm_np(seismic)[None, ...]
        return torch.from_numpy(model_n), torch.from_numpy(low_n), torch.from_numpy(seismic_n)
