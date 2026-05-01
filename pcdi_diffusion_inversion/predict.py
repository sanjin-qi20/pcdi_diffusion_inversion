from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pcdi import config as cfg
from pcdi.data import extract_patch, grid_starts, load_section
from pcdi.diffusion import DiffusionSchedule, ddim_sample
from pcdi.metrics import regression_metrics
from pcdi.model import ConditionalDiT, DirectInversionCNN
from pcdi.normalizer import MinMaxNormalizer
from pcdi.physics import ricker_wavelet
from pcdi.visualize import save_comparison


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class PredictConfig:
    checkpoint: str
    full_section: bool
    datasets: list[str]
    patch_dataset: str
    patch_y: int
    patch_x: int
    section_stride: int
    predict_batch_size: int
    sampler: str
    ddim_steps: int
    cfg_scale: float
    physics_gamma: float
    physics_lambda_max: float
    use_physics_guidance: bool
    multi_gpu: bool
    device: str


def latest_checkpoint() -> Path:
    checkpoints = sorted((PROJECT_ROOT / "runs").glob("*/checkpoint.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found under pcdi_diffusion_inversion/runs")
    return checkpoints[-1]


def active_config() -> PredictConfig:
    smoke = cfg.RUN_MODE == "smoke"
    device = cfg.resolve_device()
    run_name = cfg.SMOKE_RUN_NAME if smoke else cfg.RUN_NAME
    return PredictConfig(
        checkpoint=cfg.checkpoint_path(run_name),
        full_section=False if smoke else cfg.FULL_SECTION_PREDICT,
        datasets=cfg.PREDICT_DATASETS,
        patch_dataset=cfg.PATCH_DATASET,
        patch_y=cfg.PATCH_Y,
        patch_x=cfg.PATCH_X,
        section_stride=cfg.SECTION_STRIDE,
        predict_batch_size=cfg.PREDICT_BATCH_SIZE,
        sampler=cfg.SAMPLER,
        ddim_steps=cfg.SMOKE_DDIM_STEPS if smoke else cfg.DDIM_STEPS,
        cfg_scale=cfg.CFG_SCALE,
        physics_gamma=cfg.PHYSICS_GAMMA,
        physics_lambda_max=cfg.PHYSICS_LAMBDA_MAX,
        use_physics_guidance=cfg.USE_PHYSICS_GUIDANCE,
        multi_gpu=cfg.use_multi_gpu(device),
        device=device,
    )


def load_inference_state(checkpoint_path: Path, device: torch.device, multi_gpu: bool):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_config = checkpoint["config"]
    normalizer = MinMaxNormalizer(**checkpoint["normalizer"])
    model = ConditionalDiT(
        image_size=train_config["patch_size"],
        patch_size=train_config["token_patch_size"],
        hidden_size=train_config["hidden_size"],
        depth=train_config["depth"],
        heads=train_config["heads"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    direct_model = None
    if "direct_model" in checkpoint:
        direct_model = DirectInversionCNN(
            base_channels=train_config.get("direct_base_channels", 96),
            depth=train_config.get("direct_depth", 8),
        ).to(device)
        direct_model.load_state_dict(checkpoint["direct_model"])
        direct_model.eval()

    use_multi_gpu = multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1
    if use_multi_gpu:
        model = torch.nn.DataParallel(model)
        if direct_model is not None:
            direct_model = torch.nn.DataParallel(direct_model)
    schedule = DiffusionSchedule(timesteps=train_config["diffusion_steps"], device=device)
    wavelet = ricker_wavelet(device=device)
    return train_config, normalizer, model, direct_model, schedule, wavelet, use_multi_gpu


def infer_norm(
    predict_cfg: PredictConfig,
    model,
    direct_model,
    schedule: DiffusionSchedule,
    normalizer: MinMaxNormalizer,
    wavelet: torch.Tensor,
    cond: torch.Tensor,
    seismic_norm: torch.Tensor,
) -> torch.Tensor:
    if predict_cfg.sampler == "direct" and direct_model is not None:
        with torch.no_grad():
            return direct_model(cond, seismic_norm)
    return ddim_sample(
        model=model,
        schedule=schedule,
        cond_norm=cond,
        seismic_obs_norm=seismic_norm,
        normalizer=normalizer,
        wavelet=wavelet,
        steps=predict_cfg.ddim_steps,
        cfg_scale=predict_cfg.cfg_scale,
        physics_gamma=predict_cfg.physics_gamma if predict_cfg.use_physics_guidance else 0.0,
        physics_lambda_max=predict_cfg.physics_lambda_max,
    )


def predict_patch(
    predict_cfg: PredictConfig,
    checkpoint_path: Path,
    train_config: dict,
    normalizer: MinMaxNormalizer,
    model: ConditionalDiT,
    direct_model: DirectInversionCNN | None,
    schedule: DiffusionSchedule,
    wavelet: torch.Tensor,
    device: torch.device,
) -> None:
    section = load_section(PROJECT_ROOT, predict_cfg.patch_dataset)
    patch_size = int(train_config["patch_size"])
    y = min(max(predict_cfg.patch_y, 0), section.model.shape[0] - patch_size)
    x = min(max(predict_cfg.patch_x, 0), section.model.shape[1] - patch_size)
    true, low, seismic = extract_patch(section, y, x, patch_size)

    cond = torch.from_numpy(normalizer.cond_to_norm_np(low)[None, None]).to(device)
    seismic_norm = torch.from_numpy(normalizer.seismic_to_norm_np(seismic)[None, None]).to(device)
    pred_norm = infer_norm(predict_cfg, model, direct_model, schedule, normalizer, wavelet, cond, seismic_norm)
    pred = normalizer.norm_to_model_np(pred_norm.squeeze().detach().cpu().numpy())

    out_dir = checkpoint_path.parent / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{predict_cfg.patch_dataset}_y{y}_x{x}"
    np.save(out_dir / f"{stem}_prediction.npy", pred.astype(np.float32))
    metrics = regression_metrics(true, pred)
    (out_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_comparison(true, low, pred, seismic, out_dir / f"{stem}_comparison.png")

    print(f"Saved prediction: {out_dir / f'{stem}_prediction.npy'}")
    print(f"Saved figure: {out_dir / f'{stem}_comparison.png'}")
    print(json.dumps(metrics, indent=2))


def blend_weight(patch_size: int) -> np.ndarray:
    ramp = np.hanning(patch_size).astype(np.float32)
    ramp = np.maximum(ramp, 0.08)
    return np.outer(ramp, ramp).astype(np.float32)


def predict_full_section(
    predict_cfg: PredictConfig,
    dataset_name: str,
    checkpoint_path: Path,
    train_config: dict,
    normalizer: MinMaxNormalizer,
    model: ConditionalDiT,
    direct_model: DirectInversionCNN | None,
    schedule: DiffusionSchedule,
    wavelet: torch.Tensor,
    device: torch.device,
) -> None:
    section = load_section(PROJECT_ROOT, dataset_name)
    patch_size = int(train_config["patch_size"])
    stride = max(1, int(predict_cfg.section_stride))
    batch_size = max(1, int(predict_cfg.predict_batch_size))
    y_starts = grid_starts(section.model.shape[0], patch_size, stride)
    x_starts = grid_starts(section.model.shape[1], patch_size, stride)
    coords = [(y, x) for y in y_starts for x in x_starts]
    acc = np.zeros(section.model.shape, dtype=np.float32)
    weights = np.zeros(section.model.shape, dtype=np.float32)
    patch_weight = blend_weight(patch_size)

    print(
        f"Predicting full {dataset_name} section: shape={section.model.shape}, "
        f"patches={len(coords)}, batch_size={batch_size}, stride={stride}"
    )
    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start : start + batch_size]
        low_patches = []
        seismic_patches = []
        for y, x in batch_coords:
            _, low, seismic = extract_patch(section, y, x, patch_size)
            low_patches.append(normalizer.cond_to_norm_np(low))
            seismic_patches.append(normalizer.seismic_to_norm_np(seismic))

        cond = torch.from_numpy(np.stack(low_patches)[:, None]).to(device)
        seismic_norm = torch.from_numpy(np.stack(seismic_patches)[:, None]).to(device)
        pred_norm = infer_norm(predict_cfg, model, direct_model, schedule, normalizer, wavelet, cond, seismic_norm)
        pred = normalizer.norm_to_model_np(pred_norm.squeeze(1).detach().cpu().numpy())

        for item, (y, x) in enumerate(batch_coords):
            yy = slice(y, y + patch_size)
            xx = slice(x, x + patch_size)
            acc[yy, xx] += pred[item] * patch_weight
            weights[yy, xx] += patch_weight

        done = min(start + batch_size, len(coords))
        print(f"  {dataset_name}: {done}/{len(coords)} patches")

    section_pred = acc / np.maximum(weights, 1e-8)
    out_dir = checkpoint_path.parent / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_name}_full_section"
    np.save(out_dir / f"{stem}_prediction.npy", section_pred.astype(np.float32))
    metrics = regression_metrics(section.model, section_pred)
    (out_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_comparison(section.model, section.low, section_pred, section.seismic, out_dir / f"{stem}_comparison.png")

    print(f"Saved full-section prediction: {out_dir / f'{stem}_prediction.npy'}")
    print(f"Saved full-section figure: {out_dir / f'{stem}_comparison.png'}")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    predict_cfg = active_config()
    checkpoint_path = Path(predict_cfg.checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = latest_checkpoint()
    device = torch.device(predict_cfg.device)
    train_config, normalizer, model, direct_model, schedule, wavelet, use_multi_gpu = load_inference_state(
        checkpoint_path,
        device,
        predict_cfg.multi_gpu,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Sampler: {predict_cfg.sampler}")
    print(f"Multi-GPU DataParallel: {use_multi_gpu}")
    if predict_cfg.sampler == "direct" and direct_model is None:
        print("Direct model not found in checkpoint; falling back to diffusion sampler.")
        predict_cfg.sampler = "diffusion"

    if predict_cfg.full_section:
        for dataset_name in predict_cfg.datasets:
            predict_full_section(
                predict_cfg,
                dataset_name,
                checkpoint_path,
                train_config,
                normalizer,
                model,
                direct_model,
                schedule,
                wavelet,
                device,
            )
    else:
        predict_patch(
            predict_cfg,
            checkpoint_path,
            train_config,
            normalizer,
            model,
            direct_model,
            schedule,
            wavelet,
            device,
        )


if __name__ == "__main__":
    main()

