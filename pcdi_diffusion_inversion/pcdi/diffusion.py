from __future__ import annotations

import torch
import torch.nn.functional as F

from .normalizer import MinMaxNormalizer
from .physics import adjoint_guidance


class DiffusionSchedule:
    def __init__(
        self,
        timesteps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = "cpu",
    ) -> None:
        self.timesteps = int(timesteps)
        self.device = torch.device(device)
        steps = torch.arange(self.timesteps, dtype=torch.float32, device=self.device)
        betas = beta_start + (beta_end - beta_start) * (1.0 - torch.cos(torch.pi * steps / self.timesteps)) * 0.5
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_om = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_om * noise, noise

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_om = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        return (xt - sqrt_om * noise_pred) / (sqrt_ab + 1e-8)


@torch.no_grad()
def ddim_sample(
    model,
    schedule: DiffusionSchedule,
    cond_norm: torch.Tensor,
    seismic_obs_norm: torch.Tensor | None,
    normalizer: MinMaxNormalizer,
    wavelet: torch.Tensor,
    steps: int = 40,
    cfg_scale: float = 1.0,
    physics_gamma: float = 0.02,
    physics_lambda_max: float = 0.08,
    noise_variance: float = 0.0,
    grad_clip: float = 1.0,
) -> torch.Tensor:
    model.eval()
    device = cond_norm.device
    batch = cond_norm.shape[0]
    xt = torch.randn_like(cond_norm)
    last_x0 = None
    stride = max(1, schedule.timesteps // max(1, steps))
    timesteps = list(range(schedule.timesteps - 1, -1, -stride))
    if timesteps[-1] != 0:
        timesteps.append(0)

    for i, t in enumerate(timesteps[:-1]):
        t_prev = timesteps[i + 1]
        t_tensor = torch.full((batch,), t, device=device, dtype=torch.long)
        noise_cond = model(xt, t_tensor, cond_norm)
        if cfg_scale != 1.0:
            noise_uncond = model(xt, t_tensor, None)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = noise_cond

        alpha_t = schedule.alpha_bars[t]
        alpha_prev = schedule.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        x0 = schedule.predict_x0(xt, t_tensor, noise_pred).clamp(-1.2, 1.2)
        last_x0 = x0
        direction = torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * noise_pred
        xt_next = torch.sqrt(alpha_prev) * x0 + direction

        if seismic_obs_norm is not None and physics_gamma > 0:
            grad, _ = adjoint_guidance(x0, seismic_obs_norm, normalizer, wavelet, clip=grad_clip)
            lam = physics_gamma * float(alpha_t) / (noise_variance + 1e-3)
            lam = min(lam, physics_lambda_max)
            xt_next = xt_next - lam * grad
            last_x0 = (x0 - lam * grad).clamp(-1.2, 1.2)

        xt = xt_next.clamp(-1.5, 1.5)

    if last_x0 is None:
        return xt.clamp(-1.0, 1.0)
    return last_x0.clamp(-1.0, 1.0)


def diffusion_training_loss(
    model,
    schedule: DiffusionSchedule,
    model_norm: torch.Tensor,
    cond_norm: torch.Tensor,
    cond_dropout: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = model_norm.shape[0]
    device = model_norm.device
    t = torch.randint(0, schedule.timesteps, (batch,), device=device)
    xt, noise = schedule.add_noise(model_norm, t)
    cond = cond_norm
    if cond_dropout > 0:
        mask = torch.rand(batch, device=device) < cond_dropout
        if mask.any():
            cond = cond.clone()
            cond[mask] = 0.0
    pred = model(xt, t, cond)
    diff_loss = F.mse_loss(pred, noise)
    x0_pred = schedule.predict_x0(xt, t, pred).clamp(-1.2, 1.2)
    recon_loss = F.l1_loss(x0_pred, model_norm)
    return diff_loss, recon_loss, x0_pred
