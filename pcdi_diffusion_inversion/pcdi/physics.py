from __future__ import annotations

import torch
import torch.nn.functional as F

from .normalizer import MinMaxNormalizer


def ricker_wavelet(length: int = 64, dominant_freq: float = 30.0, dt: float = 0.001, device=None) -> torch.Tensor:
    device = device or torch.device("cpu")
    t = torch.arange(length, dtype=torch.float32, device=device) * dt - (length // 2) * dt
    p = torch.pi * dominant_freq * t
    w = (1.0 - 2.0 * p.square()) * torch.exp(-p.square())
    return w / (torch.linalg.vector_norm(w) + 1e-8)


def conv_depth_same(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    k = int(kernel.numel())
    pad_total = k - 1
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    x_pad = F.pad(x, (0, 0, pad_top, pad_bottom))
    weight = kernel.view(1, 1, k, 1).to(dtype=x.dtype, device=x.device)
    return F.conv2d(x_pad, weight)


def forward_model(model_phys: torch.Tensor, wavelet: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Post-stack normal-incidence convolutional forward model.

    Input and output have shape [B, 1, depth, trace].
    """
    m = torch.clamp(model_phys, min=eps)
    reflectivity = torch.zeros_like(m)
    top = m[:, :, :-1, :]
    bottom = m[:, :, 1:, :]
    reflectivity[:, :, 1:, :] = (bottom - top) / (bottom + top + eps)
    return conv_depth_same(reflectivity, wavelet)


def adjoint_guidance(
    x0_norm: torch.Tensor,
    seismic_obs_norm: torch.Tensor,
    normalizer: MinMaxNormalizer,
    wavelet: torch.Tensor,
    clip: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic adjoint gradient of 0.5 ||F(m)-d||^2 in normalized coordinates."""
    model_phys = normalizer.norm_to_model(x0_norm)
    seismic_obs = normalizer.norm_to_seismic(seismic_obs_norm)
    seismic_syn = forward_model(model_phys, wavelet)
    residual = seismic_syn - seismic_obs

    delta = conv_depth_same(residual, torch.flip(wavelet, dims=[0]))
    m = torch.clamp(model_phys, min=eps)
    grad_phys = torch.zeros_like(m)

    top = m[:, :, :-1, :]
    bottom = m[:, :, 1:, :]
    denom2 = (top + bottom + eps).square()
    delta_pair = delta[:, :, 1:, :]

    grad_phys[:, :, :-1, :] += delta_pair * (-2.0 * bottom / denom2)
    grad_phys[:, :, 1:, :] += delta_pair * (2.0 * top / denom2)

    grad_norm = grad_phys * normalizer.model_norm_scale
    rms = torch.sqrt(torch.mean(grad_norm.square(), dim=(1, 2, 3), keepdim=True) + eps)
    direction = torch.clamp(grad_norm / rms, min=-clip, max=clip)
    residual_rms = torch.sqrt(torch.mean(residual.square(), dim=(1, 2, 3)) + eps)
    return direction, residual_rms

