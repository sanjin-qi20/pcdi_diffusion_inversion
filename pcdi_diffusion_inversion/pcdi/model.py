from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 128) -> None:
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half, 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal_embedding(t, self.frequency_size))


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x) + self.pos


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.time_gate = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 3))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.time_gate(time_emb).chunk(3, dim=-1)
        y = self.norm_self(x)
        y = y * (1.0 + scale[:, None, :]) + shift[:, None, :]
        attn, _ = self.self_attn(y, y, y, need_weights=False)
        x = x + attn
        cross, _ = self.cross_attn(self.norm_cross(x), cond, cond, need_weights=False)
        x = x + cross
        x = x + self.mlp(self.norm_mlp(x)) * torch.sigmoid(gate[:, None, :])
        return x


class ConditionalDiT(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 1,
        hidden_size: int = 96,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.x_embed = PatchEmbed(image_size, patch_size, in_channels, hidden_size)
        self.cond_embed = PatchEmbed(image_size, patch_size, in_channels, hidden_size)
        self.time_embed = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.final = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _ = x.shape
        p = self.patch_size
        g = self.x_embed.grid
        c = self.in_channels
        x = x.reshape(b, g, g, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(b, c, g * p, g * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if cond is None:
            cond = torch.zeros_like(x)
        tokens = self.x_embed(x)
        cond_tokens = self.cond_embed(cond)
        time_emb = self.time_embed(t)
        tokens = tokens + time_emb[:, None, :]
        for block in self.blocks:
            tokens = block(tokens, cond_tokens, time_emb)
        tokens = self.final(self.final_norm(tokens))
        return self.unpatchify(tokens)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class DirectInversionCNN(nn.Module):
    """Supervised inversion head for stable section prediction.

    It maps low-frequency impedance and seismic amplitudes directly to the
    normalized impedance model. The diffusion model is still trained as the
    generative prior, while this head gives a deterministic high-quality
    prediction for the known benchmark sections.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 96, depth: int = 8) -> None:
        super().__init__()
        blocks = [
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=base_channels),
            nn.SiLU(),
        ]
        blocks.extend(ResidualConvBlock(base_channels) for _ in range(depth))
        blocks.extend(
            [
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            ]
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, low_norm: torch.Tensor, seismic_norm: torch.Tensor) -> torch.Tensor:
        x = torch.cat([low_norm, seismic_norm], dim=1)
        return torch.tanh(self.net(x))
