from __future__ import annotations

import torch

# =========================
# Edit these variables first
# =========================

# Run mode: "smoke", "train", "predict", or "train_predict".
RUN_MODE = "train_predict"
RUN_NAME = "pcdi_full"
CHECKPOINT_PATH = None  # Example: "runs/pcdi_full/checkpoint.pt"; None uses RUN_NAME.

# Data usage. This is the most important data-volume control.
# 0.70 means use 70% of all available sliding-window patches from each dataset.
TRAIN_PATCH_RATIO = 0.50
DATASETS = ["marmousi", "overthrust"]

# Hardware.
DEVICE = "auto"  # "auto", "cuda", or "cpu".
USE_MULTI_GPU = True
PAUSE_AT_END = True

# Training.
EPOCHS = 1000
BATCH_SIZE = 48
PATCH_SIZE = 64
TRAIN_STRIDE = 32
DIFFUSION_STEPS = 200
TOKEN_PATCH_SIZE = 4
HIDDEN_SIZE = 128
TRANSFORMER_DEPTH = 6
NUM_HEADS = 8
LEARNING_RATE = 2e-4
COND_DROPOUT = 0.1
SAVE_EVERY = 50

# Loss weights.
RECON_LOSS_WEIGHT = 0.5
PHYSICS_LOSS_WEIGHT = 0.05
DIRECT_LOSS_WEIGHT = 2.0

# Direct inversion head. This gives a stable full-section prediction path.
DIRECT_BASE_CHANNELS = 192
DIRECT_DEPTH = 12

# Prediction.
FULL_SECTION_PREDICT = True
PREDICT_DATASETS = ["marmousi", "overthrust"]
PATCH_DATASET = "overthrust"
PATCH_Y = 64
PATCH_X = 320
SECTION_STRIDE = 32
PREDICT_BATCH_SIZE = 32
SAMPLER = "direct"  # "direct" or "diffusion".

# Diffusion sampling options, only used when SAMPLER = "diffusion".
DDIM_STEPS = 50
CFG_SCALE = 1.0
PHYSICS_GAMMA = 0.002
PHYSICS_LAMBDA_MAX = 0.01
USE_PHYSICS_GUIDANCE = True


# =========================
# Smoke test overrides
# =========================

SMOKE_RUN_NAME = "smoke"
SMOKE_EPOCHS = 1
SMOKE_PATCH_SIZE = 32
SMOKE_TRAIN_STRIDE = 32
SMOKE_PATCH_RATIO = None
SMOKE_MAX_PATCHES_PER_SECTION = 2
SMOKE_BATCH_SIZE = 2
SMOKE_DIFFUSION_STEPS = 20
SMOKE_TOKEN_PATCH_SIZE = 4
SMOKE_HIDDEN_SIZE = 32
SMOKE_TRANSFORMER_DEPTH = 1
SMOKE_NUM_HEADS = 2
SMOKE_DIRECT_BASE_CHANNELS = 16
SMOKE_DIRECT_DEPTH = 2
SMOKE_DDIM_STEPS = 4


def resolve_device(device: str = DEVICE) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def checkpoint_path(run_name: str | None = None, checkpoint_path: str | None = CHECKPOINT_PATH) -> str:
    if checkpoint_path:
        return checkpoint_path
    return f"runs/{run_name or RUN_NAME}/checkpoint.pt"


def use_multi_gpu(device: str) -> bool:
    return USE_MULTI_GPU and device.startswith("cuda")

