# PCDI Diffusion Inversion

This project is a compact implementation of the paper algorithm in Adjoint-Guided Physics-Constrained Diffusion Sampling for Post-Stack Convolutional Impedance Inversion
It trains a conditional diffusion prior from noiseless Marmousi2 and Overthrust triplets:
seismic data, low-frequency model, and true model.

The required data has been copied into this project:

- Marmousi2: `data/Marmousi2/Seismic_1ms_Sample1701_CMP1001.bin`,
  `data/Marmousi2/Imp_1ms_S_Sample1701_CMP1001.bin`,
  `data/Marmousi2/Imp_1ms_Sample1701_CMP1001.bin`
- Overthrust: `data/Overthrust/s_syn_il100.npy`,
  `data/Overthrust/model_low_il100.npy`,
  `data/Overthrust/model_il100.npy`

Marmousi impedance binaries are divided by 1000 so their scale is compatible with
the Overthrust velocity/impedance range.

## Quick smoke test

Double click `run_click.bat` for formal train + prediction. In an IDE, you can
also open `run_click.py` and click Run directly.

All run settings are at the top of `pcdi/config.py`. To run a fast link test,
set:

```python
RUN_MODE = "smoke"
```

Then run:

```powershell
cd pcdi_diffusion_inversion
python run_click.py
```

The output is written to `runs/smoke/predictions`.

## Normal training

```powershell
cd pcdi_diffusion_inversion
python run_click.py
```

Before formal training, set the key variables in `pcdi/config.py`:

```python
RUN_MODE = "train_predict"
RUN_NAME = "pcdi_full"
TRAIN_PATCH_RATIO = 0.70
BATCH_SIZE = 48
PREDICT_DATASETS = ["marmousi", "overthrust"]
```

Run prediction only after a checkpoint already exists:

```python
RUN_MODE = "predict"
RUN_NAME = "pcdi_full"
```

Use only one GPU when needed by setting:

```python
USE_MULTI_GPU = False
```

Use the diffusion sampler explicitly only when you want generative refinement:

```python
SAMPLER = "diffusion"
PHYSICS_GAMMA = 0.002
PHYSICS_LAMBDA_MAX = 0.01
```

## Algorithm mapping

- Training learns a conditional diffusion prior `p(m | m_c)` by noise prediction.
- The condition is the low-frequency model.
- Sampling uses deterministic DDIM.
- At each reverse step, the clean estimate is forward modeled with a 1D
  post-stack convolutional operator.
- The seismic residual is mapped back to the model by the analytic adjoint of
  the reflectivity-plus-convolution operator.
- The normalized, clipped adjoint direction is subtracted from the DDIM mean.

The default data-volume control is `TRAIN_PATCH_RATIO`; increase it toward
`1.0` to use all patches from each input section.
