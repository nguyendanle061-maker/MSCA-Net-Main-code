# MSCA-Net: Z24 training configuration and architecture

Focused code release for the MSCA-Net structural damage classification study.
Contains the adaptive sensor-mixing front-end, multi-scale convolutional blocks,
squeeze-and-excitation, residual connections, downsampling, attention pooling,
and the Z24 training/evaluation pipeline. Baselines, Chuong Duong simulations,
manuscripts, datasets, and pretrained checkpoints are not included.

## Setup

Python 3.10 or newer. Install a CUDA-enabled PyTorch build appropriate for your
machine if GPU training is required, then install the remaining dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

## Z24 data

Obtain the data through KU Leuven:
https://bwk.kuleuven.be/bwm/z24
Registration and the provider's data-use conditions apply. This repository does
not redistribute the Z24 data.

Prepare authorized processed arrays in `data/z24/`:

- `inputs.npy`: float32, shape `(1530, 27, 6000)` (record, sensor, time).
- `labels.npy`: integer shape `(1530,)`, labels 0 through 16; the reference arrays
  have 90 examples per class.

The original raw-file-to-array conversion and per-record source/session mapping
are not included in this focused release. Providing the same array layout alone
does not establish identical channel ordering, source records, or class mapping.
Exact numerical reproduction requires the original processed inputs and labels.

The configuration uses a stratified 70/15/15 split (seed 42), a random training
crop and centre evaluation crop of length 768, first differences, sample
normalization, and training jitter. Inspect `src/data/dataset.py` for the exact
operation order and split implementation. State/session confounding limits
interpretation as generalisation to independent measurement sessions.

## Train and evaluate

Run from the repository root:

```powershell
python -m src.train --config configs/z24.yaml
python -m src.evaluate --config configs/z24.yaml
```

Resume an interrupted run:

```powershell
python -m src.train --config configs/z24.yaml --resume
```

The default `device: auto` selects CUDA when available. Training runs for 100
epochs with early stopping disabled and retains the checkpoint with the highest
validation macro-F1. Checkpoints, histories and metrics are written to `outputs/`.
Missing input arrays raise an error; training never silently substitutes dummy data.

## Scope

The Z24 configuration uses 27 channels, 17 classes, width 128, four blocks,
kernels (9,19,39), and graph embedding dimension 10. The adjacency is learned
and input-independent; it is not verified physical sensor connectivity.
This repository supplies the architecture and Z24 configuration/pipeline, not
an independently reproducible archive of all experiments in the manuscript.
