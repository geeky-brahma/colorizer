# Colorizer — Image Colorization with a UNet-based Model

This repository contains code, models and utilities for training and running an image colorization model based on a UNet-style convolutional network. The goal is to take a grayscale image (L channel) and predict the color information (a/b channels in LAB space, or directly RGB), producing plausible and visually pleasing colorizations.

This README explains the theoretical background, the practical design and architecture choices, how training and inference are implemented in this repo, and guidance for evaluation and deployment. It is written so a technical reader can understand what was done and why, and reproduce or extend the work.

---

## Table of contents
- Project overview
- Model architecture (what we used and why)
- Data and preprocessing
- Training setup and loss
- Checkpoints, logs and artifacts
- Inference and evaluation
- Practical tips for deployment, pruning and quantization
- Files of interest
- Reproducibility notes & next steps

---

## Project overview

Colorization is an ill-posed inverse problem: many color images map to the same grayscale image. A system that colorizes must therefore learn plausible color distributions conditioned on luminance structure and semantics.

This project uses a UNet-style encoder-decoder convolutional model that predicts color channels for a given grayscale input. The design trades off simplicity and strong local receptive fields (via stacked convs and down/up-sampling) with skip connections that preserve spatial details. The repository contains training code, a saved best checkpoint (e.g. `training/runs/best.pt`), inference scripts, and a lightweight reporting utility.

Key high-level properties:
- Input: single-channel grayscale image resized to the training resolution (default 256×256).
- Output: typically 2-channel ab prediction (LAB color space) or 3-channel RGB depending on model/head.
- Loss: L1 loss on the predicted color channels (ab) — stable and perceptually reasonable for colorization.

---

## Why we started with a regression toy (the precursor)

When prototyping this project we intentionally started with a very small, well-scoped regression toy: a tiny convolutional network trained on CIFAR-10 to predict the mean pixel intensity of an image. There are a few practical reasons and useful intuitions behind that choice:

- Fast iteration and debugging: a regression toy runs quickly on CPU or a small GPU, so it's easy to iterate on data loading, training loops, AMP handling, checkpointing, and visualization without waiting hours for full colorization experiments.
- Isolate infrastructure bugs: before introducing the complexity of full image-to-image mapping and color-space conversions, the regression task lets you verify the correctness of batching, device transfers, loss accumulation, checkpoint saves/loads, and plotting code.
- Build confidence in training dynamics: simple scalar or low-dimensional targets make it easier to see whether the optimizer, learning-rate choice, and mixed-precision behavior are sane. If the regression training does not converge, it points to infra/bugs rather than model capacity or dataset issues.
- Communicate provenance: keeping a small demo (we put it under `precursor_regression/`) documents that the team validated core infrastructure and training loops before scaling to full-resolution colorization.

These practical wins explain why a regression prototype is a useful first step — it is not intended to be a useful colorizer itself, but a reliable sanity-check and developer artifact.

---

## Intuition for moving from regression to UNet colorization

Once the core training infra and dataset pipeline are confirmed with small experiments, we move to per-pixel image→image models (UNet-style) for the colorization task. The intuition and the staged progression are:

- From scalar to dense prediction: predicting a single summary value (mean intensity) is a much easier statistical problem than predicting a full image of color channels. After the scalar experiment proves the training loop, the next step is to increase output dimensionality gradually (e.g., small RGB autoencoders) and finally full-resolution UNet models.
- Need for spatial context and multi-scale features: colorization requires both global semantic cues (is this sky, skin, or foliage?) and local edge-aware details. The UNet architecture provides large receptive fields (through downsampling and bottleneck) while skip connections preserve fine spatial details, making it a natural choice for dense per-pixel prediction.
- Stability and diagnostics: having validated losses and training utilities on the toy task helps debug the more expensive UNet runs. You can reuse the same logging, checkpointing, and AMP code with minimal changes.
- Color-space and head design: early prototypes in RGB helped validate IO paths; the production pipeline switches to LAB with a 2-channel ab head (and Tanh) because predicting ab in a normalized range often makes learning easier and separates luminance (L) from chrominance (ab). The regression→small-autoencoder→UNet progression makes the transition to LAB head and full-resolution inputs less error-prone.

In short: start small to get the plumbing right, then scale model complexity once the training loop and data handling are robust. The notebook and the `precursor_regression/` script capture that deliberate progression so the project history is explicit.

---

## Model architecture — UNetColorizer (what & why)

Files: `training/models/unet_colorizer.py` (UNet implementation used in training)

High-level description
- The model follows a classical UNet pattern: an encoder path that progressively downsamples and extracts features, a bottleneck that operates at the smallest spatial resolution, and a symmetric decoder path that upsamples and re-combines features using skip connections from corresponding encoder stages.

Why UNet?
- Skip connections preserve spatial detail: colorization needs both semantic context (what object is present) and fine-grained texture edges so skip connections help reconstruct crisp output.
- Encoder-decoder provides large receptive fields (encoder + bottleneck) to capture global color cues (sky vs. skin vs. vegetation) while decoders reintroduce spatial resolution.
- UNet is simple, well-understood, efficient to implement and performs well on per-pixel prediction tasks (segmentation, colorization, etc.).

Implementation details in this repo
- conv_block(in_ch, out_ch): two stacked 3×3 convolutions with BatchNorm + ReLU — helps stable training and richer representations.
- Downsampling via MaxPool2d; upsampling via ConvTranspose2d (learned upsampling), followed by concatenation with encoder features and conv_block for refinement.
- Bottleneck uses the same conv_block pattern with a larger number of channels.
- Head: 1×1 conv projecting to either `out_ch=2` (ab) followed by Tanh (if normalized to [-1,1]) or a `out_ch=3` RGB head with appropriate activation.

Parameterization choices and intuition
- Base channel width (e.g., base=64) balances capacity vs compute. Wider base channels increase representational power, especially in bottleneck layers.
- BatchNorm and ReLU accelerate and stabilize training. Tanh output on ab channels implies training labels were scaled appropriately (commonly ab/128 to [-1,1]).

Notes about checkpoint naming mismatch
- Some checkpoints saved by training scripts use different naming (e.g., wrapper `.net` segments). The inference utilities in this repo include non-strict loading and small name-mapping fallbacks to accommodate these differences. If you need an exact strict load, a small remapping script can be applied to rewrite keys in the checkpoint.

---

## Data and preprocessing

Typical choices used in training (see `training/train.py`):
- Dataset: an image-folder dataset that loads color images and converts them to LAB (L channel + ab channels) for training. The repo's training config references a `data_dir` in the checkpoint's `cfg` (the dataset used at training time).
- Input resizing: images are resized to a fixed training size (default 256×256). This simplifies batching and keeps compute bounded.
- Normalization:
  - L channel: often scaled to [0,1] or [0,100] depending on how `from_network_output`/postprocessing is implemented.
  - ab channels: typically normalized to [-1, 1] by dividing by 128 (because ab ranges ~[-128,127]). The network head used Tanh which suggests that ab was scaled to [-1,1] during training.

Data augmentation
- During training random augmentations (flip, crop, color jitter on the color images before conversion to LAB) can improve generalization. The training pipeline in `train.py` supports augmentation toggles in the `TrainConfig`.

---

## Training setup and loss

Training entrypoint: `colorizer_v1/train.py` (uses a `TrainConfig` for hyperparameters and `tyro` for CLI parsing).

Summary of important hyperparameters (from a checkpoint `cfg` present in `training/runs/best.pt`):
- base_channels: 64 (base channel width for encoder/decoder)
- batch_size: 64 (logical batch; training supports gradient accumulation so physical batch may differ)
- image_size: 256
- lr: 2e-4 (AdamW)
- epochs: configurable (example runs used 10–15)
- amp: optional mixed-precision (enabled only on CUDA devices)
- accum_steps: gradient accumulation steps supported for effective larger batch sizes

Loss and optimization
- Loss: L1Loss between predicted ab and target ab (averaged over pixels). L1 is robust and tends to produce stable color predictions.
- Optimizer: AdamW with weight decay (default in `train.py`).
- Checkpointing: the training routine saves `last.pt` and `best.pt` (best by epoch_loss). Checkpoints include `model`, `opt`, `scaler`, `cfg`, `epoch`, and `epoch_loss`.

Training dynamics and reproducibility
- The project saves RNG seed in `cfg` and uses `set_seed` to ensure reproducibility.
- cUDNN benchmarking is enabled when possible for better throughput with fixed input sizes.

---

## Checkpoints, reports and what we produced

- Checkpoints: `training/runs/best.pt` (trained checkpoint used for inference). Some alternate checkpoint files may exist (e.g., additional `best` variants saved during runs).
- Inference utilities: see `training/infer.py` and the repository root `run.py` for example inference and debugging helpers that load a checkpoint and run single-image inference. (The repository may also include other small helper scripts under `training/`.)
- Reporting: reporting and analysis scripts live under `scripts/` and produce the `reports/` folder (example: `scripts/weight_stats.py` produces weight/parameter statistics used by the reports).

The repository also includes small utilities to:
- Inspect parameter counts and histograms (`make_reports.py`)
- Produce side-by-side visualizations and save PNG outputs for documentation

---

## Inference / How to run

Prerequisites
- Create and activate a Python virtual environment and install the requirements (PyTorch, torchvision, pillow, matplotlib, opencv-python). For CPU-only PyTorch use the official CPU wheels or adjust CUDA indexes.

Basic inference (example)
1. Put a grayscale test image in the repo root (e.g., `test_image1.png`).
2. Use the provided `debug_infer.py` or `test_colorizer.py` scripts which load `colorizer_v1/best.pt`, preprocess the image, forward it through the model and save a visualization (`colorized_debug_output.png` or similar).

Important notes for inference
- The model expects the input L to be normalized the same way as training (the supplied scripts normalize and resize to 256×256). If your training used a different normalization, match it during inference.
- For output in LAB space: predicted ab values are often in [-1,1] and must be rescaled by 128 to get actual ab units and combined with L to convert to RGB (use OpenCV `cv2.cvtColor` with `COLOR_Lab2RGB`).

---

## Evaluation metrics

Standard metrics for colorization and image reconstruction used here or suggested:
- L1 (per-channel or per-pixel) on ab channels — training loss used.
- PSNR and SSIM on reconstructed RGB images compared to ground-truth color images — measured on a holdout validation set.

The project contains code templates and small utilities to compute per-sample metrics and histogram them in `scripts/` (for example, `scripts/weight_stats.py`) and related evaluation functions.

---

## Practical recommendations (deployment, compression)

Given the parameter distribution and the layer concentration (top ~5 layers contain the majority of params), practical approaches to shrink or speed up the model:

1. Per-channel quantization of the largest convolutional layers (8-bit) — usually a high ROI starting point.
2. Structured channel pruning on the bottleneck and large decoder convs; prune and fine-tune to regain accuracy.
3. Consider fusing conv+bn+relu at inference time and using vendor-optimized kernels for large convs.
4. If deploying to CPU-only or mobile, prefer `opencv-python-headless` and convert model to ONNX then to a runtime (e.g., TensorRT, OpenVINO) that supports quantization.

---

## Files of interest (quick map)

`training/models/unet_colorizer.py` — model implementation (UNetColorizer).
`training/train.py` — training loop and configuration.
`training/runs/best.pt` — example saved checkpoint used for inference (path where training run outputs are stored).
`training/infer.py`, `run.py` — inference/debug scripts and example runners that load a checkpoint and save outputs.
`training/tests/smoke_test.py` — a small smoke test for quick checks.
`scripts/weight_stats.py` — reporting/weight-inspection utilities used to generate artifacts in `reports/`.
`reports/` — generated assets (plots, CSVs, sample outputs, and the inference markdowns).

---

## Reproducibility tips

- Use the same version of PyTorch and torchvision when reproducing training. Differences in kernels and RNG can change results.
- If using GPU training, ensure CUDA/cuDNN versions are compatible with the installed PyTorch wheel.
- The checkpoint includes `cfg` (training configuration) and the optimizer/scaler states; to resume training load everything from `.pt`.

---

## Limitations & future work

- The UNet architecture provides strong local reconstructions but can struggle with global color ambiguity. Future work could combine UNet with a semantic prior (e.g., pretrained segmentation or classification features) or incorporate a generative/distributional head (GAN or probabilistic model) to sample diverse colors.
- Evaluation: perceptual user studies or learned perceptual metrics (LPIPS) can be added for better quality assessment.

---

If you want, I can: add a one-page summary for a README badge, generate a short slide with the figures from `reports/`, or produce a strict mapping script that rewrites checkpoint keys so `strict=True` loads succeed. Tell me which you'd like next.
