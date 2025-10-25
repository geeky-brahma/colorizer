import os
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import tyro

from colorizer.config import TrainConfig
from colorizer.data.dataset import ImageFolderLabDataset
from colorizer.models.unet_colorizer import UNetColorizer


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def train(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Optimize convolution algorithms for fixed input sizes
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    print(f"Using device: {device}")
    set_seed(cfg.seed)

    ds = ImageFolderLabDataset(cfg.data_dir, image_size=cfg.image_size, augment=True)
    # Support effective batch size via gradient accumulation: per-step batch is reduced
    accum_steps = max(1, int(cfg.accum_steps))
    if cfg.batch_size < accum_steps:
        raise ValueError(
            f"batch_size ({cfg.batch_size}) must be >= accum_steps ({accum_steps})"
        )
    physical_bs = cfg.batch_size // accum_steps
    dl = DataLoader(
        ds,
        batch_size=physical_bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )

    model = UNetColorizer(in_ch=1, out_ch=2, base=cfg.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    # Use new torch.amp API and only enable AMP on CUDA devices
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    loss_fn = nn.L1Loss()

    # Optimizer steps count (after accumulation)
    total_steps = cfg.epochs * math.ceil(len(ds) / (physical_bs * accum_steps))
    best_loss = float("inf")

    out_dir = Path(cfg.output_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{cfg.epochs}")
        running = 0.0
        opt.zero_grad(set_to_none=True)
        for i, (L_t, ab_t) in enumerate(pbar, start=1):
            L_t = L_t.to(device, non_blocking=True)
            ab_t = ab_t.to(device, non_blocking=True)
            # autocast with device_type for future-proofing
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred_ab = model(L_t)
                loss = loss_fn(pred_ab, ab_t) / accum_steps

            scaler.scale(loss).backward()
            # Step optimizer every accum_steps or at the end of the epoch
            if (i % accum_steps) == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            running += loss.item() * accum_steps  # track true loss value
            if (i % accum_steps) == 0:
                global_step += 1
                pbar.set_postfix({"loss": f"{(loss.item()*accum_steps):.4f}"})

        # If the last batch didn't trigger a step due to remainder, finish it
        if (i % accum_steps) != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            global_step += 1

        epoch_loss = running / len(dl)
        # Save checkpoints
        save_checkpoint(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "cfg": cfg.__dict__,
                "epoch_loss": epoch_loss,
            },
            str(out_dir / "checkpoints" / "last.pt"),
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch_loss": epoch_loss,
                },
                str(out_dir / "checkpoints" / "best.pt"),
            )

    print("Training complete. Best epoch loss:", best_loss)


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    train(cfg)
