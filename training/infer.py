import os
import glob
from pathlib import Path
from typing import List

import torch
import numpy as np
from PIL import Image
import tyro

from colorizer.models.unet_colorizer import UNetColorizer
from colorizer.utils.color import rgb_to_lab_tensor, to_network_inputs, from_network_output


class InferConfig(tyro.conf.FlagConversionOff):
    input: str
    checkpoint: str
    output: str
    image_size: int = 256


def load_model(ckpt_path: str, device: torch.device) -> UNetColorizer:
    model = UNetColorizer(in_ch=1, out_ch=2, base=64)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])  # type: ignore[index]
    model.to(device).eval()
    return model


def list_images(path: str) -> List[str]:
    if os.path.isdir(path):
        files = []
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
            files.extend(glob.glob(os.path.join(path, f"**/*{ext}"), recursive=True))
        return sorted(files)
    else:
        return [path]


def colorize_image(model: UNetColorizer, device: torch.device, img_path: str, out_path: str, image_size: int):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    img_rgb = np.array(img)
    lab = rgb_to_lab_tensor(img_rgb)
    L_t, _ = to_network_inputs(lab)
    with torch.no_grad():
        pred_ab = model(L_t.unsqueeze(0).to(device))
    rgb = from_network_output(L_t, pred_ab.squeeze(0).cpu())
    Image.fromarray(rgb).save(out_path)


def main(cfg: InferConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.checkpoint, device)

    inputs = list_images(cfg.input)
    out_dir = Path(cfg.output)
    if len(inputs) > 1 or os.path.isdir(cfg.input):
        out_dir.mkdir(parents=True, exist_ok=True)
    for p in inputs:
        if out_dir.is_dir():
            out_path = out_dir / (Path(p).stem + "_colored.jpg")
        else:
            out_path = Path(cfg.output)
        colorize_image(model, device, p, str(out_path), cfg.image_size)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    cfg = tyro.cli(InferConfig)
    main(cfg)
