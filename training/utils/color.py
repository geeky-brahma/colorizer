import torch
import numpy as np
from skimage import color as skcolor


L_MEAN = 50.0
L_MAX = 100.0
AB_MAX = 128.0  # a,b in [-128, 127]


def rgb_to_lab_tensor(img_rgb: np.ndarray) -> np.ndarray:
    """
    img_rgb: HxWx3 in [0, 255] uint8 or float [0,1]
    returns LAB float64 in ranges L [0,100], a/b roughly [-128,127]
    """
    if img_rgb.dtype != np.float32 and img_rgb.dtype != np.float64:
        img_rgb = img_rgb.astype(np.float32) / 255.0
    img_lab = skcolor.rgb2lab(img_rgb)
    return img_lab


def lab_to_rgb_tensor(img_lab: np.ndarray) -> np.ndarray:
    rgb = skcolor.lab2rgb(img_lab)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


def to_network_inputs(img_lab: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert LAB numpy image to network inputs/targets.
    Returns (L_tensor, ab_tensor) with shapes [1,H,W] and [2,H,W], respectively, scaled to [-1,1].
    """
    L = img_lab[..., 0]
    ab = img_lab[..., 1:3]

    # scale L from [0,100] -> [-1,1]
    L_scaled = (L / L_MAX) * 2.0 - 1.0
    # scale ab from [-128,127] -> [-1,1]
    ab_scaled = ab / AB_MAX

    L_t = torch.from_numpy(L_scaled).float().unsqueeze(0)
    ab_t = torch.from_numpy(ab_scaled).float().permute(2, 0, 1)
    return L_t, ab_t


def from_network_output(L_t: torch.Tensor, ab_t: torch.Tensor) -> np.ndarray:
    """
    Convert network tensors back to uint8 RGB image.
    L_t: [1,H,W] in [-1,1]
    ab_t: [2,H,W] in [-1,1]
    Returns RGB uint8 HxWx3
    """
    L = ((L_t.squeeze(0).cpu().numpy() + 1.0) * 0.5) * L_MAX
    ab = (ab_t.permute(1, 2, 0).cpu().numpy()) * AB_MAX

    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float32)
    lab[..., 0] = L
    lab[..., 1:3] = ab
    rgb = lab_to_rgb_tensor(lab)
    return rgb
