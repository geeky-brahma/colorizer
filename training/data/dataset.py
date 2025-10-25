import os
import glob
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from colorizer.utils.color import rgb_to_lab_tensor, to_network_inputs


class ImageFolderLabDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 256,
        augment: bool = True,
        extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.augment = augment
        self.transform = transform
        self.paths = []
        allowed = set(e.lower() for e in extensions)
        for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            if os.path.isfile(p):
                ext = os.path.splitext(p)[1].lower()
                if ext in allowed:
                    self.paths.append(p)
        self.paths.sort()
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found under {root}")

    def __len__(self):
        return len(self.paths)

    def _load_image(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        return np.array(img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        img_rgb = self._load_image(path)
        if self.augment:
            # simple horizontal flip augmentation
            if np.random.rand() < 0.5:
                img_rgb = img_rgb[:, ::-1, :]

        lab = rgb_to_lab_tensor(img_rgb)
        L_t, ab_t = to_network_inputs(lab)
        if self.transform:
            L_t, ab_t = self.transform(L_t, ab_t)
        return L_t, ab_t
