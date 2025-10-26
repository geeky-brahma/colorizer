"""Small regression

Trains a tiny conv network to predict the mean pixel intensity of CIFAR-10 images.
"""
import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.datasets as datasets


SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CIFARRegDataset(Dataset):
    """CIFAR-10 images, target = mean pixel intensity in [0,1]"""

    def __init__(self, root="./data", train=True, transform=None, download=True):
        self.base = datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        if self.transform is not None:
            img_t = self.transform(img)
        else:
            img_t = T.ToTensor()(img)
        target = img_t.mean().float()
        return img_t, target.unsqueeze(0)


class TinyConvReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x.squeeze(1)


def train_epoch(model, loader, opt, criterion, scaler, device, use_amp):
    model.train()
    running = 0.0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).squeeze(1)
        opt.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            opt.step()
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)


def eval_epoch(model, loader, criterion, device, use_amp):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).squeeze(1)
            if use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(imgs)
                    loss = criterion(preds, targets)
            else:
                preds = model(imgs)
                loss = criterion(preds, targets)
            running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting results")
    parser.add_argument("--save-dir", type=str, default="precursor_checkpoints")
    args = parser.parse_args(argv)

    set_seed(SEED)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print("Device:", device)

    transform = T.Compose([T.ToTensor()])
    full_train = CIFARRegDataset(train=True, transform=transform, download=True)
    n_val = 5000
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    test_ds = CIFARRegDataset(train=False, transform=transform, download=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = TinyConvReg().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val = float("inf")
    train_losses, val_losses = [], []
    os.makedirs(args.save_dir, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        va = eval_epoch(model, val_loader, criterion, device, use_amp)
        train_losses.append(tr); val_losses.append(va)
        print(f"Epoch {ep}/{args.epochs}  Train MSE: {tr:.6f}  Val MSE: {va:.6f}")
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_regression.pt"))
            print("  Saved best model.")

    if not args.no_plot:
        try:
            plt.figure(figsize=(6,4))
            plt.plot(range(1, args.epochs+1), train_losses, label="train MSE")
            plt.plot(range(1, args.epochs+1), val_losses, label="val MSE")
            plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.grid(True)
            plt.show()
        except Exception as e:
            print("Plotting failed:", e)

    # simple visualization of a few test images and predicted scalar -> grayscale patch
    model.eval()
    imgs, targets = next(iter(test_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        preds = model(imgs).cpu().numpy()

    targets_np = targets.squeeze(1).cpu().numpy()
    batch_size = imgs.shape[0]
    n_show = min(6, batch_size)
    fig, axs = plt.subplots(n_show, 2, figsize=(6, 3*n_show))
    fig.suptitle("Left: Input image  —  Right: Predicted grayscale patch", fontsize=14)
    for i in range(n_show):
        img_np = imgs[i].cpu().permute(1,2,0).numpy()
        img_np = np.clip(img_np, 0, 1)
        axs[i,0].imshow(img_np)
        axs[i,0].axis('off')

        pred_val = float(preds[i])
        pred_clamped = max(0.0, min(1.0, pred_val))
        single = np.ones((img_np.shape[0], img_np.shape[1])) * pred_clamped
        axs[i,1].imshow(single, cmap='viridis', vmin=0, vmax=1)
        axs[i,1].set_title(f"Pred: {pred_clamped:.3f}\nTrue: {targets_np[i]:.3f}")
        axs[i,1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
