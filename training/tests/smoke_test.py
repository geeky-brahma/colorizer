import torch
from colorizer.models.unet_colorizer import UNetColorizer


def test_forward_pass():
    model = UNetColorizer()
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 2, 64, 64)
    assert y.min() >= -1.5 and y.max() <= 1.5  # tanh bounds


if __name__ == "__main__":
    test_forward_pass()
    print("Smoke test passed.")
