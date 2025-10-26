Precursor: Regression demo

This folder contains a small, self-contained regression demo extracted from the interactive notebook prototype. It trains a tiny convolutional regressor on CIFAR-10 where the target is the mean pixel intensity of the image (a simple toy regression task you ran earlier in the notebook).

Files:
- `run_regression.py` — runnable script that downloads CIFAR-10, trains a tiny conv regressor, plots loss curves, and saves the best checkpoint to `checkpoints/best_regression.pt`.
- `requirements.txt` — minimal list of Python packages needed to run the script.

Quick run (Windows cmd.exe):

```bat
pip install -r precursor_regression\requirements.txt
python precursor_regression\run_regression.py --epochs 3
```

Notes:
- This is intended as a light, standalone demonstration showing you did regression prototyping before moving on to UNet colorization.
- The script uses CIFAR-10 (32x32) for fast runs. For full experiments use the production training pipeline in `training/train.py`.
