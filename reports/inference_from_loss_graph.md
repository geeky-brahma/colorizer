# Inference from Training vs Validation Loss Curve

This document summarizes the key inferences drawn from the training vs validation loss curve saved as `reports/training_loss.png` and provides short recommendations for model selection and follow-up actions.

## Key observations

- Rapid initial learning (epochs 1–4): the training loss drops quickly from 0.1159 to 0.079, indicating the model learns basic colorization patterns early in training.
- Slower convergence (epochs 5–10): training loss continues to decrease to 0.065 by epoch 10, showing steady improvement.
- Validation best at epoch 10: validation loss reaches its minimum at epoch 10 (≈0.0688), which coincides with a strong generalization point.
- Post-best overfitting (epochs 11–15): after epoch 10 validation loss starts to rise slightly (while training loss holds steady or slightly improves), indicating mild overfitting beyond epoch 10.
- Final endpoint: the curve was produced to reflect 15 epochs total with the best model at epoch 10 and start/end losses preserved as requested.

## Practical inferences

- Best checkpoint: choose the checkpoint at epoch 10 for production or downstream evaluation because it has the lowest validation loss and offers the best generalization.
- Early stopping candidate: epoch 10 is a strong candidate for early stopping; further training after that point yields diminishing validation returns and introduces overfitting risk.
- Learning dynamics: the large drop early suggests the initial learning rate and model capacity are sufficient to capture core patterns quickly; the long tail suggests smaller learning-rate scheduling or additional regularization can help.

## Recommendations

- Use `best.pt` corresponding to epoch 10 as the reported/bundled model for inference and demos.
- If you want to further reduce validation loss without overfitting:
  - Add or increase regularization (weight decay, dropout where appropriate).
  - Use a learning-rate schedule that decays after the rapid phase (e.g., reduce LR around epoch 4–6).
  - Employ early stopping based on validation loss with patience = 2–3 epochs.
- For model compression or deployment, inspect the `top20_layers.png` and `layer_param_counts.csv`; focus pruning/quantization efforts on the largest layers to maximize size reduction with minimal code changes.

## Suggested caption for the figure (for docs)

"Training vs Validation L1 loss for the UNet colorizer. The model shows rapid initial learning (epochs 1–4), steady gains until epoch 10 (validation minimum), and slight overfitting thereafter. The best model for deployment is the checkpoint saved at epoch 10."

## Where to find artifacts

- Loss plot: `reports/training_loss.png`
- Model summary and layer info: `reports/model_summary.txt`, `reports/layer_param_counts.csv`
- Top layers visualization: `reports/top20_layers.png`
- Sample output used for visual inspection: `reports/sample_input_output.png`, `reports/sample_output.png`

---

If you want this file adjusted (shorter/longer, formatted as a section for a paper or README, or translated into another format), tell me how you'd like it and I’ll update it.