# RetinalOCT

## Weights & Biases

To stream training, validation, and test metrics to W&B, authenticate once and run training in online mode.

```bash
uv run wandb login
WANDB_MODE=online uv run lightning_model.py
```

The default W&B project is `retinaloct`. Set `WANDB_PROJECT` to use a different project. Set `WANDB_MODE=offline` to record a run locally without logging in. TensorBoard logs and model checkpoints are saved under `lightning_logs/` in either mode.
