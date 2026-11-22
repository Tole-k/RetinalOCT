# RetinalOCT

## Weights & Biases

To stream training, validation, and test metrics to W&B, authenticate once and run training in online mode.

```bash
uv run wandb login
WANDB_MODE=online uv run lightning_model.py
```

The default W&B project is `retinaloct`. Set `WANDB_PROJECT` to use a different project. Set `WANDB_MODE=offline` to record a run locally without logging in. TensorBoard logs and model checkpoints are saved under `lightning_logs/` in either mode.

## Triton inference

Run the four Triton scripts from the project root. Export selects the best checkpoint from the newest Lightning run with one and writes `model.onnx`. Build uses Docker and the NVIDIA Container Toolkit to compile a single FP16 engine at `triton_model_repository/retinaloct/1/model.plan`.

```bash
uv run --locked --group serve python export_triton.py
bash build_tensorrt.sh
```

Start Triton in the foreground on `127.0.0.1:8000`. It selects a config from `triton_model_repository/retinaloct/configs/` and serves the same `retinaloct` engine in either mode.

```bash
bash serve_triton.sh unbatched
```

Run the benchmark from a second terminal once the model is ready.

```bash
uv run --locked --group serve python benchmark_triton.py
```

Stop the server with Ctrl+C, restart it in `batched` mode, and run the same benchmark command again.

```bash
bash serve_triton.sh batched
```

The benchmark detects the active scheduler and runs three trials each at 1, 8, and 32 concurrent clients. Every trial sends identical single-image FP32 inputs, warms up for 15 seconds, and measures for 60 seconds. It records latency percentiles, samples per second, and the server's effective batch sizes. Keep other GPU workloads off during the comparison.

Both modes accumulate in `triton_model_repository/.benchmark/comparison/`, with `trials.csv`, `summary.csv`, and `results.json` containing the hardware and active configs. Use the same `--run-name` for both modes to start a new comparison. After exporting a new checkpoint, rebuild the engine and restart the server before benchmarking it.
