# RetinalOCT

## Weights & Biases

To stream training, validation, and test metrics to W&B, authenticate once and run training in online mode.

```bash
uv run wandb login
WANDB_MODE=online uv run lightning_model.py
```

The default W&B project is `retinaloct`. Set `WANDB_PROJECT` to use a different project. Set `WANDB_MODE=offline` to record a run locally without logging in. TensorBoard logs and model checkpoints are saved under `lightning_logs/` in either mode.

## Triton inference

Run the Triton scripts from the project root. Export selects the best checkpoint from the newest Lightning run with one and writes `model.onnx`. Build uses Docker and the NVIDIA Container Toolkit to compile a single FP16 engine at `triton_model_repository/retinaloct/1/model.plan`.

```bash
uv run --locked --group serve python export_triton.py
bash build_tensorrt.sh
```

Start Triton in the foreground on `127.0.0.1:8000`. It selects a config from `triton_model_repository/retinaloct/configs/` and serves the same `retinaloct` engine in all three modes.

```bash
bash serve_triton.sh unbatched
```

Once the model is ready, send this held-out scan from the local Kaggle cache in a second terminal. The client uses the training transforms and prints the predicted class and four logits.

```bash
uv run --locked --group serve python predict_triton.py \
  "$HOME/.cache/kagglehub/datasets/paultimothymooney/kermany2018/versions/2/OCT2017 /test/NORMAL/NORMAL-1017237-1.jpeg"
```

Keep the path quoted. The space after `OCT2017` is part of the folder name.

Run the benchmark from the same terminal.

```bash
uv run --locked --group serve python benchmark_triton.py
```

Stop the server with Ctrl+C, restart it in `batched` mode, and run the same benchmark command again.

```bash
bash serve_triton.sh batched
```

For the third comparison, restart with `bash serve_triton.sh quick_batched` and run the benchmark again. Triton can combine requests already in the queue in this mode. The batching delay is zero.

The benchmark detects the active scheduler and runs three trials each at 1, 8, and 32 concurrent clients. Every trial sends identical single-image FP32 inputs, warms up for 15 seconds, and measures for 60 seconds. It records latency percentiles, samples per second, and the server's effective batch sizes. Keep other GPU workloads off during the comparison.

All three modes accumulate in `triton_model_repository/.benchmark/comparison/`, with `trials.csv`, `summary.csv`, and `results.json` containing the hardware and active configs. Use the same `--run-name` for all modes to start a new comparison. After exporting a new checkpoint, rebuild the engine and restart the server before benchmarking it.

### Profile the inference bottleneck

With Triton running locally, profile the active configuration from a second terminal.

```bash
uv run --locked --group serve python profile_triton.py
```

The script measures Triton's queue, input, model execution, and output times alongside end-to-end latency and CPU/GPU utilization. It times response decoding and measures a local HTTP echo with the same request and reply sizes as a transport control. The echo includes HTTP and Python handling. The time outside Triton's model statistics includes transport and HTTP handling on both sides. CPU cores use `1.0` for one fully occupied logical core over the measurement window. Keep other traffic off the server during profiling. The full report goes to `triton_model_repository/.benchmark/`.
