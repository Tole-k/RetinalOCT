# RetinalOCT

## Weights & Biases

To stream training, validation, and test metrics to W&B, authenticate once and run training in online mode.

```bash
uv run wandb login
WANDB_MODE=online uv run lightning_model.py
```

The default W&B project is `retinaloct`. Set `WANDB_PROJECT` to use a different project. Set `WANDB_MODE=offline` to record a run locally without logging in. TensorBoard logs and model checkpoints are saved under `lightning_logs/` in either mode.

## Triton inference

After training finishes, export picks the best checkpoint from the newest numbered Lightning run with a saved best checkpoint. It prints the selected path, checks the ONNX model against PyTorch on CPU, and writes both Triton configurations. Use a new repository subdirectory for each checkpoint. The TensorRT build uses Docker with the NVIDIA Container Toolkit and should run on the GPU that will serve the model.

```bash
repository=triton_model_repository/run-13
test_directory='/home/tolek/.cache/kagglehub/datasets/paultimothymooney/kermany2018/versions/2/OCT2017 /test'
results=triton_benchmarks/run-13

uv run --locked --with onnxruntime \
  python -m scripts.export_triton --repository "$repository"
bash scripts/triton.sh build "$repository"
```

Start with the ONNX baseline. The server runs in the foreground and listens on `127.0.0.1:8000`.

```bash
bash scripts/triton.sh serve "$repository" retinaloct_onnx
```

In a second terminal, set the same `repository`, `test_directory`, and `results` values. The prediction check reads the checkpoint path recorded by the export, so it uses the same weights even if a newer training run exists. Run the checks after Triton reports the model ready. Both Python scripts accept `--checkpoint` for an explicit override.

```bash
model=retinaloct_onnx
uv run --locked  python -m scripts.check_triton \
  --repository "$repository" --test-directory "$test_directory" --model "$model"
bash scripts/triton.sh benchmark "$model" "$results"
```

Stop the ONNX server, start `retinaloct_trt` with the same `serve` command, then repeat the checks with `model=retinaloct_trt`. The prediction check uses the same RGB conversion, `256 × 256` resize, and tensor conversion as training. Both models accept one FP32 image and return four logits in `CNV`, `DME`, `DRUSEN`, `NORMAL` order.

Keep the GPU free of other workloads during the benchmark. Compare median p95 latency across the three CSV files for each model and record requests per second. Serve `retinaloct_trt` if it matches all held-out PyTorch predictions and its median p95 is at least 5% lower. Use the verified `retinaloct_onnx` baseline if the TensorRT build or checks fail or the latency improvement is smaller.

### TensorRT dynamic batching benchmark

Run these commands from the project root. The benchmark uses one FP16 TensorRT engine for both configurations and sends identical single-image FP32 inputs over HTTP on the same machine. Dynamic batching uses a maximum batch size of 8, preferred sizes of 4 and 8, and a 2 ms queue delay. The comparison runs three trials per configuration at 1, 8, and 32 concurrent clients. Each trial warms up for 15 seconds and measures the next 60 seconds.

```bash
checkpoint=$(cat triton_model_repository/run-13/source_checkpoint.txt)
benchmark_directory=triton_benchmarks/batching-run-13

uv run --locked --group serve python prepare_triton_batching.py "$checkpoint" "$benchmark_directory"
uv run --locked --with aiohttp bash run_triton_batching.sh "$benchmark_directory" --wait-for-gpu
```

Use an empty directory for each new comparison. The runner builds the engine, starts a temporary Triton server on port 18080, and writes `trials.csv` and `summary.csv` under the benchmark directory. `--wait-for-gpu` waits for other GPU compute jobs to finish before starting.
