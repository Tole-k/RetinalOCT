"""Benchmark the running Triton server and summarize both scheduling modes."""

import argparse
import asyncio
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from statistics import mean, stdev

import aiohttp
import numpy as np

PROJECT = Path(__file__).resolve().parent
WARMUP_SECONDS = 15
WINDOW_SECONDS = 60
CONCURRENCIES = ((1, 8, 32), (8, 32, 1), (32, 1, 8))
METRICS = ("samples_per_second", "p50_ms", "p95_ms", "p99_ms")


def inference_request(payload: bytes) -> tuple[bytes, dict[str, str]]:
    metadata = json.dumps({
        "inputs": [{"name": "images", "shape": [1, 3, 256, 256], "datatype": "FP32",
                    "parameters": {"binary_data_size": len(payload)}}],
        "outputs": [{"name": "logits", "parameters": {"binary_data": True}}],
    }).encode()
    return metadata + payload, {
        "Content-Type": "application/octet-stream",
        "Inference-Header-Content-Length": str(len(metadata)),
    }


async def infer(session, endpoint, request) -> tuple[bytes, int]:
    body, headers = request
    async with session.post(endpoint, data=body, headers=headers) as response:
        if response.status != 200:
            raise RuntimeError(f"Triton returned HTTP {response.status}: {await response.text()}")
        return await response.read(), int(response.headers["Inference-Header-Content-Length"])


async def window(session, endpoint, request, clients, seconds) -> list[float]:
    clock = asyncio.get_running_loop().time
    deadline = clock() + seconds
    latencies = []

    async def client() -> None:
        while (start := clock()) < deadline:
            await infer(session, endpoint, request)
            end = clock()
            if end < deadline:
                latencies.append((end - start) * 1000)

    async with asyncio.TaskGroup() as group:
        for _ in range(clients):
            group.create_task(client())
    if not latencies:
        raise RuntimeError("No successful requests completed during the window")
    return latencies


async def executions(session, url) -> dict[int, int]:
    async with session.get(f"{url}/stats") as response:
        response.raise_for_status()
        stats = await response.json()
    return {int(row["batch_size"]): int(row["compute_infer"]["count"])
            for row in stats["model_stats"][0]["batch_stats"]}


def write_csv(path: Path, rows: list[dict]) -> None:
    staging = path.with_suffix(".tmp")
    with staging.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    staging.replace(path)


def summarize(output: Path, runs: dict) -> None:
    trials = [row for mode in sorted(runs) for row in runs[mode]["trials"]]
    summary = []
    for mode, run in sorted(runs.items()):
        for clients in sorted(CONCURRENCIES[0]):
            group = [row for row in run["trials"] if row["clients"] == clients]
            record = {"mode": mode, "clients": clients, "trials": len(group)}
            for metric in METRICS:
                values = [row[metric] for row in group]
                record[f"{metric}_mean"] = mean(values)
                record[f"{metric}_sd"] = stdev(values)
            record["mean_server_batch_size"] = sum(row["server_samples"] for row in group) / sum(
                row["server_executions"] for row in group
            )
            summary.append(record)
    write_csv(output / "trials.csv", trials)
    write_csv(output / "summary.csv", summary)


async def benchmark(repository: Path, url: str, run_name: str) -> None:
    output = repository / ".benchmark" / run_name
    payload = np.random.default_rng(42).random((1, 3, 256, 256), dtype=np.float32).astype("<f4").tobytes()
    request = inference_request(payload)
    endpoint = f"{url}/v2/models/retinaloct"
    with (repository / "retinaloct/1/model.plan").open("rb") as engine:
        engine_hash = hashlib.file_digest(engine, "sha256").hexdigest()

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), trust_env=False) as session:
        async with session.get(f"{endpoint}/config") as response:
            response.raise_for_status()
            config = await response.json()
        scheduler = config.pop("dynamic_batching", None)
        mode = "batched" if scheduler is not None else "unbatched"
        if config.get("platform") != "tensorrt_plan" or config.get("max_batch_size", 0) < 2:
            raise RuntimeError("Expected the shared batch-capable TensorRT model")
        body, offset = await infer(session, f"{endpoint}/infer", request)
        logits = np.frombuffer(body[offset:], dtype="<f4")
        if logits.shape != (4,) or not np.isfinite(logits).all():
            raise RuntimeError("Expected four finite FP32 logits")

        cpu = next((line.partition(":")[2].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith("model name")), platform.machine())
        conditions = {
            "engine_sha256": engine_hash, "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "cpu": cpu, "client_host": platform.node(), "server_url": url,
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                            "--format=csv,noheader"], text=True).strip(),
            "warmup_seconds": WARMUP_SECONDS, "measurement_seconds": WINDOW_SECONDS,
            "model_config": config,
        }
        result_path = output / "results.json"
        results = json.loads(result_path.read_text()) if result_path.is_file() else {"conditions": conditions, "runs": {}}
        if results["conditions"] != conditions:
            raise RuntimeError("Benchmark conditions differ from the saved run. Choose a new --run-name.")

        trials = []
        for repetition, order in enumerate(CONCURRENCIES, start=1):
            for clients in order:
                print(f"{mode}, trial {repetition}, {clients} clients", flush=True)
                await window(session, f"{endpoint}/infer", request, clients, WARMUP_SECONDS)
                before = await executions(session, endpoint)
                latencies = await window(session, f"{endpoint}/infer", request, clients, WINDOW_SECONDS)
                after = await executions(session, endpoint)
                batches = {size: total - before.get(size, 0) for size, total in after.items()}
                count = sum(size * total for size, total in batches.items())
                if not len(latencies) <= count <= len(latencies) + clients:
                    raise RuntimeError("Server counts differ from client counts. Benchmark a dedicated server.")
                if mode == "unbatched" and any(total for size, total in batches.items() if size > 1):
                    raise RuntimeError("Triton grouped requests in unbatched mode")
                p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
                trials.append({"mode": mode, "repetition": repetition, "clients": clients,
                               "seconds": WINDOW_SECONDS, "samples": len(latencies),
                               "samples_per_second": len(latencies) / WINDOW_SECONDS,
                               "p50_ms": p50, "p95_ms": p95, "p99_ms": p99,
                               "server_samples": count, "server_executions": sum(batches.values())})

    results["runs"][mode] = {"scheduler": scheduler, "trials": trials}
    output.mkdir(parents=True, exist_ok=True)
    staging = result_path.with_suffix(".tmp")
    staging.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    staging.replace(result_path)
    summarize(output, results["runs"])
    print(f"Saved {mode} trials and summary to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=PROJECT / "triton_model_repository")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--run-name", default="comparison")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in {"", ".", ".."}:
        parser.error("--run-name must be a single directory name")
    asyncio.run(benchmark(args.repository, args.url.rstrip("/"), args.run_name))
