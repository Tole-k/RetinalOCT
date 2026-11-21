"""Measure two Triton scheduling configurations against a shared TensorRT engine."""

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
import numpy as np

URL = "http://127.0.0.1:18080"
MODELS = {"off": "retinaloct_unbatched", "on": "retinaloct_dynamic"}
WARMUP_SECONDS = 15
MEASUREMENT_SECONDS = 60


def make_request(payload: bytes, batch_size: int = 1) -> tuple[bytes, dict[str, str]]:
    metadata = json.dumps({
        "inputs": [{
            "name": "images", "shape": [batch_size, 3, 256, 256], "datatype": "FP32",
            "parameters": {"binary_data_size": batch_size * len(payload)},
        }],
        "outputs": [{"name": "logits", "parameters": {"binary_data": True}}],
    }).encode()
    return metadata + payload * batch_size, {
        "Content-Type": "application/octet-stream",
        "Inference-Header-Content-Length": str(len(metadata)),
    }


async def control(session, route, method="GET"):
    options = {"json": {}} if method == "POST" else {}
    async with session.request(method, URL + route, **options) as response:
        if response.status != 200:
            raise RuntimeError(f"{method} {route}: HTTP {response.status}: {await response.text()}")
        return await response.json() if route.endswith(("config", "stats")) else None


async def check_gpu(server_pid: str) -> None:
    process = await asyncio.create_subprocess_exec(
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError(stderr.decode().strip())
    other_processes = set(stdout.decode().split()) - {server_pid}
    if other_processes:
        raise RuntimeError(f"Competing GPU processes: {', '.join(sorted(other_processes))}")


async def confirm_logits(session, model, payload, batch_size, reference):
    body, headers = make_request(payload, batch_size)
    async with session.post(f"{URL}/v2/models/{model}/infer", data=body, headers=headers) as response:
        response.raise_for_status()
        result = await response.read()
        offset = int(response.headers["Inference-Header-Content-Length"])
    metadata = json.loads(result[:offset])["outputs"][0]
    if metadata["shape"] != [batch_size, 4] or metadata["datatype"] != "FP32":
        raise RuntimeError(f"Unexpected output metadata: {metadata}")
    logits = np.frombuffer(result[offset:], dtype="<f4").reshape(batch_size, 4)
    np.testing.assert_allclose(logits, np.repeat(reference, batch_size, axis=0), rtol=0.02, atol=0.01)
    return logits.copy()


def execution_counts(stats: dict) -> dict[int, int]:
    result = stats["model_stats"][0]
    return {
        int(batch["batch_size"]): int(batch["compute_infer"]["count"])
        for batch in result["batch_stats"]
    }


async def measure(session, model, request, concurrency, seconds, server_pid):
    loop = asyncio.get_running_loop()
    body, headers = request
    endpoint = f"{URL}/v2/models/{model}/infer"
    samples = []
    start = loop.time()
    deadline = start + seconds

    async def client() -> None:
        while (sent := loop.time()) < deadline:
            async with session.post(endpoint, data=body, headers=headers) as response:
                if response.status != 200:
                    raise RuntimeError(f"Triton returned HTTP {response.status}: {await response.text()}")
                await response.read()
            received = loop.time()
            if received < deadline:
                samples.append((received - start, (received - sent) * 1000))

    async def guard_gpu() -> None:
        while (remaining := deadline - loop.time()) > 0:
            await check_gpu(server_pid)
            await asyncio.sleep(min(5, remaining))

    async with asyncio.TaskGroup() as group:
        group.create_task(guard_gpu())
        for _ in range(concurrency):
            group.create_task(client())
    if not samples:
        raise RuntimeError("The measurement window contained no successful requests")
    return np.asarray(samples, dtype=np.float64)


async def run(output_directory: Path, server_pid: str, verify_only: bool) -> None:
    payload = (output_directory / "sample.fp32").read_bytes()
    inputs = json.loads((output_directory / "inputs.json").read_text())
    if hashlib.sha256(payload).hexdigest() != inputs["payload_sha256"]:
        raise RuntimeError("The input payload changed")
    reference = np.asarray(inputs["reference_logits"])
    request = make_request(payload)

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        if verify_only:
            verification = {}
            for mode, model in MODELS.items():
                await control(session, f"/v2/repository/models/{model}/load", "POST")
                try:
                    config = await control(session, f"/v2/models/{model}/config")
                    if config["max_batch_size"] != 8 or ("dynamic_batching" in config) != (mode == "on"):
                        raise RuntimeError(f"Unexpected scheduler config for {model}: {config}")
                    logits = await confirm_logits(session, model, payload, 1, reference)
                    await confirm_logits(session, model, payload, 8, reference)
                    verification[mode] = {"config": config, "logits": logits.tolist()}
                finally:
                    await control(session, f"/v2/repository/models/{model}/unload", "POST")
            np.testing.assert_array_equal(verification["on"]["logits"], verification["off"]["logits"])
            (output_directory / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
            print("Verified the scheduler settings and TensorRT outputs at batch sizes 1 and 8.", flush=True)
            return

        trials = output_directory / "trials"
        trials.mkdir(exist_ok=True)
        for repetition, concurrency_order in enumerate(((1, 8, 32), (8, 32, 1), (32, 1, 8)), start=1):
            for client_count in concurrency_order:
                modes = ("off", "on") if (repetition + client_count) % 2 == 0 else ("on", "off")
                for mode in modes:
                    model = MODELS[mode]
                    prefix = trials / f"trial-{repetition}-c{client_count}-{mode}"
                    if prefix.with_suffix(".json").is_file():
                        continue
                    print(f"{datetime.now(UTC).isoformat()} trial {repetition}, {client_count} clients, batching {mode}", flush=True)
                    await check_gpu(server_pid)
                    await control(session, f"/v2/repository/models/{model}/load", "POST")
                    stats_route = f"/v2/models/{model}/stats"
                    try:
                        await measure(session, model, request, client_count, WARMUP_SECONDS, server_pid)
                        before = execution_counts(await control(session, stats_route))
                        samples = await measure(session, model, request, client_count, MEASUREMENT_SECONDS, server_pid)
                        after = execution_counts(await control(session, stats_route))
                        await check_gpu(server_pid)
                    finally:
                        await control(session, f"/v2/repository/models/{model}/unload", "POST")

                    batch_counts = {size: count - before.get(size, 0) for size, count in after.items()}
                    if mode == "off" and any(count for size, count in batch_counts.items() if size != 1):
                        raise RuntimeError("Triton batched requests in the disabled configuration")
                    server_samples = sum(size * count for size, count in batch_counts.items())
                    if not len(samples) <= server_samples <= len(samples) + client_count:
                        raise RuntimeError(f"Server and client prediction counts disagree: {server_samples}, {len(samples)}")
                    p50, p95, p99 = np.percentile(samples[:, 1], [50, 95, 99])
                    result = {
                        "repetition": repetition, "clients": client_count, "dynamic_batching": mode,
                        "samples": len(samples), "seconds": MEASUREMENT_SECONDS,
                        "samples_per_second": len(samples) / MEASUREMENT_SECONDS,
                        "p50_ms": p50, "p95_ms": p95, "p99_ms": p99,
                        "batch_executions": batch_counts,
                    }
                    np.save(prefix.with_suffix(".npy"), samples)
                    prefix.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
                    print(f"{result['samples_per_second']:.2f} samples/s, p95 {p95:.2f} ms, batches {batch_counts}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("server_pid")
    parser.add_argument("--verify-only", action="store_true")
    options = parser.parse_args()
    asyncio.run(run(options.output_directory, options.server_pid, options.verify_only))
