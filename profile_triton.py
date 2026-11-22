"""Measure where time goes during local Triton inference."""

import argparse
import asyncio
import csv
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter, process_time
from urllib.parse import urlsplit

import aiohttp
import numpy as np
from aiohttp import web

from benchmark_triton import PROJECT, infer, inference_request

STAGES = ("success", "queue", "compute_input", "compute_infer", "compute_output")


async def stats(session, endpoint):
    async with session.get(f"{endpoint}/stats") as response:
        response.raise_for_status()
        result = await response.json()
    return result["model_stats"][0]


def host_ticks() -> tuple[int, int]:
    user, nice, system, idle, iowait, irq, softirq, steal = map(
        int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]
    )
    busy = user + nice + system + irq + softirq + steal
    return busy, busy + idle + iowait


def process_cpu(pid: str | None) -> float | None:
    if pid is None:
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return None
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def decode_logits(body: bytes, offset: int) -> np.ndarray:
    output, = json.loads(body[:offset])["outputs"]
    if output["name"] != "logits" or output["datatype"] != "FP32" or output["shape"] != [1, 4]:
        raise RuntimeError(f"Unexpected inference output: {output}")
    return np.frombuffer(body[offset:], dtype="<f4").reshape(1, 4)


async def measure(session, endpoint, request, clients, seconds):
    clock = asyncio.get_running_loop().time
    deadline = clock() + seconds
    http_ms, decode_ms, completed_ms = [], [], []

    async def client() -> None:
        while (started := clock()) < deadline:
            body, offset = await infer(session, endpoint, request)
            received = clock()
            decode_logits(body, offset)
            decoded = clock()
            http_ms.append((received - started) * 1000)
            decode_ms.append((decoded - received) * 1000)
            if decoded < deadline:
                completed_ms.append((decoded - started) * 1000)

    async with asyncio.TaskGroup() as group:
        for _ in range(clients):
            group.create_task(client())
    if not completed_ms:
        raise RuntimeError("No inference completed during the measurement window")
    return {
        "samples_per_second": len(completed_ms) / seconds,
        "p95_ms": float(np.percentile(completed_ms, 95)),
        "http_ms": mean(http_ms),
        "decode_ms": mean(decode_ms),
        "requests_including_drain": len(http_ms),
    }


async def measure_with_resources(session, endpoint, request, clients, seconds, gpu, server_pid):
    monitor = await asyncio.create_subprocess_exec(
        "nvidia-smi", "-i", str(gpu), "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits", "-lms", "500",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    busy_before, total_before = host_ticks()
    server_before = process_cpu(server_pid)
    started, client_before = perf_counter(), process_time()
    try:
        result = await measure(session, endpoint, request, clients, seconds)
        elapsed, client_after = perf_counter() - started, process_time()
        server_after = process_cpu(server_pid)
        busy_after, total_after = host_ticks()
    finally:
        if monitor.returncode is None:
            monitor.terminate()
        stdout, stderr = await monitor.communicate()
    gpu_samples = [float(line) for line in stdout.decode().splitlines() if line.strip().isdigit()]
    if not gpu_samples:
        raise RuntimeError(f"Could not sample GPU utilization. {stderr.decode().strip()}")
    result.update(
        host_cpu_percent=100 * (busy_after - busy_before) / (total_after - total_before),
        client_cpu_cores=(client_after - client_before) / elapsed,
        triton_cpu_cores=(server_after - server_before) / elapsed
        if server_before is not None and server_after is not None else None,
        gpu_percent=mean(gpu_samples),
        gpu_samples=len(gpu_samples),
    )
    return result


async def profile(args):
    model_url = f"{args.url.rstrip('/')}/v2/models/retinaloct"
    payload = np.random.default_rng(42).random((1, 3, 256, 256), dtype=np.float32).astype("<f4").tobytes()
    request = inference_request(payload)
    started = process_time()
    for _ in range(200):
        inference_request(payload)
    encode_ms = (process_time() - started) * 1000 / 200

    gpu_info = subprocess.check_output(
        ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    gpu_processes = list(csv.reader(subprocess.check_output(
        ["nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid,process_name",
         "--format=csv,noheader,nounits"], text=True).splitlines(), skipinitialspace=True))
    triton_pids = [pid for pid, name in gpu_processes if Path(name).name == "tritonserver"]
    server_pid = args.server_pid or (triton_pids[0] if len(triton_pids) == 1 else None)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), trust_env=False) as session:
        async with session.get(f"{model_url}/config") as response:
            response.raise_for_status()
            config = await response.json()
        body, offset = await infer(session, f"{model_url}/infer", request)
        if not np.isfinite(decode_logits(body, offset)).all():
            raise RuntimeError("Triton returned non-finite logits")

        async def echo(http_request):
            await http_request.read()
            return web.Response(body=body, headers={"Inference-Header-Content-Length": str(offset)})

        app = web.Application()
        app.router.add_post("/", echo)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        rows = []
        try:
            await web.TCPSite(runner, "127.0.0.1", 0).start()
            echo_url = f"http://127.0.0.1:{runner.addresses[0][1]}/"
            for clients in args.clients:
                print(f"Profiling {clients} concurrent clients", flush=True)
                await measure(session, f"{model_url}/infer", request, clients, args.warmup)
                before = await stats(session, model_url)
                row = await measure_with_resources(session, f"{model_url}/infer", request, clients,
                                                   args.seconds, args.gpu, server_pid)
                after = await stats(session, model_url)
                start_stats, end_stats = before["inference_stats"], after["inference_stats"]
                count = int(end_stats["success"]["count"]) - int(start_stats["success"]["count"])
                if count != row["requests_including_drain"]:
                    raise RuntimeError("Server and client request counts differ. Use a dedicated Triton endpoint.")
                server_ms = {stage: (int(end_stats[stage]["ns"]) - int(start_stats[stage]["ns"])) / count / 1e6
                             for stage in STAGES}
                await measure(session, echo_url, request, clients, 1)
                control = await measure(session, echo_url, request, clients, args.control_seconds)
                row.update(clients=clients, server_ms=server_ms,
                           outside_server_ms=row["http_ms"] - server_ms["success"],
                           loopback_http_ms=control["http_ms"],
                           average_batch=(int(after["inference_count"]) - int(before["inference_count"])) /
                           (int(after["execution_count"]) - int(before["execution_count"])))
                rows.append(row)
        finally:
            await runner.cleanup()

    return {"url": args.url, "gpu": gpu_info, "gpu_processes_before_run": gpu_processes,
            "triton_pid": server_pid, "config": config, "request_bytes": len(request[0]),
            "warmup_seconds": args.warmup, "measurement_seconds": args.seconds,
            "control_seconds": args.control_seconds, "request_encode_cpu_ms_if_rebuilt": encode_ms,
            "request_reused": True, "measurements": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--clients", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--control-seconds", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--server-pid", help="Host PID of Triton when GPU process detection is ambiguous")
    parser.add_argument("--output", type=Path, default=PROJECT / "triton_model_repository/.benchmark"
                        / f"profile-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json")
    args = parser.parse_args()
    if urlsplit(args.url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("use a local Triton endpoint so host CPU, GPU and loopback measurements apply")
    if min(args.seconds, args.warmup, args.control_seconds, *args.clients) < 1:
        parser.error("client counts and durations must be positive")

    report = asyncio.run(profile(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Clients  samples/s  GPU%  CPU%  Triton cores  Client cores  Queue ms  Input ms  Model ms  Output ms  Outside ms  Decode ms  Echo ms")
    for row in report["measurements"]:
        server = row["server_ms"]
        triton_cpu = row["triton_cpu_cores"]
        triton_cores = f"{triton_cpu:.2f}" if triton_cpu is not None else "n/a"
        print(f"{row['clients']:>7}  {row['samples_per_second']:>9.1f}  {row['gpu_percent']:>4.1f}  "
              f"{row['host_cpu_percent']:>4.1f}  {triton_cores:>12}  {row['client_cpu_cores']:>12.2f}  "
              f"{server['queue']:>8.2f}  {server['compute_input']:>8.2f}  {server['compute_infer']:>8.2f}  "
              f"{server['compute_output']:>9.2f}  {row['outside_server_ms']:>10.2f}  "
              f"{row['decode_ms']:>9.3f}  {row['loopback_http_ms']:>7.2f}")
    print(f"Saved complete stage timings to {args.output}")


if __name__ == "__main__":
    main()
