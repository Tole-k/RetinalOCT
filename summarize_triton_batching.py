"""Validate the raw latency samples and summarize completed trial pairs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np


METRICS = ["samples_per_second", "p50_ms", "p95_ms", "p99_ms"]


def summarize(output_directory: Path) -> None:
    grouped = defaultdict(list)
    rows = []
    for trial in sorted((output_directory / "trials").glob("*.json")):
        row = json.loads(trial.read_text())
        samples = np.load(trial.with_suffix(".npy"), allow_pickle=False)
        assert row["seconds"] == 60
        assert samples.shape == (row["samples"], 2)
        assert np.isfinite(samples).all() and (samples >= 0).all()
        assert (samples[:, 0] < 60).all()
        assert row["samples_per_second"] == len(samples) / 60
        np.testing.assert_allclose([row[key] for key in METRICS[1:]], np.quantile(samples[:, 1], [0.5, 0.95, 0.99]))
        rows.append(row)
        grouped[(row["clients"], row["dynamic_batching"])].append(row)

    assert set(grouped) == {(clients, mode) for clients in (1, 8, 32) for mode in ("off", "on")}
    assert all(sorted(row["repetition"] for row in group) == [1, 2, 3] for group in grouped.values())
    fields = ["repetition", "clients", "dynamic_batching", "seconds", "samples", *METRICS, "batch_executions"]
    with (output_directory / "trials.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**row, "batch_executions": json.dumps(row["batch_executions"])} for row in rows)

    summary = []
    for (clients, mode), group in sorted(grouped.items()):
        batches = defaultdict(int)
        for row in group:
            for size, count in row["batch_executions"].items():
                batches[int(size)] += count
        average_batch = sum(size * count for size, count in batches.items()) / sum(batches.values())
        record = {"clients": clients, "dynamic_batching": mode, "trials": len(group)}
        for metric in METRICS:
            values = [row[metric] for row in group]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_sd"] = stdev(values)
        record["mean_server_batch_size"] = average_batch
        summary.append(record)

    with (output_directory / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Validated all {len(rows)} trials and wrote {output_directory / 'summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    summarize(parser.parse_args().output_directory)
