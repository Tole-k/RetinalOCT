#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$2" != --wait-for-gpu ) ]]; then
    printf 'Usage: bash run_triton_batching.sh OUTPUT_DIRECTORY [--wait-for-gpu]\n' >&2
    exit 2
fi

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(cd -- "$1" && pwd -P)
name="retinaloct-benchmark-$$"
release=26.05
python_bin=${RETINALOCT_BENCHMARK_PYTHON:-python}

if [[ ! -s "$root/model.onnx" || ! -s "$root/inputs.json" || ! -s "$root/sample.fp32" ]]; then
    printf 'Prepare the benchmark directory with prepare_triton_batching.py first.\n' >&2
    exit 1
fi
"$python_bin" -B -c 'import aiohttp, numpy'

while true; do
    competing=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ -z "$competing" ]]; then
        break
    fi
    if [[ "${2:-}" != --wait-for-gpu ]]; then
        printf 'GPU compute processes are active: %s\n' "$competing" >&2
        exit 1
    fi
    printf '%s waiting for GPU process(es) %s\n' "$(date -Is)" "$competing"
    sleep 30
done

lscpu > "$root/cpu.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$root/gpu.csv"

if [[ ! -s "$root/model.plan" ]]; then
    printf '%s building the shared TensorRT engine\n' "$(date -Is)"
    docker run --rm --pull=never --gpus=1 --user "$(id -u):$(id -g)" \
        --mount "type=bind,source=$root,target=/work" \
        "nvcr.io/nvidia/tensorrt:$release-py3" /opt/tensorrt/bin/trtexec \
        --onnx=/work/model.onnx --saveEngine=/work/model.plan.partial --fp16 --skipInference \
        --minShapes=images:1x3x256x256 --optShapes=images:4x3x256x256 \
        --maxShapes=images:8x3x256x256 > "$root/builder.log" 2>&1
    mv -- "$root/model.plan.partial" "$root/model.plan"
fi

sha256sum "$root/model.plan" > "$root/engine.sha256"
for model in retinaloct_unbatched retinaloct_dynamic; do
    destination="$root/repository/$model/1/model.plan"
    if [[ ! -e "$destination" ]]; then
        ln -T -- "$root/model.plan" "$destination"
    fi
    [[ "$destination" -ef "$root/model.plan" ]]
done

container=$(docker run --detach --rm --pull=never --gpus=1 --name "$name" \
    -p 127.0.0.1:18080:8000 --mount "type=bind,source=$root/repository,target=/models,readonly" \
    "nvcr.io/nvidia/tritonserver:$release-py3" tritonserver --model-repository=/models \
    --model-control-mode=explicit --disable-auto-complete-config --allow-grpc=false --allow-metrics=false)
cleanup() {
    docker logs "$container" > "$root/triton.log" 2>&1 || true
    docker stop "$container" >/dev/null || true
}
trap cleanup EXIT

for attempt in {1..60}; do
    if curl --silent --fail http://127.0.0.1:18080/v2/health/ready >/dev/null; then
        break
    fi
    sleep 1
done
curl --silent --show-error --fail http://127.0.0.1:18080/v2/health/ready >/dev/null
server_pid=$(docker inspect --format '{{.State.Pid}}' "$container")
printf '%s verifying TensorRT models in Triton\n' "$(date -Is)"
"$python_bin" -Bu "$project/measure_triton_batching.py" "$root" "$server_pid" --verify-only
"$python_bin" -Bu "$project/measure_triton_batching.py" "$root" "$server_pid"
"$python_bin" -B "$project/summarize_triton_batching.py" "$root"
