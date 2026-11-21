#!/usr/bin/env bash
set -euo pipefail

release=26.05

usage() {
    cat >&2 <<'EOF'
Usage:
  bash scripts/triton.sh build REPOSITORY
  bash scripts/triton.sh serve REPOSITORY MODEL
  bash scripts/triton.sh benchmark MODEL RESULTS_DIRECTORY

MODEL is retinaloct_onnx or retinaloct_trt.
EOF
    exit 2
}

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

require_model() {
    case "$1" in
        retinaloct_onnx|retinaloct_trt) ;;
        *) usage ;;
    esac
}

case "${1:-}" in
    build)
        [[ $# -eq 2 ]] || usage
        repository=$(cd -- "$2" && pwd -P)
        engine_directory="$repository/retinaloct_trt/1"
        engine="$engine_directory/model.plan"

        [[ -s "$repository/retinaloct_onnx/1/model.onnx" ]] || fail "Export the ONNX model first."
        [[ -f "$repository/retinaloct_trt/config.pbtxt" && -d "$engine_directory" ]] || fail "TensorRT repository layout is missing."
        [[ ! -e "$engine" && ! -L "$engine" ]] || fail "An engine already exists at $engine."

        staging=$(mktemp -d "$engine_directory/.build.XXXXXXXX")
        trap 'rm -f -- "$staging/model.plan"; rmdir -- "$staging"' EXIT
        docker run --rm --gpus=1 --user "$(id -u):$(id -g)" \
            --mount "type=bind,source=$repository,target=/models,readonly" \
            --mount "type=bind,source=$staging,target=/output" \
            "nvcr.io/nvidia/tensorrt:$release-py3" \
            /opt/tensorrt/bin/trtexec \
            --onnx=/models/retinaloct_onnx/1/model.onnx \
            --saveEngine=/output/model.plan \
            --fp16 --skipInference

        [[ -s "$staging/model.plan" ]] || fail "TensorRT did not produce an engine."
        ln -T -- "$staging/model.plan" "$engine"
        printf 'Saved TensorRT engine to %s\n' "$engine"
        ;;
    serve)
        [[ $# -eq 3 ]] || usage
        repository=$(cd -- "$2" && pwd -P)
        model=$3
        require_model "$model"

        exec docker run --rm --gpus=1 \
            -p 127.0.0.1:8000:8000 \
            --mount "type=bind,source=$repository,target=/models,readonly" \
            "nvcr.io/nvidia/tritonserver:$release-py3" \
            tritonserver --model-repository=/models \
            --model-control-mode=explicit --load-model="$model"
        ;;
    benchmark)
        [[ $# -eq 3 ]] || usage
        model=$2
        require_model "$model"
        curl --fail --silent --show-error "http://127.0.0.1:8000/v2/models/$model/ready" >/dev/null
        mkdir -p -- "$3"
        results=$(cd -- "$3" && pwd -P)
        trial_directory=$(mktemp -d "$results/$model.XXXXXXXX")
        printf 'Benchmark CSV files will be saved to %s\n' "$trial_directory"

        for trial in 1 2 3; do
            docker run --rm --gpus=1 --network=host --user "$(id -u):$(id -g)" \
                --mount "type=bind,source=$trial_directory,target=/results" \
                "nvcr.io/nvidia/tritonserver:$release-py3-sdk" \
                perf_analyzer \
                -m "$model" -u 127.0.0.1:8000 \
                --input-data=zero --concurrency-range=1:1:1 \
                --warmup-request-count=100 --measurement-interval=10000 --percentile=95 \
                -f "/results/trial-$trial.csv"
            [[ -s "$trial_directory/trial-$trial.csv" ]] || fail "Trial $trial did not produce a CSV file."
        done
        ;;
    *) usage ;;
esac
