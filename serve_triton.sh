#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
    quick_batched|batched|unbatched) mode=$1 ;;
    *) printf 'Usage: bash serve_triton.sh {unbatched|batched|quick_batched} [REPOSITORY]\n' >&2; exit 2 ;;
esac

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repository=$(cd -- "${2:-$project/triton_model_repository}" && pwd -P)
[[ -s "$repository/retinaloct/1/model.plan" ]] || { printf 'Run build_tensorrt.sh first.\n' >&2; exit 1; }
[[ -s "$repository/retinaloct/configs/$mode.pbtxt" ]] || { printf 'Missing %s config.\n' "$mode" >&2; exit 1; }

exec docker run --rm --gpus=1 -p 127.0.0.1:8000:8000 \
    --mount "type=bind,source=$repository,target=/models,readonly" \
    nvcr.io/nvidia/tritonserver:26.05-py3 tritonserver --model-repository=/models \
    --model-control-mode=explicit --load-model=retinaloct \
    --model-config-name="$mode" --disable-auto-complete-config
