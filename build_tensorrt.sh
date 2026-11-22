#!/usr/bin/env bash
set -euo pipefail

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repository=$(cd -- "${1:-$project/triton_model_repository}" && pwd -P)
model="$repository/retinaloct"
engine="$model/1/model.plan"

[[ -s "$repository/model.onnx" ]] || { printf 'Run export_triton.py first.\n' >&2; exit 1; }
batch_size=$(sed -nE 's/^[[:blank:]]*max_batch_size[[:blank:]]*:[[:blank:]]*([0-9]+)\b.*/\1/p' "$model/configs/batched.pbtxt")
if [[ ! "$batch_size" =~ ^[0-9]+$ ]] || (( batch_size < 2 )); then
    printf 'The batched config must declare a max_batch_size of at least 2.\n' >&2
    exit 1
fi
for mode in unbatched quick_batched; do
    mode_batch_size=$(sed -nE 's/^[[:blank:]]*max_batch_size[[:blank:]]*:[[:blank:]]*([0-9]+)\b.*/\1/p' "$model/configs/$mode.pbtxt")
    if [[ "$mode_batch_size" != "$batch_size" ]]; then
        printf 'The %s config must also declare max_batch_size: %s.\n' "$mode" "$batch_size" >&2
        exit 1
    fi
done
opt_size=$((batch_size < 4 ? batch_size : 4))

mkdir -p -- "$model/1"
staging=$(mktemp -d "$repository/.build.XXXXXXXX")
trap 'rm -f -- "$staging/model.plan"; rmdir -- "$staging"' EXIT
docker run --rm --gpus=1 --user "$(id -u):$(id -g)" \
    --mount "type=bind,source=$repository,target=/models,readonly" \
    --mount "type=bind,source=$staging,target=/output" \
    nvcr.io/nvidia/tensorrt:26.05-py3 /opt/tensorrt/bin/trtexec \
    --onnx=/models/model.onnx --saveEngine=/output/model.plan --fp16 --skipInference \
    --minShapes=images:1x3x256x256 --optShapes="images:${opt_size}x3x256x256" \
    --maxShapes="images:${batch_size}x3x256x256"

[[ -s "$staging/model.plan" ]]
mv -T -- "$staging/model.plan" "$engine"
printf 'Saved TensorRT engine to %s\n' "$engine"
