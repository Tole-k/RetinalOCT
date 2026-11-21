"""Export a batch-capable RetinalOCT graph and write the comparison configs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from lightning_model import Model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint file does not exist: {args.checkpoint}")
    output_dir = args.output_directory.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        parser.error(f"choose an empty benchmark directory: {output_dir}")
    torch.set_num_threads(2)
    model = Model.load_from_checkpoint(args.checkpoint, map_location="cpu", pretrained=False).eval()
    network = nn.Sequential(model.backbone, model.head).eval()
    sample = np.random.default_rng(42).random((1, 3, 256, 256), dtype=np.float32).astype("<f4")
    tensor = torch.from_numpy(sample)

    export_options = (
        {"dynamic_axes": {"images": {0: "batch"}, "logits": {0: "batch"}}}
        if args.legacy
        else {"dynamic_shapes": ({0: torch.export.Dim("batch", min=1, max=8)},)}
    )
    path = output_dir / "model.onnx"
    with torch.no_grad():
        reference = model(tensor).numpy()
        torch.onnx.export(
            network,
            (tensor.repeat(2, 1, 1, 1),),
            path,
            input_names=["images"],
            output_names=["logits"],
            opset_version=18,
            dynamo=not args.legacy,
            external_data=False,
            **export_options,
        )

    onnx.checker.check_model(str(path))
    graph = onnx.load(path, load_external_data=False).graph
    for value in [graph.input[0], graph.output[0]]:
        if not value.type.tensor_type.shape.dim[0].dim_param:
            raise RuntimeError(f"{value.name} lacks a symbolic batch dimension")

    payload = sample.tobytes()
    (output_dir / "sample.fp32").write_bytes(payload)
    (output_dir / "inputs.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "reference_logits": reference.tolist(),
        "exporter": "legacy" if args.legacy else "dynamo",
    }, indent=2) + "\n")

    for name, batching in (("retinaloct_unbatched", ""), ("retinaloct_dynamic", """
dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 2000
}
""")):
        model_directory = output_dir / "repository" / name
        (model_directory / "1").mkdir(parents=True, exist_ok=True)
        (model_directory / "config.pbtxt").write_text(f'''name: "{name}"
platform: "tensorrt_plan"
max_batch_size: 8
input [{{ name: "images" data_type: TYPE_FP32 dims: [3, 256, 256] }}]
output [{{ name: "logits" data_type: TYPE_FP32 dims: [4] }}]
instance_group [{{ kind: KIND_GPU count: 1 }}]
{batching}''')
    print(f"Exported {path} with a symbolic batch dimension")


if __name__ == "__main__":
    main()
