"""Export TensorRT build input into an existing Triton repository."""

import argparse
import re
from pathlib import Path
from stat import S_ISREG
from tempfile import NamedTemporaryFile

import onnx
import torch
from torch import nn

from lightning_model import Model

PROJECT = Path(__file__).resolve().parent
LOGS_DIRECTORY = PROJECT / "lightning_logs"
SOURCE_CHECKPOINT_PROPERTY = "retinaloct.source_checkpoint"


def find_best_checkpoint(logs_directory: Path) -> Path | None:
    candidates = []
    for checkpoint in logs_directory.glob("version_*/checkpoints/best-*.ckpt"):
        version = checkpoint.parent.parent.name.removeprefix("version_")
        if not version.isdecimal():
            continue
        try:
            metadata = checkpoint.stat()
        except FileNotFoundError:
            continue  # Lightning can prune an older best checkpoint during training.
        if S_ISREG(metadata.st_mode):
            candidates.append((int(version), metadata.st_mtime_ns, checkpoint))
    return max(candidates)[2] if candidates else None


def export_model(checkpoint: Path, destination: Path, max_batch_size: int) -> None:
    model = Model.load_from_checkpoint(checkpoint, map_location="cpu", pretrained=False).eval()
    network = nn.Sequential(model.backbone, model.head).eval()
    dynamic_shapes = None
    sample_batch_size = 1
    if max_batch_size > 1:
        dynamic_shapes = ({0: torch.export.Dim("batch", min=1, max=max_batch_size)},)
        sample_batch_size = 2
    sample = torch.zeros(sample_batch_size, 3, 256, 256, dtype=torch.float32)

    with torch.no_grad():
        program = torch.onnx.export(
            network,
            (sample,),
            input_names=["images"],
            output_names=["logits"],
            dynamo=True,
            opset_version=18,
            dynamic_shapes=dynamic_shapes,
        )

    if program is None:
        raise RuntimeError("The ONNX exporter did not return a model")
    program.model.metadata_props[SOURCE_CHECKPOINT_PROPERTY] = str(checkpoint)
    program.save(destination, external_data=False)
    onnx.checker.check_model(str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, help="Override the latest training run's best checkpoint"
    )
    parser.add_argument("--repository", type=Path, default=PROJECT / "triton_model_repository")
    args = parser.parse_args()

    repository = args.repository.resolve()
    batch_sizes = []
    for mode in ("batched", "unbatched"):
        config = repository / "retinaloct" / "configs" / f"{mode}.pbtxt"
        if not config.is_file():
            parser.error(f"expected an existing Triton configuration at {config}")
        batch_size = re.search(r"(?m)^[ \t]*max_batch_size[ \t]*:[ \t]*(\d+)\b", config.read_text())
        if batch_size is None:
            parser.error(f"missing max_batch_size in {config}")
        batch_sizes.append(int(batch_size[1]))
    max_batch_size, unbatched_max = batch_sizes
    if max_batch_size < 2 or max_batch_size != unbatched_max:
        parser.error("both configurations must declare the same max_batch_size of at least 2")

    checkpoint = args.checkpoint or find_best_checkpoint(LOGS_DIRECTORY)
    if checkpoint is None:
        parser.error(f"no best-*.ckpt found in {LOGS_DIRECTORY}; supply --checkpoint to use another file")
    if not checkpoint.is_file():
        parser.error(f"checkpoint file does not exist: {checkpoint}")
    checkpoint = checkpoint.resolve()
    print(f"Using checkpoint {checkpoint}", flush=True)
    destination = repository / "model.onnx"
    with NamedTemporaryFile(prefix=".onnx-export-", suffix=".onnx", dir=repository, delete=False) as temporary:
        staged_model = Path(temporary.name)
    try:
        export_model(checkpoint, staged_model, max_batch_size)
        staged_model.replace(destination)
    finally:
        staged_model.unlink(missing_ok=True)

    print(f"Exported {destination}")


if __name__ == "__main__":
    main()
