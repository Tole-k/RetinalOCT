"""Export a Lightning checkpoint into a fresh Triton model repository."""

import argparse
from pathlib import Path
from stat import S_ISREG
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from lightning_model import Model

LOGS_DIRECTORY = Path(__file__).resolve().parents[1] / "lightning_logs"


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


def write_config(model_directory: Path, platform: str) -> None:
    model_directory.joinpath("config.pbtxt").write_text(
        f'''name: "{model_directory.name}"
platform: "{platform}"
max_batch_size: 0
input [{{ name: "images" data_type: TYPE_FP32 dims: [1, 3, 256, 256] }}]
output [{{ name: "logits" data_type: TYPE_FP32 dims: [1, 4] }}]
instance_group [{{ kind: KIND_GPU count: 1 }}]
''',
        encoding="utf-8",
    )


def export_model(checkpoint: Path, destination: Path) -> None:
    model = Model.load_from_checkpoint(checkpoint, map_location="cpu", pretrained=False).eval()
    network = nn.Sequential(model.backbone, model.head).eval()
    generator = torch.Generator().manual_seed(0)
    sample = torch.rand(1, 3, 256, 256, generator=generator, dtype=torch.float32)

    with torch.no_grad():
        expected = model(sample).numpy()
        torch.onnx.export(
            network,
            (sample,),
            destination,
            input_names=["images"],
            output_names=["logits"],
            dynamo=True,
            opset_version=18,
            external_data=False,
        )

    onnx.checker.check_model(str(destination))
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    (actual,) = session.run(["logits"], {"images": sample.numpy()})
    if actual.shape != (1, 4) or actual.dtype != np.float32 or not np.isfinite(actual).all():
        raise RuntimeError("The exported model must return four finite FP32 logits.")
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, help="Override the latest training run's best checkpoint"
    )
    parser.add_argument("--repository", required=True, type=Path, help="New Triton repository directory")
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_best_checkpoint(LOGS_DIRECTORY)
    if checkpoint is None:
        parser.error(f"no best-*.ckpt found in {LOGS_DIRECTORY}; supply --checkpoint to use another file")
    if not checkpoint.is_file():
        parser.error(f"checkpoint file does not exist: {checkpoint}")
    checkpoint = checkpoint.resolve()
    repository = args.repository.absolute()
    if repository.exists() or repository.is_symlink():
        parser.error(f"choose a new repository directory: {repository}")

    print(f"Using checkpoint {checkpoint}", flush=True)
    repository.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{repository.name}-", dir=repository.parent) as directory:
        staged_repository = Path(directory) / "repository"
        onnx_model = staged_repository / "retinaloct_onnx"
        tensorrt_model = staged_repository / "retinaloct_trt"
        (onnx_model / "1").mkdir(parents=True)
        (tensorrt_model / "1").mkdir(parents=True)

        export_model(checkpoint, onnx_model / "1" / "model.onnx")
        write_config(onnx_model, "onnxruntime_onnx")
        write_config(tensorrt_model, "tensorrt_plan")
        (staged_repository / "source_checkpoint.txt").write_text(f"{checkpoint}\n", encoding="utf-8")
        staged_repository.rename(repository)

    print(f"ONNX export passed CPU parity and was saved to {repository}")


if __name__ == "__main__":
    main()
