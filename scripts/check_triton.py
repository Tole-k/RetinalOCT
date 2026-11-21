"""Compare logits and top predictions from Triton with the training checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import torch
import tritonclient.http as httpclient
from torchvision.datasets import ImageFolder

from image_data_module import ImageDataModule
from lightning_model import Model

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]


@torch.inference_mode()
def check_model(checkpoint: Path, test_directory: Path, model_name: str, url: str) -> None:
    with httpclient.InferenceServerClient(url, connection_timeout=5.0, network_timeout=60.0) as client:
        if not client.is_model_ready(model_name, model_version="1"):
            raise RuntimeError(f"Model {model_name} version 1 is not ready at {url}.")

        transform = ImageDataModule(batch_size=1, num_workers=0).transforms
        dataset = ImageFolder(test_directory, transform=transform)
        if dataset.classes != CLASSES:
            raise RuntimeError(f"Expected class order {CLASSES}, got {dataset.classes}.")

        model = Model.load_from_checkpoint(checkpoint, map_location="cpu", pretrained=False).eval()
        images = httpclient.InferInput("images", [1, 3, 256, 256], "FP32")
        logits = httpclient.InferRequestedOutput("logits")
        max_difference = 0.0

        for index in range(len(dataset)):
            image, _ = dataset[index]
            image_path = dataset.samples[index][0]
            batch = image.unsqueeze(0).contiguous()
            expected = model(batch).numpy()
            images.set_data_from_numpy(batch.numpy())
            response = client.infer(model_name, [images], model_version="1", outputs=[logits])
            actual = response.as_numpy("logits")

            if actual is None or actual.shape != (1, 4) or actual.dtype != np.float32:
                raise RuntimeError(f"{image_path}: expected an FP32 logits tensor of shape [1, 4].")
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise RuntimeError(f"{image_path}: found non-finite logits.")

            pytorch_class = CLASSES[int(expected.argmax(axis=1)[0])]
            triton_class = CLASSES[int(actual.argmax(axis=1)[0])]
            if triton_class != pytorch_class:
                raise RuntimeError(
                    f"{image_path}: PyTorch predicted {pytorch_class}, Triton predicted {triton_class}."
                )

            max_difference = max(max_difference, float(np.max(np.abs(actual - expected))))

    print(f"{model_name} matched all {len(dataset)} PyTorch predictions.")
    print(f"Maximum absolute logit difference: {max_difference:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, help="Triton repository created by the export script")
    parser.add_argument("--checkpoint", type=Path, help="Override the checkpoint recorded at export")
    parser.add_argument("--test-directory", required=True, type=Path, help="Held-out ImageFolder test split")
    parser.add_argument("--model", required=True, choices=["retinaloct_onnx", "retinaloct_trt"])
    parser.add_argument("--url", default="127.0.0.1:8000", help="Triton HTTP host and port")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        if args.repository is None:
            parser.error("supply --repository to read the checkpoint used for export, or supply --checkpoint")
        source_file = args.repository / "source_checkpoint.txt"
        try:
            checkpoint = Path(source_file.read_text(encoding="utf-8").removesuffix("\n"))
        except (OSError, UnicodeError):
            parser.error(f"could not read {source_file}; supply the original --checkpoint explicitly")

    if not checkpoint.is_file():
        parser.error(f"checkpoint file does not exist: {checkpoint}")
    if not args.test_directory.is_dir():
        parser.error(f"test directory does not exist: {args.test_directory}")
    check_model(checkpoint, args.test_directory, args.model, args.url)


if __name__ == "__main__":
    main()
