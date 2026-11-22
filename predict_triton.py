"""Send one OCT image to Triton and print its prediction."""

import argparse
import json
from pathlib import Path

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import InferenceServerException

from image_data_module import ImageDataModule

MODEL_NAME = "retinaloct"
CLASSES = ("CNV", "DME", "DRUSEN", "NORMAL")


def predict(image_path: Path, url: str) -> np.ndarray:
    transform = ImageDataModule(batch_size=1, num_workers=0).transforms
    with Image.open(image_path) as image:
        batch = transform(image.convert("RGB")).unsqueeze(0).contiguous().numpy()

    images = httpclient.InferInput("images", batch.shape, "FP32")
    images.set_data_from_numpy(batch)
    logits_output = httpclient.InferRequestedOutput("logits")
    with httpclient.InferenceServerClient(url, connection_timeout=5.0, network_timeout=30.0) as client:
        response = client.infer(MODEL_NAME, [images], outputs=[logits_output])
        logits = response.as_numpy("logits")

    if logits is None or logits.shape != (1, len(CLASSES)) or logits.dtype != np.float32:
        raise RuntimeError("Expected Triton to return an FP32 logits tensor with shape [1, 4].")
    if not np.isfinite(logits).all():
        raise RuntimeError("Triton returned non-finite logits.")
    return logits[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to one OCT image")
    parser.add_argument("--url", default="127.0.0.1:8000", help="Triton HTTP host and port")
    args = parser.parse_args()

    try:
        logits = predict(args.image, args.url)
    except (OSError, InferenceServerException, RuntimeError) as exc:
        parser.exit(1, f"Prediction failed: {exc}\n")

    print(json.dumps({
        "image": args.image.name,
        "model": MODEL_NAME,
        "prediction": CLASSES[int(logits.argmax())],
        "logits": {label: round(float(score), 6) for label, score in zip(CLASSES, logits, strict=True)},
    }, indent=2))


if __name__ == "__main__":
    main()
