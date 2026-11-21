import os
from typing import Any

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import BackboneFinetuning
from torch import nn, optim
from torch.utils.checkpoint import checkpoint_sequential
from torchmetrics import AUROC, Accuracy
from torchvision.models import get_model


class CheckpointedSwinFeatures(nn.Sequential):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training or not torch.is_grad_enabled():
            return super().forward(input)

        for layer in self:
            if isinstance(layer, nn.Sequential):
                input = checkpoint_sequential(layer, len(layer), input, use_reentrant=False)
            else:
                input = layer(input)
        return input


class ResumableBackboneFinetuning(BackboneFinetuning):
    @property
    def state_key(self) -> str:
        return BackboneFinetuning.__qualname__

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_fit_start(trainer, pl_module)
        if self.previous_backbone_lr is not None:
            self.make_trainable(pl_module.backbone)


class Model(L.LightningModule):
    def __init__(self, *args: Any, pretrained: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        backbone = get_model("swin_v2_b", weights="DEFAULT" if pretrained else None)
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(1024, 4)
        self.loss = nn.CrossEntropyLoss()
        self.acc = Accuracy("multiclass", num_classes=4)
        self.auc = AUROC("multiclass", num_classes=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def training_step(self, batch: torch.Tensor, batch_id: int) -> torch.Tensor:
        loss, _, _ = self._step(batch)
        self.log("train loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id: int) -> torch.Tensor:
        loss, outputs, y = self._step(batch)
        acc = self.acc(outputs, y)
        auc = self.auc(outputs, y)
        self.log("val loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict(
            {
                "val acc": acc,
                "validation auc": auc
            },
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_id: int) -> torch.Tensor:
        loss, outputs, y = self._step(batch)
        acc = self.acc(outputs, y)
        auc = self.auc(outputs, y)
        self.log_dict(
            {
                "test loss": loss,
                "test acc": acc,
                "test auc": auc
            },
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def predict_step(self, batch, batch_id) -> Any:
        _, outputs, _ = self._step(batch)
        return np.argmax(outputs)

    def _step(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        outputs = self(x)
        loss = self.loss(outputs, y)
        return loss, outputs, y

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import torch
    from lightning.pytorch.callbacks import (
        DeviceStatsMonitor,
        EarlyStopping,
        ModelCheckpoint,
    )
    from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger
    from lightning.pytorch.utilities import rank_zero_info

    from image_data_module import ImageDataModule

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", type=Path, help="Lightning checkpoint to resume training from")
    parser.add_argument("--wandb-run-id", help="Existing W&B run ID for resumed training")
    parser.add_argument(
        "--activation-checkpointing", action="store_true", help="Recompute Swin activations to save GPU memory"
    )
    args = parser.parse_args()

    if bool(args.resume_from) != bool(args.wandb_run_id):
        parser.error("supply --resume-from and --wandb-run-id together")

    tensorboard_version = None
    if args.resume_from is not None:
        if not args.resume_from.is_file():
            parser.error(f"checkpoint file does not exist: {args.resume_from}")
        run_directory = args.resume_from.resolve().parent.parent
        version = run_directory.name.removeprefix("version_")
        if run_directory.parent == Path("lightning_logs").resolve() and version.isdecimal():
            tensorboard_version = int(version)

    torch.set_float32_matmul_precision("medium")
    model = Model(pretrained=args.resume_from is None)
    if args.activation_checkpointing:
        model.backbone.features = CheckpointedSwinFeatures(*model.backbone.features.children())
    backbone_ft = ResumableBackboneFinetuning(5)
    early_stopping = EarlyStopping("val loss", patience=5)
    device_stast = DeviceStatsMonitor()
    dm = ImageDataModule(128, 8)
    if args.resume_from is None:
        trainer = L.Trainer(
            accelerator="gpu",
            devices="auto",
            logger=True,
            precision="16-mixed",
            callbacks=[backbone_ft, early_stopping],
            enable_checkpointing=False,
            fast_dev_run=1,
        )
        trainer.fit(model, dm)
    checkpoint = ModelCheckpoint(
        monitor="val loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{step}",
    )
    loggers: list[Logger] = [
        TensorBoardLogger(save_dir=".", version=tensorboard_version),
        WandbLogger(
            project=os.getenv("WANDB_PROJECT", "retinaloct"),
            id=args.wandb_run_id,
            resume="must" if args.resume_from is not None else "allow",
        ),
    ]

    trainer = L.Trainer(
        accelerator="gpu",
        devices="auto",
        logger=loggers,
        precision="16-mixed",
        callbacks=[backbone_ft, early_stopping, device_stast, checkpoint],
    )
    trainer.fit(model, dm, ckpt_path=args.resume_from)
    rank_zero_info(f"Saved best model to {checkpoint.best_model_path}")
    trainer.test(model, dm, ckpt_path="best")
