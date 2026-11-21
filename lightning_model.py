from typing import Any

import lightning as L
import numpy as np
import torch
from torch import nn, optim
from torchmetrics import AUROC, Accuracy
from torchvision.models import get_model


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
                "validation auc": auc
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
    import torch
    from lightning.pytorch.callbacks import (
        BackboneFinetuning,
        DeviceStatsMonitor,
        EarlyStopping,
        ModelCheckpoint,
    )
    from lightning.pytorch.utilities import rank_zero_info

    from image_data_module import ImageDataModule

    torch.set_float32_matmul_precision("medium")
    model = Model()
    backbone_ft = BackboneFinetuning(5)
    early_stopping = EarlyStopping("val loss", patience=5)
    device_stast = DeviceStatsMonitor()
    dm = ImageDataModule(32, 4)
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
    trainer = L.Trainer(
        accelerator="gpu",
        devices="auto",
        logger=True,
        precision="16-mixed",
        callbacks=[backbone_ft, early_stopping, device_stast, checkpoint],
    )
    trainer.fit(model, dm)
    rank_zero_info(f"Saved best model to {checkpoint.best_model_path}")
    trainer.test(model, dm, ckpt_path="best")
