from typing import Any
import lightning as L

from torchvision.models import get_model
import torch
from torch import nn, optim
from torchmetrics import Accuracy, AUROC
import numpy as np


class Model(L.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = get_model("swin_v2_b", weights="DEFAULT")
        self.head = nn.Linear(1024, 4)
        self.loss = nn.CrossEntropyLoss()
        self.acc = Accuracy("multiclass", num_classes=4)
        self.auc = AUROC("multiclass", num_classes=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def training_step(self, batch: torch.Tensor, batch_id: int) -> torch.Tensor:
        loss, _, _ = self._step(batch)
        self.log("train loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id: int) -> torch.Tensor:
        loss, outputs, y = self._step(batch)
        acc = self.acc(outputs, y)
        auc = self.auc(outputs, y)
        self.log_dict(
            {"validation loss": loss, "validation accuracy": acc, "validation auc": auc},
            on_step=True,
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
            {"validation loss": loss, "validation accuracy": acc, "validation auc": auc},
            on_step=True,
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
        return optim.AdamW(self.parameters())


if __name__ == "__main__":
    from lightning.pytorch.callbacks import BackboneFinetuning, EarlyStopping
    from dataset_download import ImageDataModule
    import torch

    torch.set_float32_matmul_precision("medium")
    model = Model()
    backbone_ft = BackboneFinetuning()
    early_stopping = EarlyStopping("validation loss")
    dm = ImageDataModule(128, 16)
    trainer = L.Trainer(
        accelerator="gpu",
        devices="auto",
        logger=True,
        precision="16-mixed",
        callbacks=[backbone_ft, early_stopping],
        profiler="simple",
        fast_dev_run=1,
    )
    trainer.fit(model, dm)
    trainer = L.Trainer(
        accelerator="gpu", devices="auto", logger=True, precision="16-mixed", callbacks=[backbone_ft, early_stopping], profiler="simple"
    )
    trainer.fit(model, dm)
