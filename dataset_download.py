import kagglehub
import lightning as L
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os


class ImageDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        self.path = kagglehub.dataset_download("paultimothymooney/kermany2018")

    def setup(self, stage: str) -> None:
        self.train_ds = ImageFolder(os.path.join(self.path, "OCT2017", "train"))
        self.val_ds = ImageFolder(os.path.join(self.path, "OCT2017", "val"))
        self.test_ds = ImageFolder(os.path.join(self.path, "OCT2017", "test"))

    def train_dataloader(self):
        return DataLoader(self.train_ds, self.batch_size, True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, self.batch_size, num_workers=self.num_workers)
