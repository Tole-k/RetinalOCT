import os

import kagglehub
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    def prepare_data(self) -> None:
        self.path = kagglehub.dataset_download("paultimothymooney/kermany2018")

    def setup(self, stage: str) -> None:
        self.train_ds = ImageFolder(os.path.join(self.path, "OCT2017 ", "train"), self.transforms)
        self.val_ds = ImageFolder(os.path.join(self.path, "OCT2017 ", "val"), self.transforms)
        self.test_ds = ImageFolder(os.path.join(self.path, "OCT2017 ", "test"), self.transforms)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
