#! .venv/bin/python

import os
import shutil
import tempfile
from fastcore.xtras import Path
import logging
import yaml
from yaml import Loader

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

import torch
from torchvision import datasets
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import AUROC
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from vit_pytorch.model import VisionTransformer

directory = Path("scripts")
config = yaml.safe_load(open(directory / "training_config.yml", 'r'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


DATA_PATH = config['data']['path']
test_size = config['model']['test_size']
logging.basicConfig(level=eval(config['logging_level']))

class Net(pl.LightningModule):
    def __init__(self, fold_idx=0):
        super().__init__()

        self._model = VisionTransformer(config=config).to(device)
        self.fold_idx = fold_idx
        self.loss_function = CrossEntropyLoss()
        self.metric = AUROC()
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.max_epochs = 1000
        self.check_val = 30
        self.warmup_epochs = 20
        self.metric_values = []
        self.epoch_loss_values = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        # prepare data
        dset = datasets.CIFAR10(DATA_PATH, download=True)
        images, targets = dset.data, dset.targets # (N, 32, 32) and (N, 1)
        indices = np.arange(0, len(dset)) #.reshape(-1, 1)
        img_trn, img_tst, y_trn, y_tst = train_test_split(images, targets, 
                                                          test_size=test_size, 
                                                          stratify=targets, 
                                                          shuffle=True, 
                                                          random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = {}
        for idx, (trn_idx, val_idx) in enumerate(cv.split(img_trn, y_trn)):
            folds[idx] = {"train": trn_idx, "valid": val_idx}
        self.train_data = img_trn, y_trn
        self.test_data = img_tst, y_tst
        self.folds = folds

    def train_dataloader(self):
        indices = self.folds[self.fold_idx]['train']
        train_ds = Dataset(self.train_data[indices])
        train_loader = DataLoader(
            train_ds,
            batch_size=20,
            shuffle=True,
            num_workers=4,
            pin_memory=True, 
        )
        return train_loader

    def val_dataloader(self):
        indices = self.folds[self.fold_idx]['valid']
        val_ds = Dataset(self.train_data[indices])
        val_loader = DataLoader(val_ds, 
                                batch_size=20, 
                                shuffle=False, 
                                num_workers=4, 
                                pin_memory=True)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self._model.parameters(), 
                                      lr=1e-4, 
                                      weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        output = self.forward(images)
        loss = self.loss_function(output, labels) 
        return {"val_loss": loss}

    def on_validation_epoch_end(self):  
        # mean_val_loss = torch.tensor(val_loss / num_items)
        mean_val_score = self.metric.aggregate().item()
        self.metric.reset()
        tensorboard_logs = {
            "val_score": mean_val_score,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.metric_values.append(mean_val_dice)
        self.validation_step_outputs.clear()  # free memory
        return {"log": tensorboard_logs}
    
def run():
    pass


if __name__ == "__main__":
    
