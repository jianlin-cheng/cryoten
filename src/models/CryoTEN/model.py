import glob
import os
import shutil
from typing import Any

import torch
from lightning.pytorch import LightningModule
import torchmetrics
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
import glob

class CryoTENLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ReduceLROnPlateau,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])
        
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.net = net

        # loss function
        self.mse_loss = torchmetrics.MeanSquaredError()

        self.example_input_array = torch.zeros(1, 1, 48, 48, 48)
    
    def forward(self, x: torch.Tensor):
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int):
        x, y, emd_id_list_batch = batch
        
        preds = self.forward(x)

        mask = (preds == 0) & (y == 0)
        masked_preds = preds[~mask]
        masked_y = y[~mask]

        loss = self.mse_loss(torch.flatten(masked_preds), torch.flatten(masked_y))

        # Calculate and log other metrics if needed
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch), sync_dist=True
        )

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, emd_id_list_batch = batch
        preds = self.forward(x)

        loss = self.mse_loss(torch.flatten(preds), torch.flatten(y))

        # Calculate and log other metrics if needed
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch), sync_dist=True
        )

        return {"loss": loss}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, files = batch
        
        preds = self.forward(x)

        return preds, files

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "strict": True,
                "frequency": 6,
            },
        }
