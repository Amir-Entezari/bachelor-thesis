import torch 

import numpy as np

import pytorch_lightning as pl

from pytorch_lightning.strategies import DDPStrategy

from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn

# !pip install --upgrade packaging


import pytorch_lightning as pl
import torch.nn as nn


class MyCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.training_step_outputs.clear()

class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model, train_loader=None):
        super().__init__()
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.test_step_outputs = []
        self.training_step_outputs = []

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X, perturb=True).squeeze()
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        step_result = logits.cpu().numpy()
        step_gt = y.cpu().numpy()
        test_out = (step_result, step_gt)
        self.test_step_outputs.append(test_out)

    def on_test_epoch_end(self):
        # Compile test outputs
        gt = np.concatenate([out[1] for out in self.test_step_outputs], axis=0)
        outputs = np.vstack([out[0] for out in self.test_step_outputs])

        if outputs.ndim == 3:  # Check if outputs have an extra dimension
            outputs = outputs.squeeze(axis=1)
        print(outputs)
        # Compute metrics
        result = multiclass_metrics_fn(
            gt, outputs, metrics=["accuracy", "cohen_kappa", "f1_weighted"]
        )

        # Log test metrics
        self.log("test_accuracy", result["accuracy"], on_epoch=True, sync_dist=True)
        self.log("test_cohen_kappa", result["cohen_kappa"], on_epoch=True, sync_dist=True)
        self.log("test_f1_weighted", result["f1_weighted"], on_epoch=True, sync_dist=True)

        # Clear outputs for the next LOSO iteration
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args['lr'],
            weight_decay=self.args['weight_decay'],
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[int(1 * self.args['n_step']), int(4 * self.args['n_step'])],
            gamma=self.args['gamma']
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def train_dataloader(self):
        return self.train_loader