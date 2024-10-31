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
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.threshold = 0.5 # threshold for binary classification task
        self.validation_step_outputs = [] # for epoch-level validation
        self.test_step_outputs = [] # for epoch-level test
        self.training_step_outputs = []

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X, perturb = True).squeeze() # perturb = True for training # squeeze for size match with target
        target = y.float() # output of model and target shouldd be the same datatype
        criterion = torch.nn.BCEWithLogitsLoss( # do not feed inputs and target in object creation stage
          pos_weight= torch.tensor([651/886], # number of negatives / number of positives in dataset
          device= self.device) # use self.device for matching the calculations in model
          )
        loss = criterion(logits, target)
        self.training_step_outputs.append(loss)
        self.log("train_loss", loss, on_epoch=True, prog_bar= False, sync_dist=True) # save and show the train_loss on epoch and step both in progress bar
        return loss # return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        probs = torch.sigmoid(logits).cpu().numpy()
        target = y.cpu().numpy()
        val_out = probs, target
        self.validation_step_outputs.append(val_out)
        # you do not need to return anything

    def on_validation_epoch_end(self):
        # save outputs as instance attributes
        validation_outputs = self.validation_step_outputs # self.outputs
        probs = np.array([])
        target = np.array([])
        for i_out in range(len(validation_outputs)):
            probs = np.append(probs, validation_outputs[i_out][0])
            target = np.append(target, validation_outputs[i_out][1])
        if (
            sum(target) * (len(target) - sum(target)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            result = binary_metrics_fn(
                target,
                probs,
                metrics=["f1",  "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "f1": 0.0
            }
        self.log("val_acc", result["accuracy"], on_epoch=True, prog_bar= False, sync_dist=True)
        self.log("val_bacc", result["balanced_accuracy"], on_epoch=True, prog_bar= False, sync_dist=True)
        self.log("f1", result["f1"], on_epoch=True, prog_bar= False, sync_dist=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        probs = torch.sigmoid(logits).cpu().numpy()
        target = y.cpu().numpy()
        test_out = probs, target
        self.test_step_outputs.append(test_out)

    def on_test_epoch_end(self):
        test_outputs = self.test_step_outputs
        probs = np.array([])
        target = np.array([])
        for i_out in range(len(test_outputs)):
            probs = np.append(probs, test_outputs[i_out][0])
            target = np.append(target, test_outputs[i_out][1])
            print(probs)
            print(target)
        if (
            sum(target) * (len(target) - sum(target)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            result = binary_metrics_fn(
                target,
                probs,
                metrics=["f1", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "f1": 0.0
            }
        self.log("test_acc", result["accuracy"], on_epoch=True, prog_bar= True, sync_dist=True)
        self.log("test_bacc", result["balanced_accuracy"], on_epoch=True, prog_bar= True, sync_dist=True)
        self.log("f1", result["f1"], on_epoch=True, prog_bar= True, sync_dist=True)
        self.print("test_acc", result["accuracy"])
        self.print("test_bacc", result["balanced_accuracy"])
        self.print("f1", result["f1"])
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-5,
        )
        # set learning rate scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones= [int(1 * 100),
            int(4 * 100)
            ] , gamma= 0.1
        )
        opt_dictionary = {"optimizer": optimizer,
         "lr_scheduler": {"scheduler": scheduler,
                          "interval": 'epoch',
                          },
                            }
        return opt_dictionary