import torch
import torch.nn as nn
import pytorch_lightning as pl
from model.TSPN import Transparent_Signal_Processing_Network
# from config import args
# from config import signal_processing_modules,feature_extractor_modules
from pytorch_lightning.callbacks import ModelCheckpoint


from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TSPN_trainer(pl.LightningModule):
    def __init__(self, signal_processing_modules, feature_extractor_modules, args):
        super().__init__()
        self.network = Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules, args)
        self.args = args

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        optimizer = Adam(self.parameters(), lr=self.lr)
        out = {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": ReduceLROnPlateau(optimizer),
            "monitor": self.args.monitor + '_' + self.args.task_list[-1],
            "frequency": self.args.patience//2

            },
        }
        return out




