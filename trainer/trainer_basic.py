import torch
import torch.nn as nn
import pytorch_lightning as pl
from model.TSPN import Transparent_Signal_Processing_Network
# from config import args
# from config import signal_processing_modules,feature_extractor_modules
from pytorch_lightning.callbacks import ModelCheckpoint


from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
from .utils import l1_reg


class TSPN_trainer(pl.LightningModule):
    def __init__(self, signal_processing_modules, feature_extractor_modules, args):
        super().__init__()
        self.network = Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules, args)
        self.args = args
        self.loss = nn.CrossEntropyLoss()
        self.acc_val = torchmetrics.Accuracy(task = "multiclass",num_classes = args.num_classes)
        self.acc_train = torchmetrics.Accuracy(task = "multiclass",num_classes = args.num_classes)
        self.acc_test = torchmetrics.Accuracy(task = "multiclass",num_classes = args.num_classes)
        
        self.save_hyperparameters()
        
        
        print('### network:\n',self.network)
    def forward(self, x):
        
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.long())
        self.acc_train(y_hat, y.long())
        
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True,reduce_fx=torch.mean)
        self.log('train_acc', self.acc_train,  on_epoch=True, prog_bar=True, logger=True,reduce_fx=torch.mean)
        
        regularization_loss = 0
        if self.args.l1_norm > 0: # l1 regularization
            for param in self.network.parameters():
                regularization_loss += l1_reg(param = param) 
            
            loss += self.args.l1_norm * regularization_loss          
            self.log('l1_loss_', regularization_loss,prog_bar =True,reduce_fx=torch.mean)    
        
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss(y_hat, y.long())
        self.acc_val(y_hat, y.long())
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True,reduce_fx=torch.mean)
        self.log('val_acc', self.acc_val, on_epoch=True, prog_bar=True, logger=True,reduce_fx=torch.mean)
        return val_loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss(y_hat, y.long())
        self.log('test_loss', test_loss,  on_epoch=True, prog_bar=True, logger=True,reduce_fx=torch.mean)
        self.acc_test(y_hat, y.long())
        self.log('test_acc', self.acc_test,  on_epoch=True, prog_bar=True, logger=True,reduce_fx=torch.mean)
        return test_loss
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        optimizer = Adam(self.parameters(), lr=self.args.learning_rate, weight_decay = self.args.weight_decay)
        out = {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": ReduceLROnPlateau(optimizer),
            "monitor": self.args.monitor ,
            "frequency": self.args.patience//2

            },
        }
        return out




