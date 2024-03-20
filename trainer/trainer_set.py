from pytorch_lightning.loggers import CSVLogger

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelPruning

import pytorch_lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from torch.utils.tensorboard.writer import SummaryWriter   


############### data ###############
from data.data_provider import get_data

def trainer_set(args,path):
    # 设置检查点回调以保存模型
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
        dirpath = path
    )
    def compute_amount(epoch):
        # the sum of all returned values need to be smaller than 1
        if epoch == args.num_epochs//4:
            return args.pruning[0]
        elif epoch == args.num_epochs//2:
            return args.pruning[1]
        elif 3 * args.num_epochs//4 < epoch:
            return args.pruning[2]
        
    if isinstance(args.pruning, (int, float)):
        prune_callback = ModelPruning("l1_unstructured",
                                      parameter_names = ['weight'],
                                      amount=args.pruning)
    elif isinstance(args.pruning, list):
        prune_callback = ModelPruning("l1_unstructured",
                                      parameter_names = ['weight'],
                                      amount = compute_amount)
    else:
        prune_callback = None
    
    # 初始化训练器
    trainer = pl.Trainer(callbacks=[checkpoint_callback] if prune_callback is None else [checkpoint_callback,prune_callback],
                        max_epochs=args.num_epochs,
                        devices= args.gpus,
                        logger = [CSVLogger(path, name="logs"),TensorBoardLogger(path, name="logs")],
                        log_every_n_steps=1,)
    
    train_dataloader, val_dataloader, test_dataloader = get_data(args)
    return trainer,train_dataloader, val_dataloader, test_dataloader