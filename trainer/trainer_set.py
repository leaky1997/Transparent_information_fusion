from pytorch_lightning.loggers import CSVLogger

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelPruning

import pytorch_lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from torch.utils.tensorboard import SummaryWriter   
# trainer = Trainer(logger=logger)
# from utils import compute_amount

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
    if isinstance(args.pruning, (int, float)):
        
        prune_callback = ModelPruning("l1_unstructured", amount=args.pruning)
    elif isinstance(args.pruning, list):
        
        def compute_amount(epoch):
            # the sum of all returned values need to be smaller than 1
            if epoch == args.num_epochs//4:
                return args.pruning[0]
            elif epoch == args.num_epochs//2:
                return args.pruning[1]
            elif 3 * args.num_epochs//4 < epoch :
                return args.pruning[2]
            
        prune_callback = ModelPruning("l1_unstructured", amount=compute_amount)
    elif args.pruning is None:
        pass
    
    callback_list = [checkpoint_callback,prune_callback] if args.pruning is not None else [checkpoint_callback]
    log_list = [CSVLogger(path, name="logs"),TensorBoardLogger(path, name="logs")]
    # 初始化训练器
    trainer = pl.Trainer(callbacks=callback_list,
                        max_epochs=args.num_epochs,
                        devices= args.gpus,
                        logger = log_list,
                        log_every_n_steps=1,)
    train_dataloader, val_dataloader, test_dataloader = get_data(args)
    return trainer,train_dataloader, val_dataloader, test_dataloader