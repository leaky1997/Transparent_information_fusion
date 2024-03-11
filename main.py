# from cgi import test
from cgi import test
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from sklearn.calibration import log
import torch

# from torch import seed
from trainer.trainer_basic import TSPN_trainer
from configs.config import args
from configs.config import signal_processing_modules,feature_extractor_modules

#data
from data.data_provider import get_data
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

seed_everything(17)
name = f'lr{args.learning_rate}_epochs{args.num_epochs}_seed{args.seed}_scale{args.scale}_l1norm{args.l1_norm}_dataset{args.data_dir[-3:]}'

print(f'Running experiment: {name}')
path = 'save/' + name
if not os.path.exists(path):
    os.makedirs(path)
# 初始化模型
model = TSPN_trainer(signal_processing_modules, feature_extractor_modules, args)
model_structure = print(model.network)

# 设置检查点回调以保存模型
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    filename='model-{epoch:02d}-{val_loss:.4f}',
    save_top_k=3,
    mode='min',
    dirpath = path
)

# 初始化训练器
trainer = pl.Trainer(callbacks=[checkpoint_callback],
                      max_epochs=args.num_epochs,
                      devices= args.gpus,
                      logger=CSVLogger(path, name = 'logs'),
                      log_every_n_steps=1,)

# dataset
train_dataloader, val_dataloader, test_dataloader = get_data(args)

# train
trainer.fit(model,train_dataloader, test_dataloader)

# # 加载最佳模型
# best_model_path = checkpoint_callback.best_model_path
# state_dict = torch.load(best_model_path)
# best_model = model.load_state_dict(state_dict['state_dict'])

# # 使用最佳模型进行测试
# trainer.test(best_model, dataloaders=test_dataloader)

