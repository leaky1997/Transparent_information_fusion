############# learning ############
from cgi import test
from logging import config
from pytorch_lightning import seed_everything

from sklearn.calibration import log
import torch
############# config##########
import argparse
from trainer.trainer_basic import TSPN_trainer
from trainer.trainer_set import trainer_set
# from configs.config import args
# from configs.config import signal_processing_modules,feature_extractor_modules
from configs.config import parse_arguments,config_network
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' for test ##########################

# 创建解析器
parser = argparse.ArgumentParser(description='TSPN')
# 添加参数
parser.add_argument('--config_dir', type=str, default='configs/config_basic.yaml',
                    help='The directory of the configuration file')
configs,args,path = parse_arguments(parser)
seed_everything(args.seed)    
# 初始化模型
signal_processing_modules, feature_extractor_modules = config_network(configs,args)
############## model train ########## 
model = TSPN_trainer(signal_processing_modules, feature_extractor_modules, args)
model_structure = print(model.network)
trainer,train_dataloader, val_dataloader, test_dataloader = trainer_set(args,path)
# train
trainer.fit(model,train_dataloader, test_dataloader)



