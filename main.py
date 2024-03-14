

############# config##########
import argparse
from model.TSPN import Transparent_Signal_Processing_Network
from trainer.trainer_basic import Basic_trainer
from trainer.trainer_set import trainer_set

import torch
from pytorch_lightning import seed_everything
from configs.config import parse_arguments,config_network
import os
import pandas as pd


# 创建解析器
parser = argparse.ArgumentParser(description='TSPN')

# 添加参数
parser.add_argument('--config_dir', type=str, default='configs/config_basic.yaml',
                    help='The directory of the configuration file')
configs,args,path = parse_arguments(parser)
seed_everything(args.seed)    

# 初始化模型
signal_processing_modules, feature_extractor_modules = config_network(configs,args)
network = Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,args)

#model trainer #
model = Basic_trainer(network, args)
model_structure = print(model.network)
trainer,train_dataloader, val_dataloader, test_dataloader = trainer_set(args,path)

# train
trainer.fit(model,train_dataloader, test_dataloader)
result = trainer.test(model,test_dataloader)

# 保存结果
result_df = pd.DataFrame(result)
result_df.to_csv(os.path.join(path, 'test_result.csv'), index=False)




