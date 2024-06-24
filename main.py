

############# config##########
import argparse
from model.TSPN import Transparent_Signal_Processing_Network \
    , Transparent_Signal_Processing_KAN
from model.NNSPN import NN_Signal_Processing_Network
from trainer.trainer_basic import Basic_plmodel
from trainer.trainer_set import trainer_set
from trainer.utils import load_best_model_checkpoint

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


MODEL_DICT = {
    'TSPN': lambda args: Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,args),
    'TKAN': lambda args: Transparent_Signal_Processing_KAN(signal_processing_modules, feature_extractor_modules,args),
    'NNSPN': lambda args: NN_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,args),
}

model_plain = MODEL_DICT[args.model](args)

# network = Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,args)
#model trainer #
model = Basic_plmodel(model_plain, args)
model_structure = print(model.network)
trainer,train_dataloader, val_dataloader, test_dataloader = trainer_set(args,path)

# train
trainer.fit(model,train_dataloader, val_dataloader) # TODO load best checkpoint

model = load_best_model_checkpoint(model,trainer)

result = trainer.test(model,test_dataloader)

# 保存结果
result_df = pd.DataFrame(result)
result_df.to_csv(os.path.join(path, 'test_result.csv'), index=False)




