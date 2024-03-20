

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

import pandas as pd

# 在循环外部创建一个空的DataFrame
results_list = []

for shot in [5,10,15,20,25]:  # Simplified loop: 1 to 10
    network = Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,args)
    model = Basic_trainer(network, args)
    model_structure = print(model.network)
    
    args.k_shot = shot  # Set the k_shot parameter
    # Set up the trainer and data loaders (ensure trainer_set is defined correctly)
    trainer, train_dataloader, val_dataloader, test_dataloader = trainer_set(args, path)

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)
    # Test the model and collect results
    result = trainer.test(model, test_dataloader)

    # Collect each result
    results_list.append(result)

# Convert the list of results to a DataFrame
result_df = pd.DataFrame.from_records(results_list)

# Save the DataFrame to a CSV file
result_df.to_csv(os.path.join(path, 'test_result.csv'), index=False)




