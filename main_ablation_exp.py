

############# config##########
import argparse
from model.TSPN import Transparent_Signal_Processing_Network, Transparent_Signal_Processing_KAN
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
# 初始化模型
signal_processing_modules, feature_extractor_modules = config_network(configs,args)


MODEL_DICT = {
    'TSPN': lambda args: Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,args),
    'TKAN': lambda args: Transparent_Signal_Processing_KAN(signal_processing_modules, feature_extractor_modules,args),
    'NNSPN': lambda args: NN_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,args),
}



#model trainer #
all_results_df = pd.DataFrame()
for lr_weight in [0.01,0.001, 0.0001, 0.00001]:
    for lr_parameter in [0.01,0.001, 0.0001, 0.00001]:
        print(f'lr_weight: {lr_weight}, lr_parameter: {lr_parameter}')
        args.learning_rate = lr_weight
        args.learnable_parameter_learning_rate = lr_parameter
        network = MODEL_DICT[args.model](args)
        model = Basic_plmodel(network, args)
        trainer, train_dataloader, val_dataloader, test_dataloader = trainer_set(args, path)

        # 训练
        trainer.fit(model, train_dataloader, val_dataloader)  # TODO: 加载最佳检查点
        model = load_best_model_checkpoint(model,trainer)
        # 测试
        result = trainer.test(model, test_dataloader)

        del model
        del network
        del trainer
        del train_dataloader

        # 将测试结果、学习率和参数学习率添加到字典中
        result_dict = {'learning_rate_weight': lr_weight, 'learnable_parameter_learning_rate': lr_parameter, **result[0]}

        # 将本次迭代的结果添加到DataFrame中
        if all_results_df.empty:
            all_results_df = pd.DataFrame([result_dict])
        else:
            new_row = pd.DataFrame([result_dict])
            all_results_df = pd.concat([all_results_df, new_row], ignore_index=True)

# 保存整个DataFrame到CSV文件
results_file_path = os.path.join(path, 'test_results.csv')
all_results_df.to_csv(results_file_path, index=False)




