from ast import arg
from os import pread
from altair import AllSortString
import torch
from torch import nn
from collections import OrderedDict
import yaml
from types import SimpleNamespace
import os 
import time 

from model.Signal_processing import SignalProcessingBase,\
        SignalProcessingModuleDict,\
        FFTSignalProcessing,\
        HilbertTransform,\
        WaveFilters,\
        Identity

from model.Feature_extract import FeatureExtractionBase,\
        FeatureExtractionModuleDict,\
        MeanFeature,\
        StdFeature,\
        VarFeature,\
        EntropyFeature,\
        MaxFeature,\
        MinFeature,\
        AbsMeanFeature,\
        KurtosisFeature,\
        RMSFeature,\
        CrestFactorFeature ,\
        SkewnessFeature,\
        ClearanceFactorFeature,\
        ShapeFactorFeature
ALL_SP = {
    'FFT': FFTSignalProcessing,
    'HT': HilbertTransform,
    'WF': WaveFilters,
    'I': Identity,
}
ALL_FE = {
    'Mean': MeanFeature,
    'Std': StdFeature,
    'Var': VarFeature,
    'Entropy': EntropyFeature,
    'Max': MaxFeature,
    'Min': MinFeature,
    'AbsMean': AbsMeanFeature,
    'Kurtosis': KurtosisFeature,
    'RMS': RMSFeature,
    'CrestFactor': CrestFactorFeature,
    'Skewness': SkewnessFeature,
    'ClearanceFactor': ClearanceFactorFeature,
    'ShapeFactor': ShapeFactorFeature,
}

def parse_arguments(parser):
    # 解析参数
    
    args_dir = parser.parse_args()
    # 使用参数
    yaml_dir = args_dir.config_dir
    # 读取YAML文件
    with open(yaml_dir, 'r') as f:
        config = yaml.safe_load(f)
    args = SimpleNamespace(**config['args'])

    
    # dataset = args.data_dir[-3:].replace('/','')
    time_stamp = time.strftime("%d-%H-%M-%S", time.localtime())
    name = f'model_{args.model}time{time_stamp}_lr{args.learning_rate}_epochs{args.num_epochs}_scale{args.scale}_dataset{args.dataset_task}'

    print(f'Running experiment: {name}')
    
    # if args.debug != 'True':
    path = 'save/' + f'model_{args.model}/' + name
    if not os.path.exists(path):
        os.makedirs(path)
    
    return config,args,path

def yaml_arguments(yaml_dir):
    # 读取YAML文件
    with open(yaml_dir, 'r') as f:
        config = yaml.safe_load(f)
    args = SimpleNamespace(**config['args'])

    
    # dataset = args.data_dir[-3:].replace('/','')
    time_stamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    name = f'post_time{time_stamp}_lr{args.learning_rate}_epochs{args.num_epochs}_scale{args.scale}_l1norm{args.l1_norm}_dataset{args.dataset_task}_seed{args.seed}'

    print(f'Running experiment: {name}')
    path = 'save/' + f'model_{args.model}/' + name
    if not os.path.exists(path):
        os.makedirs(path)
    
    return config,args,path

def config_network(config,args):
    
    signal_processing_modules = []
    for layer in config['signal_processing_configs'].values():
        signal_module = OrderedDict()
        for module_name in layer:
            
            module_class = ALL_SP[module_name]
            
            module_name = get_unique_module_name(signal_module.keys(), module_name)
            signal_module[module_name] = module_class(args)  # 假设所有模块的构造函数不需要参数
        signal_processing_modules.append(SignalProcessingModuleDict(signal_module))

    feature_extractor_modules = OrderedDict()
    for feature_name in config['feature_extractor_configs']:
        module_class = ALL_FE[feature_name]
        feature_extractor_modules[feature_name] = module_class()  # 假设所有模块的构造函数不需要参数
    
    # TODO logic
    
    return signal_processing_modules,feature_extractor_modules

def get_unique_module_name(existing_names, module_name):
    """
    根据已存在的模块名列表，为新模块生成一个唯一的名称。
    
    :param existing_names: 已存在的模块名称的集合或列表。
    :param module_name: 要检查的模块名称。
    :return: 唯一的模块名称。
    """
    if module_name not in existing_names:
        # 如果模块名不存在，则直接返回
        return module_name
    else:
        # 如果模块名已存在，尝试添加序号直到找到一个唯一的名字
        index = 1
        unique_name = f"{module_name}_{index}"
        while unique_name in existing_names:
            index += 1
            unique_name = f"{module_name}_{index}"
        return unique_name
# logic

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from collections import OrderedDict


# import yaml
# from types import SimpleNamespace

# config_dir = 'configs/config_basic.yaml' # 从工作目录开始的相对路径
# # 读取YAML文件
# with open(config_dir, 'r') as f:
#     config = yaml.safe_load(f)
# args = SimpleNamespace(**config['args'])



# signal_processing_modules = []
# for layer in config['signal_processing_configs'].values():
#     signal_module = OrderedDict()
#     for module_name in layer:
#         module_class = ALL_SP[module_name]
#         signal_module[module_name] = module_class(args)  # 假设所有模块的构造函数不需要参数
#     signal_processing_modules.append(SignalProcessingModuleDict(signal_module))

# feature_extractor_modules = OrderedDict()
# for feature_name in config['feature_extractor_configs']:
#     module_class = ALL_FE[feature_name]
#     feature_extractor_modules[feature_name] = module_class()  # 假设所有模块的构造函数不需要参数
