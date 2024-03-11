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

    
    dataset = args.data_dir[-3:].replace('/','')
    time_stamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    name = f'time{time_stamp}_model{args.model}_lr{args.learning_rate}_epochs{args.num_epochs}_scale{args.scale}_l1norm{args.l1_norm}_dataset{dataset}_seed{args.seed}'

    print(f'Running experiment: {name}')
    path = 'save/' + name
    if not os.path.exists(path):
        os.makedirs(path)
    
    return config,args, path

def config_network(config,args):
    
    signal_processing_modules = []
    for layer in config['signal_processing_configs'].values():
        signal_module = OrderedDict()
        for module_name in layer:
            module_class = ALL_SP[module_name]
            signal_module[module_name] = module_class(args)  # 假设所有模块的构造函数不需要参数
        signal_processing_modules.append(SignalProcessingModuleDict(signal_module))

    feature_extractor_modules = OrderedDict()
    for feature_name in config['feature_extractor_configs']:
        module_class = ALL_FE[feature_name]
        feature_extractor_modules[feature_name] = module_class()  # 假设所有模块的构造函数不需要参数
    
    # TODO logic
    
    return signal_processing_modules,feature_extractor_modules
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
