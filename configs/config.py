from ast import arg
from os import pread
from altair import AllSortString
import torch
from torch import nn

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
# logic

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


import yaml
from types import SimpleNamespace

config_dir = 'configs/config_basic.yaml' # 从工作目录开始的相对路径
# 读取YAML文件
with open(config_dir, 'r') as f:
    config = yaml.safe_load(f)
args = SimpleNamespace(**config['args'])



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
