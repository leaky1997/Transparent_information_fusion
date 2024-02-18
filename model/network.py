from os import pread
import torch
from torch import nn

from Signal_processing import SignalProcessingBase,\
        SignalProcessingModuleDict,\
        FFTSignalProcessing,\
        HilbertTransform,\
        WaveFilters

from Feature_extract import FeatureExtractionBase,\
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

# logic

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
signal_processing_config = {
'channel_1'：{
    'layer1': [FFTSignalProcessing(input_dim, in_channels), 
              HilbertTransform(input_dim, in_channels)],
    'layer2': [HilbertTransform(input_dim, in_channels), 
              WaveFilters(input_dim, in_channels, args)],
    ...
}
‘channel_2’：{
...
}
}
'''

class SignalProcessingNetwork(nn.Module):
    def __init__(self, signal_processing_configs, feature_extractor_configs, num_classes):
        super(SignalProcessingNetwork, self).__init__()
        self.signal_processing_layers = nn.ModuleDict()
        self.feature_extractors = nn.ModuleDict()
        self.num_classes = num_classes

        # 构建信号处理层
        for channel, layers in signal_processing_configs.items(): # channels = ["channel_1", "channel_2"]
            for layer_name, modules in layers.items(): # layer_name = "layer_1", modules = [module_1, module_2]
                for i, module in enumerate(modules): # module_1, module_2
                    # 为每个模块添加一个InstanceNorm层
                    self.signal_processing_layers[f"{channel}_{layer_name}_instancenorm_{i}"] = nn.InstanceNorm1d(module.in_channels)
                    
                    # 确定线性层的输入输出维度
                    input_dim = module.input_dim # 实际上是信号输入的维度
                    output_dim = module.output_dim
                    # 为每个模块添加一个Linear层
                    self.signal_processing_layers[f"{channel}_{layer_name}_linear_{i}"] = nn.Linear(input_dim, output_dim)
                    
                    # 添加信号处理模块
                    self.signal_processing_layers[f"{channel}_{layer_name}_module_{i}"] = module

                    # 添加spik_connection
                    self.signal_processing_layers[f"{channel}_{layer_name}_spike_connection_{i}"] = nn.Linear(input_dim, output_dim)
                    
        # TODO: 根据 feature_extractor_configs 构建特征提取层
        
    def forward_sp_per_channel(self, x, channel):
        
        channel_x = x[:, channel, :]
        prev_layer_outputs = {"input": channel_x}

        # 遍历所有信号处理层
        for layer_name in self.signal_processing_configs[channel].keys():
            # 遍历层内的所有模块
            for i in range(len(self.signal_processing_configs[channel][layer_name])):

                # 应用InstanceNorm
                x = self.signal_processing_layers[f"{channel}_{layer_name}_instancenorm_{i}"](x)
                
                # 应用Linear变换
                x = self.signal_processing_layers[f"{channel}_{layer_name}_linear_{i}"](x)
                
                # 应用信号处理模块
                x = self.signal_processing_layers[f"{channel}_{layer_name}_module_{i}"](x)
        
        # TODO: 应用特征提取层并进行分类
        # 注意：需要根据特征提取配置和实际需求调整
        
        return x

    def forward(self, x):
        return x








    def forward(self, x):
        # 信号处理层
        for name, module in self.signal_processing_modules.items():
            x = module(x)
            x = self.fc_layers[name](x)

        # 特征提取
        x = self.feature_extractors(x)

        # 分类
        x = self.classifier(x)
        return x
    def parse_network(self):
        pass



# 创建信号处理模块和特征提取器
signal_processing_modules = SignalProcessingModuleDict(module_dict)
feature_extractors = FeatureExtractionModuleDict(feature_extractors)

# 创建神经网络
num_classes = 10  # 假设有10个类别
model = SignalProcessingNetwork(signal_processing_modules, feature_extractors, num_classes)