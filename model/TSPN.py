
import torch 
import torch.nn as nn
from einops import rearrange

class SignalProcessingLayer(nn.Module):
    def __init__(self, signal_processing_modules, input_channels, output_channels,skip_connection=True):
        super(SignalProcessingLayer, self).__init__()
        self.norm = nn.InstanceNorm1d(input_channels)
        self.weight_connection = nn.Linear(input_channels, output_channels)
        self.signal_processing_modules = signal_processing_modules
        self.module_num = len(signal_processing_modules)
        if skip_connection:
            self.skip_connection = nn.Linear(input_channels, output_channels)
    def forward(self, x):
        # 信号标准化
        x = rearrange(x, 'b l c -> b c l')
        normed_x = self.norm(x)
        normed_x = rearrange(normed_x, 'b c l -> b l c')
        # 通过线性层
        x = self.weight_connection(normed_x)

        # 按模块数拆分
        splits = torch.split(x, x.size(2) // self.module_num, dim=2)

        # 通过模块计算
        outputs = []
        for module, split in zip(self.signal_processing_modules.values(), splits):
            outputs.append(module(split))
        x = torch.cat(outputs, dim=2)
        # 添加skip connection
        if hasattr(self, 'skip_connection'):
            x = x + self.skip_connection(normed_x)
        return x
    
class FeatureExtractorlayer(nn.Module):
    def __init__(self, feature_extractor_modules,in_channels=1, out_channels=1):
        super(FeatureExtractorlayer, self).__init__()
        self.weight_connection = nn.Linear(in_channels, out_channels)
        self.feature_extractor_modules = feature_extractor_modules

    def norm(self,x): # feature normalization
        mean = x.mean(dim = 0,keepdim = True)
        std = x.std(dim = 0,keepdim = True)
        out = (x-mean)/(std + 1e-10)
        return out
           
    def forward(self, x):
        x = self.weight_connection(x)
        x = rearrange(x, 'b l c -> b c l')
        outputs = []
        for module in self.feature_extractor_modules.values():
            outputs.append(module(x))
        res = torch.cat(outputs, dim=1)
        return self.norm(res)

class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes): # TODO logic
        super(Classifier, self).__init__()
        self.clf = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
            
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.clf(x)

class Transparent_Signal_Processing_Network(nn.Module):
    def __init__(self, signal_processing_modules,feature_extractor, args):
        super(Transparent_Signal_Processing_Network, self).__init__()
        self.layer_num = len(signal_processing_modules)
        self.signal_processing_modules = signal_processing_modules
        self.feature_extractor_modules = feature_extractor
        self.args = args

        self.init_signal_processing_layers()
        self.init_feature_extractor_layers()
        self.init_classifier()

    def init_signal_processing_layers(self):
        print('# build signal processing layers')
        in_channels = self.args.in_channels
        out_channels = self.args.out_channels 

        self.signal_processing_layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.signal_processing_layers.append(SignalProcessingLayer(self.signal_processing_modules[i],
                                                                       in_channels,
                                                                         out_channels,
                                                                         self.args.skip_connection).to(self.args.device))
            in_channels = out_channels 
            assert out_channels % self.signal_processing_layers[i].module_num == 0 
            out_channels = int(out_channels * self.args.scale)
        self.channel_for_feature = out_channels // self.args.scale

    def init_feature_extractor_layers(self):
        print('# build feature extractor layers')
        self.feature_extractor_layers = FeatureExtractorlayer(self.feature_extractor_modules,self.channel_for_feature,self.channel_for_feature)
        len_feature = len(self.feature_extractor_modules)
        self.channel_for_classifier = self.channel_for_feature * len_feature


    def init_classifier(self):
        print('# build classifier')
        self.clf = Classifier(self.channel_for_classifier, self.args.num_classes)

    def forward(self, x):

        for layer in self.signal_processing_layers:
            x = layer(x)
        x = self.feature_extractor_layers(x)
        x = self.clf(x)
        
        return x
if __name__ == '__main__':
    from config import args
    from config import signal_processing_modules,feature_extractor_modules
    
    net = Transparent_Signal_Processing_Network(signal_processing_modules,feature_extractor_modules, args)
    x = torch.randn(2, 4096, 2).cuda()
    y = net(x)
    print(y.shape)
    