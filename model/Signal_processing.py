from numpy import identity
import torch
import torch.nn as nn
import torch.nn.functional as F

# base class for signal processing modules
class SignalProcessingBase(torch.nn.Module):
    def __init__(self, args):
        super(SignalProcessingBase, self).__init__()
        self.args = args
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.device = args.device
        self.to(self.device)


    def forward(self, x):
        # x should be B,L,C first
        raise NotImplementedError("This method should be implemented by subclass.")
    
    def test_forward(self):
        test_input = torch.randn(2, self.in_dim, self.in_channels)
        output = self.forward(test_input)
        assert output.shape == (2, self.out_dim, self.out_channels), f"\
        input shape is {test_input.shape}, \n\
        Output shape is {output.shape}, \n\
        expected {(2, self.out_dim, self.out_channels)}"

class SignalProcessingModuleDict(torch.nn.ModuleDict):
    def __init__(self, module_dict):
        super(SignalProcessingModuleDict, self).__init__(module_dict)

    def forward(self, x, key):
        if key in self:
            return self[key](x)
        else:
            raise KeyError(f"No signal processing module found for key: {key}")
        
    def test_forward(self):
        for key in self.keys():
            self[key].test_forward()

# TODO
# SignalProcessingModuleDict_2_arity
# subclass for FFT module

class FFTSignalProcessing(SignalProcessingBase):
    '''
    args:
    input_dim: 输入信号的长度
    '''
    def __init__(self, args):
        # FFT 不改变通道数，只改变长度，因此 output_dim = input_dim // 2
        super(FFTSignalProcessing, self).__init__(args)
        self.name = "$FFT$"
    def forward(self, x):
        # 假设 x 的形状为 [B, L, C]
        fft_result = torch.fft.rfft(x, dim=1, norm='ortho')  # 对长度L进行FFT
        return fft_result

# subclass for Hilbert module
class HilbertTransform(SignalProcessingBase):
    def __init__(self, args):
        # 希尔伯特变换不改变维度
        super(HilbertTransform, self).__init__(args)
        self.name = "HT"
    def forward(self, x):
        N = x.shape[-1]
        Xf = torch.fft.fft(x, dim=1)  # 对最后一个维度执行FFT
        if (N % 2 == 0):
            Xf[..., 1:N // 2] *= 2
            Xf[..., N // 2 + 1:] = 0
        else:
            Xf[..., 1:(N + 1) // 2] *= 2
            Xf[..., (N + 1) // 2:] = 0
        return torch.fft.ifft(Xf, dim=1).abs()
    
# WaveFilters module
class WaveFilters(SignalProcessingBase):
    def __init__(self, args):
        super(WaveFilters, self).__init__(args)

        self.name = "$WF$"
        self.device = args.device
        self.to(self.device)
        in_channels = args.in_channels
        in_dim = args.in_dim
        
        # 初始化频率和带宽参数
        self.f_c = nn.Parameter(torch.empty(1, 1,in_channels, device=self.device))
        self.f_b = nn.Parameter(torch.empty(1, 1,in_channels, device=self.device))
        
        # 自定义参数初始化
        self.initialize_parameters()
        
        # 预生成滤波器
        self.filters = self.filter_generator(in_channels, in_dim//2 + 1)

    def initialize_parameters(self):
        # 根据提供的参数初始化f_c和f_b
        nn.init.normal_(self.f_c, mean=self.args.f_c_mu, std=self.args.f_c_sigma)
        nn.init.normal_(self.f_b, mean=self.args.f_b_mu, std=self.args.f_b_sigma)

    # TODO add other filter
        
    def filter_generator(self, in_channels, freq_length): 
        omega = torch.linspace(0, 0.5, freq_length, device=self.device).view(1, -1, 1)
        filters = torch.exp(-((omega - self.f_c) / (2 * self.f_b)) ** 2)
        return filters

    def forward(self, x): 
        freq = torch.fft.rfft(x, dim=1, norm='ortho')
        
        # 应用滤波器到所有通道
        filtered_freq = freq * self.filters # B,L//2,C * 1,L//2,c
        
        x_hat = torch.fft.irfft(filtered_freq, dim=1, norm='ortho', n=self.input_dim)
        return x_hat.real

class Identity(SignalProcessingBase):
    def __init__(self, args):
        super(Identity, self).__init__(args)
        self.name = "$I$"
    def forward(self, x):
        return x

if __name__ == "__main__":
    # 测试模块
    import copy
    class Args:
        def __init__(self):
            self.device = 'cpu'
            self.f_c_mu = 0.1
            self.f_c_sigma = 0.01
            self.f_b_mu = 0.1
            self.f_b_sigma = 0.01
            self.in_dim = 1024
            self.out_dim = 1024
            self.in_channels = 10
            self.out_channels = 10

    args = Args()
    argsfft = copy.deepcopy(args)

    argsfft.out_dim = argsfft.in_dim // 2 + 1

    fft_module = FFTSignalProcessing(argsfft)
  
    hilbert_module = HilbertTransform(args)

    wave_filter_module = WaveFilters(args)

    identity_module = Identity(args)

    module_dict = {
        "$F$": fft_module,
        "$FO$": hilbert_module,
        "$HT$": wave_filter_module,
        "$I$": identity_module,
    }

    signal_processing_modules = SignalProcessingModuleDict(module_dict)

    from collections import OrderedDict
    import pandas as pd
    import copy
    class Args:
        def __init__(self):
            self.device = 'cpu'
            self.f_c_mu = 0.1
            self.f_c_sigma = 0.01
            self.f_b_mu = 0.1
            self.f_b_sigma = 0.01
            self.in_dim = 1024
            self.out_dim = 1024
            self.in_channels = 10
            self.out_channels = 10
        def save_to_csv(self, filename):
            df = pd.DataFrame.from_records([self.__dict__])
            df.to_csv(filename, index=False)

    args = Args()
    argsfft = copy.deepcopy(args)

    argsfft.out_dim = argsfft.in_dim // 2 + 1

    signal_module_1 = {
            "$HT$": HilbertTransform(args),
            "$WF$": WaveFilters(args),
            "$I$": Identity(args),
        }
    ordered_module_dict = OrderedDict(signal_module_1)
    signal_processing_modules = SignalProcessingModuleDict(signal_module_1)

    signal_module_2 = {
            "$HT$": HilbertTransform(args),
            "$WF$": WaveFilters(args),
            "$I$": Identity(args),
        }
    ordered_module_dict = OrderedDict(signal_module_2)
    signal_processing_modules_2 = SignalProcessingModuleDict(signal_module_2)