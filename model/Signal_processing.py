import torch
import torch.nn as nn
import torch.nn.functional as F

# base class for signal processing modules
class SignalProcessingBase(torch.nn.Module):
    def __init__(self, input_dim = None, output_dim = None, in_channels = None, args = None):
        super(SignalProcessingBase, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_channels = in_channels
        self.args = args

    def forward(self, x):
        raise NotImplementedError("This method should be implemented by subclass.")

class SignalProcessingModuleDict(torch.nn.ModuleDict):
    def __init__(self, module_dict):
        super(SignalProcessingModuleDict, self).__init__(module_dict)

    def forward(self, x, key):
        if key in self:
            return self[key](x)
        else:
            raise KeyError(f"No signal processing module found for key: {key}")

# subclass for FFT module

class FFTSignalProcessing(SignalProcessingBase):
    def __init__(self, input_dim, in_channels):
        # FFT 不改变通道数，只改变长度，因此 output_dim = input_dim // 2
        super(FFTSignalProcessing, self).__init__(input_dim, input_dim // 2, in_channels)

    def forward(self, x):
        # 假设 x 的形状为 [B, L, C]
        fft_result = torch.fft.rfft(x, dim=1, norm='ortho')  # 对长度L进行FFT
        return fft_result

# subclass for Hilbert module
class HilbertTransform(SignalProcessingBase):
    def __init__(self, input_dim, in_channels):
        # 希尔伯特变换不改变维度
        super(HilbertTransform, self).__init__(input_dim, input_dim, in_channels)

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
    def __init__(self, input_dim, in_channels, args):
        super(WaveFilters, self).__init__(input_dim, input_dim, in_channels,args)
        self.device = args.device
        self.to(self.device)
        
        # 初始化频率和带宽参数
        self.f_c = nn.Parameter(torch.empty(1, 1,in_channels, device=self.device))
        self.f_b = nn.Parameter(torch.empty(1, 1,in_channels, device=self.device))
        
        # 自定义参数初始化
        self.initialize_parameters()
        
        # 预生成滤波器
        self.filters = self.filter_generator(in_channels, input_dim//2 + 1)

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



if __name__ == "__main__":
    # 测试模块
    class Args:
        def __init__(self):
            self.device = 'cpu'
            self.f_c_mu = 0.1
            self.f_c_sigma = 0.01
            self.f_b_mu = 0.1
            self.f_b_sigma = 0.01

    args = Args()
    x = torch.randn(2, 1024, 3)
    fft_module = FFTSignalProcessing(1024, 3)
    print(fft_module(x).shape)
    hilbert_module = HilbertTransform(1024, 3)
    print(hilbert_module(x).shape)
    wave_filter_module = WaveFilters(1024, 3, args)
    print(wave_filter_module(x).shape)


    module_dict = {
        "$F$": FFTSignalProcessing(1024, 3),
        "$FO$": WaveFilters(1024, 3, args),
        "$HT$": HilbertTransform(1024, 3),
    }

    signal_processing_modules = SignalProcessingModuleDict(module_dict)
