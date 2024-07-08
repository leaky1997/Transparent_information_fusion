from numpy import identity
import torch
import torch.nn as nn
import torch.nn.functional as Fs
import sys
sys.path.append('/home/user/LQ/B_Signal/Transparent_information_fusion/model')
from .utils import convlutional_operator, signal_filter_, FRE

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
        test_input = torch.randn(2, self.in_dim, self.in_channels).to(self.device) 
        output = self.forward(test_input)
        assert output.shape == (2, self.out_dim, self.out_channels), f"\
        input shape is {test_input.shape}, \n\
        Output shape is {output.shape}, \n\
        expected {(2, self.out_dim, self.out_channels)}"

class SignalProcessingBase2Arity(torch.nn.Module):
    def __init__(self, args):
        super(SignalProcessingBase2Arity, self).__init__()
        self.args = args
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels //2
        self.device = args.device
        self.to(self.device)

    def split_input(self, x):
        # 拆分输入信号
        half_channels = x.shape[-1] // 2
        x1 = x[:, :, :half_channels]
        x2 = x[:, :, half_channels:]
        return x1, x2
    
    def repeat_output(self, x):
        # 合并输出信号
        return torch.cat((x, x), dim=-1)

    def forward(self, x):
        x1, x2 = self.split_input(x)
        x = self.operation(x1, x2)
        x = self.repeat_output(x)
        return x

    def operation(self, x1, x2):
        raise NotImplementedError("This method should be implemented by subclass.")
    
    def test_forward(self):
        test_input = torch.randn(2, self.in_dim, self.in_channels).to(self.device)
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
# 1
class FFTSignalProcessing(SignalProcessingBase):
    '''
    args:
    input_dim: 输入信号的长度
    '''
    def __init__(self, args):
        # FFT 不改变通道数，只改变长度，因此 output_dim = input_dim // 2
        super(FFTSignalProcessing, self).__init__(args)
        self.name = "FFT"
    def forward(self, x):
        # 假设 x 的形状为 [B, L, C]
        fft_result = torch.fft.rfft(x, dim=1, norm='ortho')  # 对长度L进行FFT
        return fft_result

# 2 ############################################## subclass for Hilbert module###############################################  
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
    
# 3 ##############################################  WaveFilters module  ############################################## 
class WaveFilters(SignalProcessingBase):
    def __init__(self, args):
        super(WaveFilters, self).__init__(args)

        self.name = "WF"
        self.device = args.device
        self.to(self.device)
        in_channels = args.scale # large enough to avoid setting

        
        # 初始化频率和带宽参数
        self.f_c = nn.Parameter(torch.empty(1, 1,in_channels, device=self.device))
        self.f_b = nn.Parameter(torch.empty(1, 1,in_channels, device=self.device))
        
        # 自定义参数初始化
        self.initialize_parameters()
        
        # 预生成滤波器

    def initialize_parameters(self):
        # 根据提供的参数初始化f_c和f_b
        nn.init.normal_(self.f_c, mean=self.args.f_c_mu, std=self.args.f_c_sigma)
        nn.init.normal_(self.f_b, mean=self.args.f_b_mu, std=self.args.f_b_sigma)

    # TODO add other filter
        
    def filter_generator(self, in_channels, freq_length): 
        omega = torch.linspace(0, 0.5, freq_length, device=self.device).view(1, -1, 1)
        
        self.omega = omega.reshape(1, 1, freq_length).repeat([1, in_channels, 1])
        
        filters = torch.exp(-((omega - self.f_c) / (2 * self.f_b)) ** 2)
        return filters

    def forward(self, x): 
        in_dim, in_channels = x.shape[-2],x.shape[-1]
        freq = torch.fft.rfft(x, dim=1, norm='ortho')
        
        self.filters = self.filter_generator(in_channels, in_dim//2 + 1)
        # 应用滤波器到所有通道
        filtered_freq = freq * self.filters[:,:,:in_channels] # B,L//2,C * 1,L//2,c
        
        x_hat = torch.fft.irfft(filtered_freq, dim=1, norm='ortho')
        return x_hat.real


# 4 ############################################# Identity module #######################################
class Identity(SignalProcessingBase):
    def __init__(self, args):
        super(Identity, self).__init__(args)
        self.name = "I"
    def forward(self, x):
        return x


#%% 5 

class Morlet(SignalProcessingBase):
    def __init__(self, args):
        super(Morlet, self).__init__(args)
        self.name = "Morlet"
        self.convolution_operator = convlutional_operator('Morlet', in_channels=args.scale, device=args.device)

    def forward(self, x):
        x_transformed = self.convolution_operator(x)
        return x_transformed
    #%% Laplace
class Laplace(SignalProcessingBase):
    def __init__(self, args):
        super(Laplace, self).__init__(args)
        self.name = "Laplace"
        self.convolution_operator = convlutional_operator('Laplace', in_channels=args.scale, device=args.device)

    def forward(self, x):
        x_transformed = self.convolution_operator(x)
        return x_transformed
    #%% Order1MAFilter
class Order1MAFilter(SignalProcessingBase):
    def __init__(self, args):
        super(Order1MAFilter, self).__init__(args)
        self.name = "order1_MA"
        self.filter_operator = signal_filter_('order1_MA', in_channels=args.scale, device=args.device)

    def forward(self, x):
        x_transformed = self.filter_operator(x)
        return x_transformed
    #%% Order2MAFilter
class Order2MAFilter(SignalProcessingBase):
    def __init__(self, args):
        super(Order2MAFilter, self).__init__(args)
        self.name = "order2_MA"
        self.filter_operator = signal_filter_('order2_MA', in_channels=args.scale, device=args.device)

    def forward(self, x):
        x_transformed = self.filter_operator(x)
        return x_transformed

class Order1DFFilter(SignalProcessingBase):
    def __init__(self, args):
        super(Order1DFFilter, self).__init__(args)
        self.name = "order1_DF"
        self.filter_operator = signal_filter_('order1_DF', in_channels=args.scale, device=args.device)

    def forward(self, x):
        x_transformed = self.filter_operator(x)
        return x_transformed

class Order2DFFilter(SignalProcessingBase):
    def __init__(self, args):
        super(Order2DFFilter, self).__init__(args)
        self.name = "order2_DF"
        self.filter_operator = signal_filter_('order2_DF', in_channels=args.scale, device=args.device)

    def forward(self, x):
        x_transformed = self.filter_operator(x)
        return x_transformed

class LogOperation(SignalProcessingBase):
    def __init__(self, args):
        super(LogOperation, self).__init__(args)
        self.name = "Log"

    def forward(self, x):
        return torch.log(x)
class SquOperation(SignalProcessingBase):
    def __init__(self, args):
        super(SquOperation, self).__init__(args)
        self.name = "Squ"

    def forward(self, x):
        return x ** 2

class SinOperation(SignalProcessingBase):
    def __init__(self, args):
        super(SinOperation, self).__init__(args)
        self.name = "sin"
        self.fre = FRE # TODO learbable

    def forward(self, x):
        return torch.sin(self.fre * x)

###############################################2 arity###################################################
class AddOperation(SignalProcessingBase2Arity):
    def __init__(self, args):
        super(AddOperation, self).__init__(args)
        self.name = "add"

    def operation(self, x1, x2):
        return x1 + x2
    
class MulOperation(SignalProcessingBase2Arity):
    def __init__(self, args):
        super(MulOperation, self).__init__(args)
        self.name = "mul"

    def operation(self, x1, x2):
        return x1 * x2
    
class DivOperation(SignalProcessingBase2Arity):
    def __init__(self, args):
        super(DivOperation, self).__init__(args)
        self.name = "div"

    def operation(self, x1, x2):
        return x1 / (x2 + 1e-8)


if __name__ == "__main__":
    # 测试模块
    import copy
    class SignalProcessingArgs:
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

    args = SignalProcessingArgs()
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
# for test

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