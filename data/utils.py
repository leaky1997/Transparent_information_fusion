import torch

def wgn2(x, snr):
    "加随机噪声"
    snr = 10**(snr/10.0)
    xpower = torch.sum(x**2)/(x.size(0)*x.size(1)*x.size(2))
    npower = xpower / snr
    return torch.rand(x.size()).cuda() * torch.sqrt(npower)

class AddNoiseTransform:
    def __init__(self, snr):
        self.snr = snr

    def __call__(self, x):
        self.snr = 10**(self.snr/10.0)
        xpower = torch.sum(x**2)/(x.size(0)*x.size(1))
        npower = xpower / self.snr
        return torch.rand(x.size()) * torch.sqrt(npower) + x