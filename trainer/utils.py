import torch

def l1_reg(param):
    return torch.sum(torch.abs(param))

