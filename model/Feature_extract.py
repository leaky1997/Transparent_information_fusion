import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractionBase(torch.nn.Module):
    def __init__(self, method_name):
        super(FeatureExtractionBase, self).__init__()
        self.method_name = method_name

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