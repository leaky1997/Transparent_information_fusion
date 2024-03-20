from pytorch_lightning.callbacks import Callback
import torch
import copy

def l1_reg(param):
    return torch.sum(torch.abs(param))

class ModelParametersLoggingCallback(Callback):
    def __init__(self):
        super().__init__()
        # 初始化一个用于保存参数历史的字典
        self.params_history = []
        self.learnable_params = []

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        # 在每个训练周期结束时调用
        epoch_params = {}
        for name, params in pl_module.named_parameters():
            # 对于模型中的每个参数，保存其当前的值（注意：这里使用.clone().detach()来获取参数值的副本）
            epoch_params[name] = params.clone().detach().cpu().numpy()  # 确保将参数移动到CPU上并转换为NumPy数组
        # 将当前周期的参数快照添加到历史记录中
        self.params_history.append(epoch_params)

    def on_train_end(self, trainer, pl_module):
        # 训练结束后可以做一些清理工作或者保存历史记录到文件
        # 例如：保存参数历史到文件（根据需要实现）
        pass