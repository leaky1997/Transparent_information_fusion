from .Resnet import ResNet

class Sincnet(ResNet):
    def __init__(self, block, layers, in_channel=1, num_class=4, zero_init_residual=False):
        super().__init__(block, layers, in_channel, num_class, zero_init_residual, first_kernel='Sinc')
        
class Sinc_net_m(ResNet):
    def __init__(self, block, layers, in_channel=1, num_class=4, zero_init_residual=False):
        super().__init__(block, layers, in_channel, num_class, zero_init_residual, first_kernel='Sinc_m')