import torch.nn as nn
from torch import Tensor


class SwitchableGroupNorm(nn.GroupNorm):
    def __init__(self, *args, norm_layer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_layer = norm_layer

    def set_norm_layer(self, norm_layer: nn.GroupNorm):
        self.norm_layer = norm_layer

    def copy_weights(self):
        if self.norm_layer is not None:
            self.norm_layer.weight.data.copy_(self.weight.data)
            self.norm_layer.bias.data.copy_(self.bias.data)

    def forward(self, x: Tensor):
        if self.norm_layer is None:
            return super().forward(x)
        return self.norm_layer(x)


class SwitchableLayerNorm(nn.LayerNorm):
    def __init__(self, *args, norm_layer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_layer = norm_layer

    def set_norm_layer(self, norm_layer: nn.LayerNorm):
        self.norm_layer = norm_layer

    def copy_weights(self):
        if self.norm_layer is not None:
            self.norm_layer.weight.data.copy_(self.weight.data)
            self.norm_layer.bias.data.copy_(self.bias.data)

    def forward(self, x: Tensor):
        if self.norm_layer is None:
            return super().forward(x)
        return self.norm_layer(x)


class SwitchableConv2d(nn.Conv2d):
    """
    A switchable 2D convolutional layer that can dynamically switch between 
    different convolutional layers.

    Args:
        *args: Variable length argument list for nn.Conv2d.
        conv_layer (nn.Conv2d, optional): An optional convolutional layer to switch to.
        **kwargs: Arbitrary keyword arguments for nn.Conv2d.

    Attributes:
        conv_layer (nn.Conv2d): The convolutional layer to switch to.

    Methods:
        set_conv_layer(conv_layer: nn.Conv2d):
            Sets the convolutional layer to switch to.
        
        copy_weights():
            Copies the weights and biases from the current layer to the switchable layer.
        
        forward(x: Tensor) -> Tensor:
            Performs the forward pass. If a switchable layer is set, it uses that layer;
            otherwise, it uses the default nn.Conv2d forward pass.
    """
    def __init__(self, *args, conv_layer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_layer = conv_layer

    def set_conv_layer(self, conv_layer: nn.Conv2d):
        self.conv_layer = conv_layer

    def copy_weights(self):
        if self.conv_layer is not None:
            self.conv_layer.weight.data.copy_(self.weight.data)
            if hasattr(self, 'bias') and self.bias is not None:
                self.conv_layer.bias.data.copy_(self.bias.data)

    def forward(self, x: Tensor):
        if self.conv_layer is None:
            return super().forward(x)
        return self.conv_layer(x)
