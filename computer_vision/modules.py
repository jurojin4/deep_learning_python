from typing import Literal, Union
from torch.nn.parameter import Parameter

import torch
import torch.nn as nn

activation_table = {"hardswish":nn.Hardswish(),
                    "leakyrelu": nn.LeakyReLU(0.1),
                    "relu":nn.ReLU(),
                    "silu":nn.SiLU()}

class ConvBlock(nn.Module):
    """
    Convolutional Block class.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: Union[int, None], batch_normalization: bool = False, activation_function: Union[Literal["hardswish", "leakyrelu", "relu", "silu"], None] = None, bias: bool = True):
        """
        Initializes the ConvBlock class.

        :param int **in_channels**: Number that represents the size of the second dimension tensor object (batch, channels, size_h, size_w).
        :param int **out_channels**: Number that represents the number of features maps, i.e the number of convolutional neurons doing a 2D convolution on the input.
        :param int **kernel_size**: Size of the convolution kernel.
        :param int **stride**: Size of the convolution stride.
        :param Union[int, None] **padding**: Size of the convolution padding.
        :param bool **batch_normalization**: Set to `False`, if `True` then a batch normalization layer is added to the convolutional block.
        :param Union[Literal["hardswish", "leakyrelu", "relu", "silu"] **activation_function**: An activation function from the table is added to the convolutional block. Set to `None`.
        :param bool **bias**: Set to `True`, adds bias to the weighted sum before applying the activation function.
        """
        assert isinstance(in_channels, int), f"in_channels has to be an {int} instance, not {type(in_channels)}."
        assert isinstance(out_channels, int), f"out_channels has to be an {int} instance, not {type(out_channels)}."
        assert isinstance(kernel_size, int), f"kernel_size has to be an {int} instance, not {type(kernel_size)}."
        assert isinstance(stride, int), f"stride has to be an {int} instance, not {type(stride)}."
        assert isinstance(padding, int) or padding is None, f"padding has to be an {int} instance or equals to {None}, not {type(padding)}."
        assert isinstance(batch_normalization, bool), f"batch_normalization has to be an {bool} instance, not {type(batch_normalization)}."
        assert activation_function in activation_table.keys() or activation_function is None, f"use_activation_function has to be an {str} instance or equals to {None}, not {type(activation_function)}."
        assert isinstance(bias, bool), f"bias has to be an {bool} instance, not {type(bias)}."
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self._conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.use_batch_normalization = batch_normalization
        if batch_normalization:
            self._batch_normalization = nn.BatchNorm2d(out_channels)
        
        self.use_activation_function = True if activation_function in activation_table else False
        if self.use_activation_function:
            self._activation_function = activation_table.get(activation_function)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the ConvBlock instance.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        if self.use_batch_normalization and self.use_activation_function:
            return self._activation_function(self._batch_normalization(self._conv2d(x)))
        elif not self.use_batch_normalization and self.use_activation_function:
            return self._activation_function(self._conv2d(x))
        elif self.use_batch_normalization and not self.use_activation_function:
            return self._batch_normalization(self._conv2d(x))
        else:
            return self._conv2d(x)
        
class ResBlock(nn.Module):
    """
    Residual Block class.
    """
    def __init__(self, in_channels: int, repetition: int, batch_normalization: bool = False, residual: bool = True):
        """
        Initializes the ResBlock class.

        :param int **in_channels**: Number that represents the size of the second dimension tensor object (batch, channels, size_h, size_w).
        :param int **repetition**: Number of repetition block.
        :param bool **batch_normalization**: Set to `False`, if `True` then a batch normalization layer is added to the convolutional block.
        """
        assert isinstance(in_channels, int), f"in_channels has to be an {int} instance, not {type(in_channels)}."
        assert isinstance(repetition, int), f"repetition has to be an {int} instance, not {type(repetition)}."
        assert isinstance(batch_normalization, bool), f"batch_normalization has to be an {bool} instance, not {type(batch_normalization)}."
        assert isinstance(residual, bool), f"residual has to be an {bool} instance, not {type(residual)}."
        super().__init__()

        self._network = nn.ModuleList()
        self.repetition = repetition
        self.residual = residual

        for _ in range(repetition):
            self._network.append(nn.Sequential(ConvBlock(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0, batch_normalization=batch_normalization, activation_function="leakyrelu"), 
                                 ConvBlock(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, batch_normalization=batch_normalization, activation_function="leakyrelu")))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the ResBlock instance.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        for block in self._network:
            if self.residual:
                x = x + block(x)
            else:
                x = block(x)
        return x
    
class RepVGGBlock(nn.Module):
    """
    RepVGGBlock base class: Rep-style convolution block.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, bias: bool = False):
        """
        Initializes the RepVGGBlock class.

        :param int **in_channels**: Number that represents the size of the second dimension tensor object (batch, channels, size_h, size_w).
        :param int **out_channels**: Number that represents the number of features maps, i.e the number of convolutional neurons doing a 2D convolution on the input.
        :param int **kernel_size**: Size of the convolution kernel. Set to `3`.
        :param int **stride**: Size of the convolution stride. Set to `1`.
        :param int **padding**: Size of the convolution padding. Set to `1`.
        """
        super().__init__()
        assert kernel_size == 3, f"kernel_size must be equal to 3, not {kernel_size}."
        assert padding == 1, f"padding must be equal to 1, not {padding}."

        padding_11 = padding - kernel_size // 2

        self._activation_function = nn.ReLU()

        self._batch_normalization = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        self._rbr_dense = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, batch_normalization=True, bias=bias)
        self._rbr_1x1 = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, batch_normalization=True, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the RepVGGBlock instance.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        if self._batch_normalization is not None:
            id_out = self._batch_normalization(x)
            return self._activation_function(self._rbr_dense(x) + self._rbr_1x1(x) + id_out)
        else:
            return self._activation_function(self._rbr_dense(x) + self._rbr_1x1(x))


class SPPFBlock(nn.Module):
    """
    Spatial Pyramid Pooling Fast block.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, activation_function: Literal["hardswish", "leakyrelu", "relu", "silu"] = "relu", bias: bool = False):
        """
        Initializes the SPPFBlock class.

        :param int **in_channels**: Number that represents the size of the second dimension tensor object (batch, channels, size_h, size_w).
        :param int **out_channels**: Number that represents the number of features maps, i.e the number of convolutional neurons doing a 2D convolution on the input.
        :param int **kernel_size**: Size of the convolution kernel. Set to `5`.
        :param bool **activation_function**: An activation function from the table is added to the convolutional block. Set to `relu`.
        :param bool **bias**: Adds bias to convolutional blocks. Set to `False`.
        """
        super().__init__()
        hidden_channels = in_channels // 2
        self._conv1 = ConvBlock(in_channels, hidden_channels, 1, 1, padding=None, batch_normalization=True, activation_function=activation_function, bias=bias)
        self._conv2 = ConvBlock(4 * hidden_channels, out_channels, 1, 1, padding=None, batch_normalization=True, activation_function=activation_function, bias=bias)
        self._maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the SPPFBlock instance.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        x = self._conv1(x)
        y1 = self._maxpool(x)
        y2 = self._maxpool(y1)
        return self._conv2(torch.cat([x, y1, y2, self._maxpool(y2)], 1))

class BottleRep(nn.Module):
    """
    BottleRep
    """
    def __init__(self, in_channels: int, out_channels: int, block: nn.Module = RepVGGBlock, weight: bool = False):
        """
        Initializes the BottleRep class.

        :param int **in_channels**: Number that represents the size of the second dimension tensor object (batch, channels, size_h, size_w).
        :param int **out_channels**: Number that represents the number of features maps, i.e the number of convolutional neurons doing a 2D convolution on the input.
        :param Module **block**: Convolutional block.
        :param bool **weight**: .
        """
        super().__init__()
        self._conv1 = block(in_channels, out_channels)
        self._conv2 = block(out_channels, out_channels)

        if in_channels != out_channels:
            self._shortcut = False
        else:
            self._shortcut = True
        if weight:
            self._alpha = Parameter(torch.ones(1))
        else:
            self._alpha = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the BottleRep instance.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        outputs = self._conv1(x)
        outputs = self._conv2(outputs)
        return outputs + self._alpha * x if self._shortcut else outputs
    
class RepBlock(nn.Module):
    """
    RepBlock is a stage block with rep-style basic block.
    """
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, block=RepVGGBlock, basic_block=RepVGGBlock):
        """
        Initializes the RepBlock class.

        :param int **in_channels**: Number that represents the size of the second dimension tensor object (batch, channels, size_h, size_w).
        :param int **out_channels**: Number that represents the number of features maps, i.e the number of convolutional neurons doing a 2D convolution on the input.
        :param Module **block**: Convolutional block.
        :param Module **basic_block**: .
        """
        super().__init__()
        if block == BottleRep:
            self._conv1 = BottleRep(in_channels, out_channels, block=basic_block, weight=True)
            n = n // 2
            self._block = nn.Sequential(*(BottleRep(out_channels, out_channels, block=basic_block, weight=False) for _ in range(n - 1))) if n > 1 else None
        else:
            self._conv1 = block(in_channels, out_channels)
            self._block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the RepBlock instance.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        x = self._conv1(x)
        if self._block is not None:
            x = self._block(x)
        return x
    
class CSPSPPFBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, e: int = 0.5, activation_function = "relu", bias: bool = False):
        """
        Initializes the CSPSPPFBlock class.

        :param int **in_channels**: Number that represents the size of the second dimension tensor object (batch, channels, size_h, size_w).
        :param int **out_channels**: Number that represents the number of features maps, i.e the number of convolutional neurons doing a 2D convolution on the input.
        :param int **kernel_size**: Size of the convolution kernel. Set to `5`.
        :param int **e**:
        :param Union[Literal["hardswish", "leakyrelu", "relu", "silu"] **activation_function**: An activation function from the table is added to the convolutional block. Set to `relu`.
        :param bool **bias**: Set to `True`, adds bias to the weighted sum before applying the activation function.
        """
        super().__init__()
        hidden_channels = int(out_channels * e)  # hidden channels
        self._conv1 = ConvBlock(in_channels, hidden_channels, 1, 1, None, batch_normalization=True, activation_function=activation_function, bias=bias)
        self._conv2 = ConvBlock(in_channels, hidden_channels, 1, 1, None, batch_normalization=True, activation_function=activation_function, bias=bias)
        self._conv3 = ConvBlock(hidden_channels, hidden_channels, 3, 1, None, batch_normalization=True, activation_function=activation_function, bias=bias)
        self._conv4 = ConvBlock(hidden_channels, hidden_channels, 1, 1, None, batch_normalization=True, activation_function=activation_function, bias=bias)

        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self._conv5 = ConvBlock(4 * hidden_channels, hidden_channels, 1, 1, None, batch_normalization=True, activation_function=activation_function, bias=bias)
        self._conv6 = ConvBlock(hidden_channels, hidden_channels, 3, 1, None, batch_normalization=True, activation_function=activation_function, bias=bias)
        self._conv7 = ConvBlock(2 * hidden_channels, out_channels, 1, 1, None, batch_normalization=True, activation_function=activation_function, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the CSPSPPFBlock instance.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        x1 = self._conv4(self._conv3(self._conv1(x)))
        y0 = self._conv2(x)
        y1 = self.m(x1)
        y2 = self.m(y1)
        y3 = self._conv6(self._conv5(torch.cat([x1, y1, y2, self.m(y2)], 1)))
        return self._conv7(torch.cat((y0, y3), dim=1))

class Transpose(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2):
        """
        Initializes the Transpose class.

        :param int **in_channels**: Number that represents the size of the second dimension tensor object (batch, channels, size_h, size_w).
        :param int **out_channels**: Number that represents the number of features maps, i.e the number of convolutional neurons doing a 2D convolution on the input.
        :param int **kernel_size**: Size of the convolution kernel. Set to `2`.
        :param int **stride**: Size of the convolution stride. Set to `2`.
        """
        super().__init__()
        self.upsample_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the RepBlock instance.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        return self.upsample_transpose(x)

class Conv_C3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding = None, groups: int = 1, activation_fn: bool = True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in k]
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups, bias=False)
        self._batch_normalization = nn.BatchNorm2d(out_channels)
        
        if activation_fn:
            self._activation_function = activation_table.get("relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._activation_function(self._batch_normalization(self._conv(x)))

class BepC3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, e = 0.5, concat: bool = True, block=RepVGGBlock):
        super().__init__()
        hidden_channels = int(out_channels * e)
        self._conv1 = Conv_C3(in_channels, hidden_channels, kernel_size=1, stride=1)
        self._conv2 = Conv_C3(in_channels, hidden_channels, kernel_size=1, stride=1)
        self._conv3 = Conv_C3(2 * hidden_channels, out_channels, kernel_size=1, stride=1)

        self._m = RepBlock(hidden_channels, hidden_channels, n, BottleRep, basic_block=block)
        self._concat = concat
        if not concat:
            self._conv3 = Conv_C3(hidden_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor):
        if self._concat:
            return self._conv3(torch.cat((self._m(self._conv1(x)), self._conv2(x)), dim=1))
        else:
            return self._conv3(self._m(self._conv1(x)))