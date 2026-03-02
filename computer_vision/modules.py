import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Convolutional Block class.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, batch_normalization: bool = False, activation_function: bool = True, bias: bool = True):
        """
        Initializes the ConvBlock class.

        :param int **in_channels**: Number that represents the size of the first dimension for a 3D object in torch.
        :param int **out_channels**: Number that represents the number of features maps, i.e the number of convolutional neurons doing a 2D convolution on the input.
        :param int **kernel_size**: Size of the convolution kernel.
        :param int **stride**: Size of the convolution stride.
        :param int **padding**: Size of the convolution padding.
        :param bool **batch_normalization**: Set to `False`, if `True` then a batch normalization layer is added to the convolutional block.
        :param bool **activation_function**: Set to `True`, if `True` then an activation layer is added to the convolutional block.
        :param bool **bias**: Set to `True`, adds bias to the weighted sum before applying the activation function.
        """
        assert isinstance(in_channels, int), f"in_channels has to be an {int} instance, not {type(in_channels)}."
        assert isinstance(out_channels, int), f"out_channels has to be an {int} instance, not {type(out_channels)}."
        assert isinstance(kernel_size, int), f"kernel_size has to be an {int} instance, not {type(kernel_size)}."
        assert isinstance(stride, int), f"stride has to be an {int} instance, not {type(stride)}."
        assert isinstance(padding, int), f"padding has to be an {int} instance, not {type(padding)}."
        assert isinstance(batch_normalization, bool), f"batch_normalization has to be an {bool} instance, not {type(batch_normalization)}."
        assert isinstance(activation_function, bool), f"use_activation_function has to be an {bool} instance, not {type(activation_function)}."
        assert isinstance(bias, bool), f"bias has to be an {bool} instance, not {type(bias)}."
        super().__init__()

        self._conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.use_batch_normalization = batch_normalization
        if batch_normalization:
            self._batch_normalization = nn.BatchNorm2d(out_channels)
        
        self.use_activation_function = activation_function
        self._activation_function = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the ConvBlock instance.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        if self.use_batch_normalization:
            return self._activation_function(self._batch_normalization(self._conv2d(x)))
        elif self.use_activation_function:
            return self._activation_function(self._conv2d(x))
        else:
            return self._conv2d(x)
        
class ResBlock(nn.Module):
    """
    Residual Block class.
    """
    def __init__(self, in_channels: int, repetition: int, batch_normalization: bool = False, residual: bool = True):
        """
        Initializes the ResBlock class.

        :param int **in_channels**: Number that represents the size of the first dimension for a 3D object in torch.
        :param int **repetition**: Number of block repetition.
        :param bool **batch_normalization**: Set to `False`, if `True` then a batch normalization layer is added to the convolutional block.
        """
        assert isinstance(in_channels, int), f"in_channels has to be an {int} instance, not {type(in_channels)}."
        assert isinstance(repetition, int), f"repetition has to be an {int} instance, not {type(repetition)}."
        assert isinstance(batch_normalization, bool), f"batch_normalization has to be an {bool} instance, not {type(batch_normalization)}."
        super().__init__()

        self._network = nn.ModuleList()
        self.repetition = repetition
        self.residual = residual

        for _ in range(repetition):
            self._network.append(nn.Sequential(ConvBlock(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0, batch_normalization=batch_normalization), 
                                 ConvBlock(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, batch_normalization=batch_normalization)))
            
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