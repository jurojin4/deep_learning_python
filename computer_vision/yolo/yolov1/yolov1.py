from math import floor
from ...modules import ConvBlock
from typing import Literal, Tuple

import os
import torch
import torch.nn as nn

# (convolutional neurons, kernel_size, stride, padding)
darknet_architecture = [
    (64, 7, 2, 3),
    "maxpooling",
    (192, 3, 1, 1),
    "maxpooling",
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    "maxpooling",
    [(256, 1, 1, 0), (512, 3, 1, 1)] * 4,
    (512, 1, 1, 0),
    (1024, 3, 1, 1),
    "maxpooling",
    [(512, 1, 1, 0), (1024, 3, 1, 1)] * 2,
    (1024, 3, 1, 1),
    (1024, 3, 2, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1)]

class Darknet(nn.Module):
    """
    Darknet class.\n\nBackbone of the YOLOV1 model.
    """
    def __init__(self, in_channels: int, image_height: int, image_width: int, bias: bool = True):
        """
        Initializes the Darknet class.

        :param int **in_channels**: Number that represents the size of the first dimension for a 3D object in torch.
        :param int **image_height**: Height of the image.
        :param int **image_width**: Width of the image.
        :param bool **bias**: Set to `True`, adds bias to the weighted sum before applying the activation function.
        """
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width

        self._backbone = self._create_backbone(in_channels=in_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the ConvBlock instance.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        return self._backbone(x)
        
    def _create_backbone(self, in_channels: int, bias: bool = True) -> nn.Sequential:
        """
        Method that creates the darknet network.

        :param int **in_channels**: Number that represents the size of the first dimension for a 3D object in torch.
        :param bool **bias**: Set to `True`, adds bias to the weighted sum before applying the activation function.
        :return: Darknet network.
        :rtype: Sequential
        """
        sequential = []
        for block in darknet_architecture:
            if isinstance(block, tuple):
                sequential.append(ConvBlock(in_channels=in_channels, out_channels=block[0], kernel_size=block[1], stride=block[2], padding=block[3], batch_normalization=True, activation_function="leakyrelu", bias=bias))
                self.image_height, self.image_width = self._image_size(self.image_height, self.image_width, kernel=block[1], stride=block[2], padding=block[3])
            
            if isinstance(block, str):
                if block == "maxpooling":
                    sequential.append(nn.MaxPool2d(2, 2))
                    self.image_height, self.image_width = self._image_size(self.image_height, self.image_width, kernel=2, stride=2, padding=0)
                    continue

            if isinstance(block, list):
                for b in block:
                    sequential.append(ConvBlock(in_channels=in_channels, out_channels=b[0], kernel_size=b[1], stride=b[2], padding=b[3], batch_normalization=True, activation_function="leakyrelu", bias=bias))
                    in_channels = b[0]
                    self.image_height, self.image_width = self._image_size(self.image_height, self.image_width, kernel=b[1], stride=b[2], padding=b[3])

                continue

            in_channels = block[0]
        return nn.Sequential(*sequential)
    
    def _image_size(self, height: int, width: int, kernel: int, stride: int, padding: int) -> Tuple[int, int]:
        """
        Method that calculates the 3D size image according height and width.

        :param int **height**: Height of the image.
        :param int **width**: Width of the image.
        :param int **kernel**: Size of the convolution kernel.
        :param int **stride**: Size of the convolution stride.
        :param int **padding**: Size of the convolution padding.
        :return: Image size after the convolution in this form (height, width).
        :rtype: Tuple[int, int]
        """
        return floor((height + 2 * padding - kernel) / stride) + 1, floor((width + 2 * padding - kernel) / stride) + 1

class Head(nn.Module):
    """
    YOLOV1 Head class.
    """
    def __init__(self, image_height: int, image_width: int, num_classes: int, S: int, B: int, mode: Literal["classification", "object_detection"]):
        """
        Initializes the Head class.

        :param int **image_height**: Height of the image.
        :param int **image_width**: Width of the image.
        :param int **num_classes**: Number of classes to predict.
        :param int **S**: Cells number along height and width.
        :param int **B**: Bounding boxes number in each cell.
        :param Literal["classification", "object_detection"] **mode**: String that defines the model mode.
        """
        super().__init__()
        self._S = S
        self._B = B
        self._num_classes = num_classes if num_classes > 1 else 0
        self._head = self._create_head(image_height, image_width, mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the head.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        return self._head(x)
    
    def _create_head(self, image_height: int, image_width: int, mode: Literal["classification", "object_detection"]) -> nn.Sequential:
        """
        Method that creates YOLOV1 head.

        :param int **image_height**: Height of the image.
        :param int **image_width**: Width of the image.
        :param Literal["classification", "object_detection"] **mode**: String that defines the model mode.
        :return: Head network.
        :rtype: Sequential
        """
        if mode == "classification":
            return nn.Sequential(
                *[nn.Flatten(),
                nn.Linear(1024 * image_height * image_width, 4096),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.1),
                nn.Linear(4096, self._num_classes),
                nn.Softmax(dim = -1)])
        else:
            return nn.Sequential(
                *[nn.Flatten(),
                nn.Linear(1024 * image_height * image_width, 4096),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.1),
                nn.Linear(4096, self._S * self._S * (self._num_classes + (self._B * 5)))])

class YOLOV1(nn.Module):
    """
    YOLOV1 class.
    """
    def __init__(self, in_channels: int, image_height: int, image_width: int, num_classes: int, S: int = 7, B: int = 2, bias: bool = True, mode: Literal["classification", "object_detection"] = "object_detection"):
        """
        Initializes the YOLOV1 class.

        :param int **in_channels**: Number that represents the size of the first dimension for a 3D object in torch.
        :param int **image_height**: Height of the image.
        :param int **image_width**: Width of the image.
        :param int **num_classes**: Number of classes to predict.
        :param int **S**: Cells number along height and width. Set to `7`.
        :param int **B**: Bounding boxes number in each cell. Set to `2`.
        :param bool **bias**: Set to `True`, adds bias to the weighted sum before applying the activation function.
        :param Literal["classification", "object_detection"] **mode**: String that defines the model mode. Set to `object_detection`.
        """
        assert isinstance(in_channels, int), f"in_channels has to be an {int} instance, not {type(in_channels)}."
        assert isinstance(image_height, int), f"image_height has to be an {int} instance, not {type(image_height)}."
        assert isinstance(image_width, int), f"image_width has to be an {int} instance, not {type(image_width)}."
        assert isinstance(num_classes, int), f"num_classes has to be an {int} instance, not {type(num_classes)}."
        assert isinstance(bias, bool), f"bias has to be a {bool} instance, not {type(bias)}."
        assert isinstance(mode, str) and mode in ["classification", "object_detection"], f"mode has to be a {str} instance and in the given list:\n[\"classification\", \"object_detection\"]"
        super().__init__()
        self._backbone = Darknet(in_channels=in_channels, image_width=image_width, image_height=image_height, bias=bias)

        if mode == "classification":
            self.S = None
            self.B = None
        else:
            self.S = S
            self.B = B
        self.num_classes = num_classes if num_classes > 1 else 0

        self._head = Head(self._backbone.image_height, self._backbone.image_height, num_classes=self.num_classes, S=self.S, B=self.B, mode=mode)
        self._mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the YOLOV1 model.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        assert isinstance(x, torch.Tensor), f"x has to a {torch.Tensor} instance, not {type(x)}."
        return self._head(self._backbone(x))
    
    def save(self, dirname: str, filename: str) -> None:
        """
        Method that saves the actual model state.
        
        :param str **dirname**: Dirname path.
        :param str **filename**: Name of the saved checkpoint without extension.
        """
        assert isinstance(dirname, str), f"dirname has to be a {str} instance, not {type(dirname)}."
        assert isinstance(filename, str), f"filename has to be a {str} instance, not {type(filename)}."
        if filename.find(".") != -1:
            print("filename should not have extension.")
            filename = filename.split(".")[0]
        state_dict = self.state_dict()
        checkpoint_path = os.path.join(dirname, filename + ".pth")
        torch.save(state_dict, checkpoint_path)

    def load(self, weights_path: str, all: bool = False) -> None:
        """
        Method that loads weights from a file (.pth).
        
        :param str **weights_path**: Path of the saved weights.
        :param bool **all**: Boolean that allows loading either all weights or only the convolution weights. Set to `False`.
        """
        self.eval()
        if all:
            self.load_state_dict(torch.load(weights_path, map_location=next(self.parameters()).device))
            print("Model load completely.")
        else:
            model_weights = torch.load(weights_path, map_location=next(self.parameters()).device)
            for key in list(model_weights.keys()):
                if "head" in key:
                    model_weights.pop(key)
            self.load_state_dict(model_weights, strict=False)
            print("Model load partially.")