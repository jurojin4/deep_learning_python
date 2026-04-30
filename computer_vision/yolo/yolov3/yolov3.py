from math import floor
from typing import List, Literal, Tuple
from ...modules import ConvBlock, ResBlock

import os
import torch
import torch.nn as nn

# (convolutional neurons, kernel_size, stride, padding)
darknet_architecture_classification = [
    (32, 3, 1, 1),
    (64, 3, 2, 1),
    ("Residual", 1),
    (128, 3, 2, 1),
    ("Residual", 2),
    (256, 3, 2, 1),
    ("Residual", 8),
    (512, 3, 2, 1),
    ("Residual", 8),
    (1024, 3, 2, 1),
    ("Residual", 4),
    ("Avgpool",)]

# (convolutional neurons, kernel_size, stride, padding)
darknet_architecture_object_detection = [
    (32, 3, 1, 1),
    (64, 3, 2, 1),
    ("Residual", 1),
    (128, 3, 2, 1),
    ("Residual", 2),
    (256, 3, 2, 1),
    ("Residual", 8),
    (512, 3, 2, 1),
    ("Residual", 8),
    (1024, 3, 2, 1),
    ("Residual", 4),
    (512, 1, 1, 0),
    (1024, 3, 1, 1),
    ("ScalePrediction",),
    (256, 1, 1, 0),
    ("Upsample",),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    ("ScalePrediction",),
    (128, 1, 1, 0),
    ("Upsample",),
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    ("ScalePrediction",)]

class ScalePrediction(nn.Module):
    """
    ScalePrediction class: YOLOV3 prediction head.
    """
    def __init__(self, in_channels: int, num_classes: int):
        """
        Initializes the ScalePrediction class.

        :param int **in_channels**: Number that represents the size of the first dimension for a 3D object in torch.
        :param int **num_classes**: Number of classes.
        :param bool **batch_normalization**: Set to `False`, if `True` then a batch normalization layer is added to the convolutional block.
        """
        assert isinstance(in_channels, int), f"in_channels has to be an {int} instance, not {type(in_channels)}."
        assert isinstance(num_classes, int), f"num_classes has to be an {int} instance, not {type(num_classes)}."
        super().__init__()

        self._head = nn.Sequential(ConvBlock(in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1, batch_normalization=True, activation_function="leakyrelu"),
                                    ConvBlock(2 * in_channels, (num_classes + 5) * 3, kernel_size=1, stride=1, padding=0, batch_normalization=False, activation_function=None))
        self._num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the ScalePrediction instance.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        return self._head(x).reshape(x.shape[0], 3, self._num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)

class YOLOV3(nn.Module):
    """
    YOLOV3 class.
    """
    def __init__(self, in_channels: int, num_classes: int, image_height: int, image_width: int,  mode: Literal["classification", "object_detection"] = "object_detection"):
        """
        Initializes the YOLOV3 class.

        :param int **in_channels**: Number that represents the size of the first dimension for a 3D object in torch.
        :param int **num_classes**: Number of classes.
        :param int **img_height**: Image height.
        :param int **img_width**: Image width.
        :param bool **batch_normalization**: Set to `False`, if `True` then a batch normalization layer is added to the convolutional block.
        :param Literal["classification", "object_detection"] **mode**: String that defines the model mode. Set to `object_detection`.
        """
        assert isinstance(in_channels, int), f"in_channels has to be an {int} instance, not {type(in_channels)}."
        assert isinstance(num_classes, int), f"num_classes has to be an {int} instance, not {type(num_classes)}."
        assert isinstance(image_height, int), f"image_height has to be an {int} instance, not {type(image_height)}."
        assert isinstance(image_width, int), f"image_width has to be an {int} instance, not {type(image_width)}."
        assert mode in ["classification", "object_detection"], f"mode has to be a {str} instance and in the given list:\n[\"classification\", \"object_detection\"]"
        super().__init__()

        self._in_channels = in_channels
        self._num_classes = num_classes if num_classes > 1 else 0
        self._image_height = image_height
        self._image_width = image_width
        self.mode = mode
        self._darknet = self._build_network(mode)
        
    def _build_network(self, mode: str) -> nn.ModuleList:
        """
        Method that builds the Darknet53 network.

        :param str **mode**: Model's mode.
        :return: Model's network.
        :rtype: ModuleList
        """
        network = nn.ModuleList()

        architecture = darknet_architecture_classification if mode == "classification" else darknet_architecture_object_detection

        for block in architecture:
            if len(block) == 4:
                out_channels, kernel_size, stride, padding = block
                network.append(ConvBlock(self._in_channels, out_channels, kernel_size, stride, padding, True, activation_function="leakyrelu"))
                self._in_channels = out_channels
                self._image_height, self._image_width = self._image_size(self._image_height, self._image_width, kernel=kernel_size, stride=stride, padding=padding)

            elif len(block) == 2:
                repetition = block[1]
                network.append(ResBlock(self._in_channels, repetition, True))

            elif len(block) == 1:
                if block[0] == "ScalePrediction":
                    network += [ResBlock(self._in_channels, 1, True, False),
                                    ConvBlock(self._in_channels, self._in_channels // 2, kernel_size=1, stride=1, padding=0, batch_normalization=True, activation_function="leakyrelu"),
                                    ScalePrediction(self._in_channels // 2, self._num_classes)]
                    self._in_channels //= 2
                elif block[0] == "Upsample":
                    network.append(nn.Upsample(scale_factor=2))
                    self._in_channels *= 3
                elif mode == "classification":
                    network.append(nn.AvgPool2d(kernel_size=3, stride=1, padding=1))
                    network.append(nn.Flatten())
                    network.append(nn.Linear(1024 * self._img_size * self._img_size, self._num_classes))
                    network.append(nn.Softmax(dim=-1))

        return network
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the YOLOV3 instance.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        if self.mode == "classification":
            return self._forward_classification(x)
        else:
            return self._forward_object_detection(x)
    
    def _forward_classification(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards an input Tensor into the YOLOV3 instance in "classification" mode.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: Tensor
        """
        for block in self._darknet:
            x = block(x)
        return x

    def _forward_object_detection(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Method that forwards an input Tensor into the YOLOV3 instance in "object_detection" mode.

        :param Tensor **x**: Input tensor.
        :return: Output tensor.
        :rtype: List[Tensor]
        """
        tracks = []
        predictions = []
        for block in self._darknet:
            if isinstance(block, ScalePrediction):
                predictions.append(block(x))
                continue
            
            x = block(x)

            if isinstance(block, ResBlock) and block.repetition == 8:
                tracks.append(x)

            elif isinstance(block, nn.Upsample):
                x = torch.cat([x, tracks[-1]], dim=1)
                tracks.pop()

        return predictions
    
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
            for key in model_weights.keys():
                if "head" in key:
                    model_weights.pop(key)
            self.load_state_dict(model_weights, strict=False)
            print("Model load partially.")