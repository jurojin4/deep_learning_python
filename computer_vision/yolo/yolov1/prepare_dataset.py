from PIL import Image
from typing import Callable, Dict, List, Literal, Tuple, Union
from torch.utils.data import Dataset

import torch
import numpy as np

class Compose(object):
    """
    Compose class.
    """
    def __init__(self, transforms: List[Callable]):
        """
        Initializes the Compose class.
        
        :param List[Callable] **transforms**: Transformations to apply to objects (images)."""
        self.transforms = transforms

    def __call__(self, image: Union[np.ndarray, Image.Image]):
        """
        Built-in Python method that allows to call an instance of the class. Applies the transformations.
        
        :param Union[np.ndarray, Image.Image] **image**:
        :return: Image transformed.
        :rtype: ArrayLike
        """
        for t in self.transforms:
            image = t(image)
        return image

class YOLOV1Dataset(Dataset):
    """
    YOLOV1Dataset class.
    """
    def __init__(self, dataset: Union[Dict[str, List[np.ndarray]], Dict[str, List[int]], Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]]]], num_classes: int, S: int = 7, B: int = 2, mode: Literal["classification", "object_detection"] = "object_detection", box_format: Literal["xyxy", "xywh", "xcycwh"] = "xywh", compose: Union[Compose, None] = None):
        """
        Initializes the YOLOV1Dataset class.

        :param Union[Dict[str, List[np.ndarray]], Dict[str, List[int]], Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]]]] **dataset**: Dataset used.
        :param int **num_classes**: Number of classes to predict.
        :param int **S**: Cells number along height and width.
        :param int **B**: Bounding boxes number in each cell.
        :param Literal["classification", "object_detection"] **mode**: String that defines the model mode. Set to `object_detection`.
        :param Literal["xyxy", "xywh", "xcycwh"] **box_format**: Format of bounding boxes. Set to `xywh`.
        :param Union[Compose, None] **compose**: Tool transforming images. Set to `None`.
        """
        assert isinstance(mode, str) and mode in ["classification", "object_detection"], f"mode has to be a {str} instance and in the given list:\n[\"classification\", \"object_detection\"]"
        assert box_format in ["xyxy", "xywh", "xcycwh"], f'box_format has to be in ["xyxy", "xywh", "xcycwh"], not equal to {box_format}.'
        assert isinstance(compose, Compose) or compose == None, f"compose has to be a {Compose} instance (or None), not {type(compose)}."
        super().__init__()
        self.num_classes = num_classes
        self._mode = mode
        self._box_format = box_format
        self._compose = compose

        if mode == "classification":
            self._images = dataset["images"]
            self._labels = dataset["labels"]
            self.S = None
            self.B = None
        else:
            self._images = dataset["images"]
            self._bboxes = dataset["bboxes"]
            self._labels = dataset["labels"]
            self.S = S
            self.B = B

            self._type = isinstance(self._images, dict)
            if self._type:
                self._keys = list(self._images.keys())

    def __len__(self) -> int:
        """
        Built-in Python method that returns the length of the object.
        :return: Number of images in the dataset.
        :rtype: int
        """
        return len(self._images)
    
    def _loc(self, line: torch.Tensor, rate: int = 5) -> int:
        """
        Method that returns the location of the bounding box in the line of size num_classes + B * 2 from case num_classes.
        The method attributes one of B spot in the line, if none is empty then the value returned is superior than B * 5 + num_classes.
        
        :param Tensor **line*: 
        :param int **rate**: Number of elements in a box.
        :return: Beginning of the box.
        :rtype: int
        """
        for i in range(self.num_classes, line.shape[0], rate):
            if line[i] == 0:
                return int(i/rate)
        return int((i/rate)) + 1
    
    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Built-in Python method that allows to access an element from the object.

        :param int **index**:
        :return: In mode "classification", a tensor and a label. In mode "object_detection, a tuple of tensor of size 2.
        :rtype: Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]
        """
        if self._mode == "classification":
            image = self._images[index]
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if self._compose is not None:
                image = self._compose(image)
            
            return image, self._labels[index]
        else:
            if self._type:
                index = self._keys[index]
                
            image = self._images[index]
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if self._compose is not None:
                image = self._compose(image)
            
            grid_label = torch.zeros(size=(self.S, self.S, self.num_classes + (5 * self.B)))

            bboxes = self._bboxes[index]
            labels = self._labels[index]

            for bbox, label in zip(bboxes, labels):
                if self._box_format == "xyxy":
                    x, y, x2, y2 = bbox
                    w = x2 - x
                    h = y2 - y
                elif self._box_format == "xywh":
                    x, y, w, h = bbox
                elif self._box_format == "xcycwh":
                    xc, yc, w, h = bbox
                    x = xc - (w / 2)
                    y = yc - (h / 2)
                i, j = int(y * self.S), int(x * self.S)

                x_cell, y_cell = self.S * x - j, self.S * y - i
                height_cell, width_cell = (h * self.S, w * self.S)

                pos = torch.nonzero(grid_label[i, j, :self.num_classes])
                if pos.nelement() != 0:
                    if pos[0].item() == label:
                        n = self._loc(grid_label[i,j])
                        if n * 5 + self.num_classes < self.B * 5 + self.num_classes:
                            grid_label[i, j, n * 5] = 1

                            box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                            try:
                                grid_label[i, j, n * 5 + 1:(n+1) * 5] = box_coordinates
                            except:
                                break
                        else:
                            break
                    else:
                        continue
                else:
                    if label is not None:
                        grid_label[i, j, self.num_classes] = 1

                        cell_bbox = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                        grid_label[i, j, self.num_classes+1:self.num_classes+5] = cell_bbox

                        if self.num_classes > 1:
                            grid_label[i, j, label] = 1
                    else:
                        break
            
            return image, grid_label