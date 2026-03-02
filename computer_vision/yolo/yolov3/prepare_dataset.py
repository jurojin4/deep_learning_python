from PIL import Image
from torch.utils.data import Dataset
from .yolo_tools import intersection_over_union_bboxes_prior
from typing import Callable, Dict, List, Literal, Tuple, Union

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

class YOLOV3Dataset(Dataset):
    """
    YOLOV3Dataset class.
    """
    def __init__(self, dataset: Union[Dict[str, List[np.ndarray]], Dict[str, List[int]], Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]]]], num_classes: int, S: Union[List[int], Tuple[int]], bboxes_prior: Union[torch.Tensor, None] = None, mode: Literal["classification", "object_detection"] = "object_detection", box_format: Literal["xyxy", "xywh", "xcycwh"] = "xywh", compose: Union[Compose, None] = None):
        """
        Initializes the YOLOV3Dataset class.

        :param Union[Dict[str, List[np.ndarray]], Dict[str, List[int]], Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]]]] **dataset**: Dataset used.
        :param int **num_classes**: Number of classes to predict.
        :param Union[List[int], Tuple[int]] **S**: Cells number along height and width per grid (3 grid).
        :param torch.Tensor **bboxes_prior**: Bounding boxes prior from dataset.
        :param Literal["classification", "object_detection"] **mode**: String that defines the model mode. Set to `object_detection`.
        :param Literal["xyxy", "xywh", "xcycwh"] **box_format**: Format of bounding boxes. Set to `xywh`.
        :param Union[Compose, None] **compose**: Tool transforming images. Set to `None`.
        """
        assert len(S) == 3, f"S has to be of length equal to 3, not {len(S)}."
        assert isinstance(bboxes_prior, torch.Tensor) or bboxes_prior is None, f"bboxes_prior has to be a {torch.Tensor} instance, not {type(bboxes_prior)}."
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
        else:
            self._images = dataset["images"]
            self._bboxes = dataset["bboxes"]
            self._labels = dataset["labels"]
            self.S = S
            self._num_bboxes_prior_per_scale = bboxes_prior.shape[1]
            self._bboxes_prior = bboxes_prior.reshape(9, 2)

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

    def __getitem__(self, index) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]:
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
            
            num_elements = 6 if self.num_classes > 1 else 5
            grid_label = [torch.zeros((self._num_bboxes_prior_per_scale, s, s, num_elements), dtype=torch.float32) for s in self.S]

            bboxes = self._bboxes[index]
            labels = self._labels[index]

            if labels != []:
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

                    has_bboxes_prior = [False] * 3

                    ious_prior = intersection_over_union_bboxes_prior(torch.tensor([w, h]), self._bboxes_prior)
                    ious_prior_indices = ious_prior.argsort(descending=True, dim=0)

                    for prior_indice in ious_prior_indices:
                        scale_indice = prior_indice // self._num_bboxes_prior_per_scale

                        prior_indice_compared_to_scale = prior_indice % self._num_bboxes_prior_per_scale

                        s = self.S[scale_indice]
                        i, j = int(s * y), int(s * x)

                        cell_taken = grid_label[scale_indice][prior_indice_compared_to_scale, i, j, 0]
                        if not cell_taken and not has_bboxes_prior[scale_indice]:
                            grid_label[scale_indice][prior_indice_compared_to_scale, i, j, 0] = 1.

                            x_cell, y_cell = s * x - j, s * y - i
                            width_cell, height_cell = (w * s, h * s)

                            box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                            grid_label[scale_indice][prior_indice_compared_to_scale, i, j, 1:5] = box_coordinates

                            if self.num_classes > 1:
                                grid_label[scale_indice][prior_indice_compared_to_scale, i, j, 5] = int(label)
                            has_bboxes_prior[scale_indice] = True

            return image, grid_label