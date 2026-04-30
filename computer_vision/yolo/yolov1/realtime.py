from typing import List
from .yolov1 import YOLOV1
from ...realtime import Realtime, get_args_parser
from .prepare_dataset import Compose
from .yolo_tools import get_bboxes_preds

import torch
import torchvision.transforms as transforms

class YOLOV1Realtime(Realtime):
    """
    YOLOV1Realtime class.
    """
    def __init__(self, weights_path, iou_threshold_overlap=None, confidence_threshold=None, batch_normalization: bool = True, normalize = False, **kwargs):
        self._kwargs = kwargs
        super().__init__(weights_path, iou_threshold_overlap, confidence_threshold, normalize)
    
    @property
    def _define_model(self):
        """
        Property that defines the model.
        """
        return YOLOV1(in_channels=3, image_height=self._image_height, image_width=self._image_width, num_classes=self._num_classes, mode=self._mode)

    def _define_compose(self, normalize: bool = False) -> Compose:
        """
        Method that defines the model.

        :param bool **normalize**: Set to `False`, if `True` then the images are normalized.
        :return: Instance to transform images.
        :rtype: Compose
        """
        if normalize:
            return Compose([transforms.Resize((self._image_height, self._image_width)), transforms.ToTensor(), transforms.Normalize(0, 255)])
        else:
            return Compose([transforms.Resize((self._image_height, self._image_width)), transforms.ToTensor()])
    
    def _model_tools(self, predictions: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Methid that defines model's tools.

        :param Tensor **predictions**: Model's predictions.
        :return: Bounding boxes converted and sorted.
        :rtype: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]
        """
        return get_bboxes_preds(predictions, self._hyperparameters["S"], self._hyperparameters["B"], num_classes=self._num_classes, iou_threshold=self._iou_threshold_overlap, confidence_threshold=self._confidence_threshold)

    @property
    def _model_hyperparameters(self):
        """
        Property that defines model's hyperparameters.
        """
        self._hyperparameters = {}
        if "S" in self._kwargs:
            if isinstance(self._kwargs["S"], int):
                self._hyperparameters["S"] = self._kwargs["S"]
        else:
            self._hyperparameters["S"] = self._model.S
            
        if "B" in self._kwargs:
            if isinstance(self._kwargs["B"], int):
                self._hyperparameters["B"] = self._kwargs["B"]
        else:
            self._hyperparameters["B"] = self._model.B

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    realtime = YOLOV1Realtime(weights_path=args.weights_path,
                              iou_threshold_overlap=args.iou_threshold_overlap,
                              confidence_threshold=args.confidence_threshold,
                              normalize=args.normalize)
    realtime()
