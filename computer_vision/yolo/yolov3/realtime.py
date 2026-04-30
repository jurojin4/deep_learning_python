from typing import List
from .yolov3 import YOLOV3
from ...realtime import Realtime, get_args_parser
from .prepare_dataset import Compose
from .yolo_tools import get_bboxes_preds

import os
import torch
import pickle
import torchvision.transforms as transforms

class YOLOV3Realtime(Realtime):
    """
    YOLOV3Realtime class.
    """
    def __init__(self, weights_path, iou_threshold_overlap=None, confidence_threshold=None, normalize = False):
        super().__init__(weights_path, iou_threshold_overlap, confidence_threshold, normalize)

    @property
    def _define_model(self):
        """
        Property that defines the model.
        """
        return YOLOV3(in_channels=3, num_classes=self._num_classes, image_height=self._image_height, image_width=self._image_width, mode=self._mode)

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
    
    def _model_tools(self, predictions: List[torch.Tensor]):
        """
        Methid that defines model's tools.

        :param Tensor **predictions**: Model's predictions.
        :return: Bounding boxes converted and sorted.
        :rtype: List[List[float | int]]
        """
        return get_bboxes_preds(predictions, bboxes_prior=self._hyperparameters["bboxes_prior"], num_classes=self._num_classes, iou_threshold=self._iou_threshold_overlap, confidence_threshold=self._confidence_threshold)

    @property
    def _model_hyperparameters(self):
        """
        Property that defines model's hyperparameters.
        """
        self._hyperparameters = {}
        with open(os.path.join(os.path.dirname(self._weights_path), "bboxes_prior.pickle"), "rb") as file:
            self._hyperparameters["bboxes_prior"] = torch.tensor(pickle.load(file))

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    realtime = YOLOV3Realtime(weights_path=args.weights_path,
                              iou_threshold_overlap=args.iou_threshold_overlap,
                              confidence_threshold=args.confidence_threshold,
                              normalize=args.normalize)
    realtime()
