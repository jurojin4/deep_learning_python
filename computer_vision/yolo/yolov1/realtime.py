from typing import List
from .yolov1 import YOLOV1
from ...realtime import Realtime
from .prepare_dataset import Compose
from .yolo_tools import get_bboxes_preds

import torch
import argparse
import torchvision.transforms as transforms

class YOLOV1Realtime(Realtime):
    def __init__(self, weights_path, iou_threshold_overlap=None, confidence_threshold=None, batch_normalization: bool = True, normalize = False, **kwargs):
        self._kwargs = kwargs
        super().__init__(weights_path, iou_threshold_overlap, confidence_threshold, batch_normalization, normalize)
    
    @property
    def _define_model(self):
        """
        Property that defines the model.
        """
        return YOLOV1(in_channels=3, img_size=self._img_size, num_classes=self._num_classes, batch_normalization=self._batch_normalization, mode=self._mode)

    def _define_compose(self, normalize: bool = False) -> Compose:
        """
        Method that defines the model.

        :param bool **normalize**: Set to `False`, if `True` then the images are normalized.
        :return: Instance to transform images.
        :rtype: Compose
        """
        if normalize:
            return Compose([transforms.Resize((self._img_size, self._img_size)), transforms.ToTensor(), transforms.Normalize(0, 255)])
        else:
            return Compose([transforms.Resize((self._img_size, self._img_size)), transforms.ToTensor()])
    
    def _model_tools(self, predictions: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Methid that defines model's tools.

        :param Tensor **predictions**: Model's predictions.
        :return: Bounding boxes converted and sorted.
        :rtype: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]
        """
        return get_bboxes_preds(self._hyperparameters["S"], self._model_hyperparameters["B"], num_classes=self._num_classes, predictions=predictions, iou_threshold=self._iou_threshold_overlap, confidence_threshold=self._confidence_threshold)

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

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOV3 Realtime', add_help=add_help)
    parser.add_argument("--weights_path", default=None, type=str, help="Weights path")
    parser.add_argument("--iou_threshold_overlap", default=0.2, type=float, help="Intersection over Union threshold overlap")
    parser.add_argument("--confidence_threshold", default=0.5, type=float, help="Confidence threshold")
    parser.add_argument("--batch_normalization", default=True, type=bool, help="Batch normalization for YOLOV3 model")
    parser.add_argument("--normalize", default=False, type=bool, help="Normalize images")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    realtime = YOLOV1Realtime(weights_path=args.weights_path,
                              iou_threshold_overlap=args.iou_threshold_overlap,
                              confidence_threshold=args.confidence_threshold,
                              batch_normalization=args.batch_normalization,
                              normalize=args.normalize)
    realtime()
