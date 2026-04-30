from .yolov3 import YOLOV3
from ...generate_video import VideoMaker, get_args_parser
from typing import Any, List, Union
from .yolo_tools import get_bboxes_preds

import os
import torch
import pickle

class YOLOV3VideoMaker(VideoMaker):
    def __init__(self, video_path: str, weights_path: str, iou_threshold_overlap: Union[float, None] = None, confidence_threshold: Union[float, None] = None, fps = 30, display_label: bool = False):
        self.model_box_format = "xywh"
        self.video_output_path = os.path.join(os.path.dirname(__file__), "plots")
        super().__init__(video_path, weights_path, iou_threshold_overlap, confidence_threshold, fps, display_label)

    def _define_model(self, weights_path: str):
        """
        Method that defines the model.

        :param **str** weights_path: Path of the model weights
        """
        self.model_name = "yolov3"
        self._model = YOLOV3(3, self._num_classes, self._image_height, self._image_width, self._mode)
        self._model.load(weights_path, all=True)
        self._model.eval()

        with open(os.path.join(os.path.dirname(weights_path), "bboxes_prior.pickle"), "rb") as file:
            self._bboxes_prior= torch.tensor(pickle.load(file))
    
    def _model_tools(self, predictions: List[torch.Tensor]):
        """
        Methid that defines model's tools.

        :param Tensor **predictions**: Model's predictions.
        :return: Bounding boxes converted and sorted.
        :rtype: List[List[float | int]]
        """
        return get_bboxes_preds(predictions, bboxes_prior=self._bboxes_prior, num_classes=self._num_classes, iou_threshold=self._iou_threshold_overlap, confidence_threshold=self._confidence_threshold)
    
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    videomaker = YOLOV3VideoMaker(video_path=args.video_path,
                   weights_path=args.weights_path,
                   iou_threshold_overlap=args.iou_threshold_overlap,
                   confidence_threshold=args.confidence_threshold,
                   fps=args.fps,
                   display_label=args.display_label)
    
    videomaker()