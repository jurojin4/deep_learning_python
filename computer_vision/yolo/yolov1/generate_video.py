from .yolov1 import YOLOV1
from typing import List, Union
from .yolo_tools import get_bboxes_preds
from ...generate_video import VideoMaker, get_args_parser

import os
import torch

class YOLOV1VideoMaker(VideoMaker):
    def __init__(self, video_path: str, weights_path: str, iou_threshold_overlap: Union[float, None] = None, confidence_threshold: Union[float, None] = None, fps = 30, display_label: bool = False):
        self.model_box_format = "xywh"
        self.video_output_path = os.path.join(os.path.dirname(__file__), "plots")
        super().__init__(video_path, weights_path, iou_threshold_overlap, confidence_threshold, fps, display_label)

    def _define_model(self, weights_path: str) -> None:
        """
        Method that defines the model.

        :param **str** weights_path: Path of the model weights
        """
        self.model_name = "yolov1"
        self._model = YOLOV1(3, self._image_width, self._image_height, self._num_classes, mode = self._mode)
        self._model.load(weights_path, all=True)
        self._model.eval()
    
    def _model_tools(self, predictions: List[torch.Tensor]):
        """
        Methid that defines model's tools.

        :param Tensor **predictions**: Model's predictions.
        :return: Bounding boxes converted and sorted.
        :rtype: List[List[float | int]]
        """
        return get_bboxes_preds(predictions, self._model.S, self._model.B, num_classes=self._num_classes, iou_threshold=self._iou_threshold_overlap, confidence_threshold=self._confidence_threshold)
    
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    videomaker = YOLOV1VideoMaker(video_path=args.video_path,
                   weights_path=args.weights_path,
                   iou_threshold_overlap=args.iou_threshold_overlap,
                   confidence_threshold=args.confidence_threshold,
                   fps=args.fps,
                   display_label=args.display_label)    
    videomaker()