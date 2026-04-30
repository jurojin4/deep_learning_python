from PIL import Image
from tqdm import tqdm
from .datasets import Compose
from typing import Any, List, Tuple, Union

import os
import cv2
import torch
import pickle
import imageio
import argparse
import numpy as np
import torchvision.transforms as transforms

class VideoMaker:
    """
    Base class for making video treated by a model.
    """
    def __init__(self, video_path: str, weights_path: str, iou_threshold_overlap: Union[float, None] = None, confidence_threshold: Union[float, None] = None, fps: int = 30, display_label: bool = False):
        """
        Initializes the VideoMaker class.

        :param str **video_path**: Path of the video.
        :param str **weights_path**: Model weights path.
        :param Union[float, None] iou_threshold_overlap: IoU threshold overlap between bouding boxes predictions.
        :param Union[float, None] **confidence_threshold**: Threshold for reliability probability of predictions according the model.
        :param int **fps**: Frames per second of the video output.
        """
        self.__getattribute__("model_box_format")
        self.__getattribute__("video_output_path")
        
        assert isinstance(video_path, str), f"video_path has to be a {str} instance, not {type(video_path)}."
        assert isinstance(weights_path, str), f"weights_path has to be a {str} instance, not {type(weights_path)}."
        assert (isinstance(iou_threshold_overlap, float) and iou_threshold_overlap >= 0. and iou_threshold_overlap <= 1) or iou_threshold_overlap is None, f"iou_threshold_overlap has to be a {float} instance in [0;1], not {type(iou_threshold_overlap)}."
        assert (isinstance(confidence_threshold, float) and confidence_threshold >= 0. and confidence_threshold <= 1) or confidence_threshold is None, f"confidence_threshold has to be a {float} instance in [0;1], not {type(confidence_threshold)}."
        assert isinstance(fps, int), f"confidence_threshold has to be a {int} instance, not {type(fps)}."
        
        self.video_path = video_path
        self.video_name = os.path.basename(video_path).split(".")[0]
        self.fps = fps
        self._display_label = display_label

        self._cap = cv2.VideoCapture(self.video_path)
        self._num_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with open(os.path.join(os.path.dirname(weights_path), "logs.pickle"), "rb") as file:
            self._logs = pickle.load(file)

        with open(os.path.join(os.path.dirname(weights_path), "categories.pickle"), "rb") as file:
            self._categories = pickle.load(file)

        self._categories = dict([(value, key) for key, value in self._categories.items()])
        
        self._mode = self._logs["dataset_type"]
        self._num_classes = self._logs["num_classes"]
        self._image_height, self._image_width = self._logs["image_size"]

        if iou_threshold_overlap is not None:
            self._iou_threshold_overlap = iou_threshold_overlap
        else:
            self._iou_threshold_overlap = self._logs["iou_threshold_overlap"]

        if confidence_threshold is not None:
            self._confidence_threshold = confidence_threshold
        else:
            self._confidence_threshold = self._logs["confidence_threshold"]

        self._define_model(weights_path)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        
        self._transformations = Compose([transforms.Resize((self._image_height, self._image_width)), transforms.ToTensor()])

        if len(self._categories) != 1:
            self._colors = []
            for i in range(len(self._categories)):
                color = (255, 255, 255)
                self._colors.append(color)
        else:
            self._colors = [(0, 255, 0)]

    def _define_model(self, weights_path: str):
        raise NotImplementedError(f'VideoMaker [{type(self).__name__}] is missing the required "_define_model" method.')
    
    def _model_tools(self, predictions: Any):
        raise NotImplementedError(f'VideoMaker [{type(self).__name__}] is missing the required "_model_tools" method.')

    def _bbox_transformation(self, bbox: List[float]) -> List[float]:
        """
        Method that transforms bounding boxes to a define box format.

        :param List[float] **bbox**: Bounding boxes.
        :return: Bounding boxes with a specified format.
        :rtype: List[float]
        """
        if self.model_box_format == "xyxy":
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
        elif self.model_box_format == "xywh":
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2] + bbox[0]
            y2 = bbox[3] + bbox[1]
        else:
            x1 = bbox[0] - (bbox[2] / 2)
            y1 = bbox[1] - (bbox[3] / 2)
            x2 = bbox[0] + (bbox[2] / 2)
            y2 = bbox[1] + (bbox[3] / 2)
        
        return [x1, y1, x2, y2]

    @property
    def _get_frame(self) -> Tuple[True, np.ndarray]:
        """
        Property that gets the next frame.
        """
        ret, frame = self._read
        if not ret:
            return ret, None

        b = frame[..., 0]
        g = frame[..., 1]
        r = frame[..., 2]

        return ret, np.stack((r, g, b), axis=-1)

    @property
    def _read(self):
        """
        Property that reads the next frame.
        """
        return self._cap.read()

    def __call__(self) -> None:
        """
        Built-in Python method that allows to call VideoMaker instances.
        """
        self._model.eval()
        frames = []
        for i in tqdm(range(self._num_frames), leave=True):
            ret, frame = self._get_frame
            if not ret:
                break
            
            height, width = frame.shape[0:2]
            x = self._transformations(Image.fromarray(frame).resize((self._image_height, self._image_width))).unsqueeze(0).to(self._device)

            with torch.no_grad():
                y_pred = self._model(x)

            pred_bboxes = self._model_tools(y_pred)
            for label, bboxes in enumerate(pred_bboxes):
                for bbox in bboxes:
                    obj = bbox[1]
                    x1, y1, x2, y2 = self._bbox_transformation(bbox[2:])
                    x1 = int(x1 * width)
                    y1 = int(y1 * height)
                    x2 = int(x2 * width)
                    y2 = int(y2 * height)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self._colors[label], 2)

                    if self._display_label:
                        cv2.putText(frame, f"{self._categories[label]}: {obj:.2f}", (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._colors[label], 1, cv2.LINE_AA)

            frames.append(frame)

        self._cap.release()

        path = os.path.join(self.video_output_path, f"{self.model_name}_{self.video_name}.mp4")        
        imageio.mimsave(path, frames, fps=self.fps, codec="libx264")
        
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='VideoMaker', add_help=add_help)
    parser.add_argument("--video_path", default="None", type=str, help='Path of the video')
    parser.add_argument("--weights_path", default="None", type=str, help='Path of the weights')
    parser.add_argument("--iou_threshold_overlap", default=None, type=float, help='IoU Threshold overlap between bouding boxes')
    parser.add_argument("--confidence_threshold", default=None, type=float, help='Threshold for reliability probability of predictions according the model')
    parser.add_argument("--fps", default=30, type=int, help='Frames per second of the video output.')
    parser.add_argument("--display_label", action="store_true", help='Display labels at the left top of the bounding boxes.')
    return parser