from PIL import Image
from typing import Union

import os
import time
import torch
import pickle
import argparse
import cv2 as cv

class Realtime:
    """
    Base class for realtime detection.
    """
    def __init__(self, weights_path: str, iou_threshold_overlap: Union[float, None] = None, confidence_threshold: Union[float, None] = None, normalize: bool = False):
        """
        Initializes the Realtime class.

        :param str **weights_path**: Path of model's weights.
        :param Union[float, None] **iou_threshold_overlap**: Threshold for IoU overlap between bounding boxes.
        :param Union[float, None] **confidence_threshold**: Threshold that represents the model's confidence for a prediction.
        :param bool **normalize**: Set to `False`, if `True` then the images are normalized.
        """
        with open(os.path.join(os.path.dirname(weights_path), "logs.pickle"), "rb") as file:
            logs = pickle.load(file)

        with open(os.path.join(os.path.dirname(weights_path), "categories.pickle"), "rb") as file:
            self._categories = pickle.load(file)

        self._weights_path = weights_path
        self._categories = dict([(value, key) for key, value in self._categories.items()])
        self._mode = logs["dataset_type"]
        self._name = logs["dataset_name"]
        self._num_classes = logs["num_classes"]
        self._image_height, self._image_width = logs["image_size"]
        self._iou_threshold_overlap = logs["iou_threshold_overlap"]
        self._confidence_threshold = logs["confidence_threshold"]

        if iou_threshold_overlap:
            self._iou_threshold_overlap = iou_threshold_overlap

        if confidence_threshold:
            self._confidence_threshold = confidence_threshold

        self._model = self._define_model
        self._name = f"{type(self._model).__name__} {self._name}"
        
        if weights_path is not None:
            self._model.load(weights_path=weights_path, all=True)
        print(f"Model parameters: {sum([p.numel() for p in self._model.parameters() if p.requires_grad])}")
        
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()

        self._compose = self._define_compose(normalize)
        self._model_hyperparameters

    @property
    def _define_model(self):
        raise NotImplementedError(f'Realtime [{type(self).__name__}] is missing the required "_define_model" method.')

    def _define_compose(self, normalize: bool = False):
        raise NotImplementedError(f'Realtime [{type(self).__name__}] is missing the required "_define_compose" method.')
    
    def _model_tools(self, predictions, **kwargs):
        raise NotImplementedError(f'Realtime [{type(self).__name__}] is missing the required "_model_tools" method.')

    @property
    def _model_hyperparameters(self):
        raise NotImplementedError(f'Realtime [{type(self).__name__}] is missing the required "_model_hyperparameters" method.')
    
    def __call__(self) -> None:
        """
        Built-in Python method that allows to call an instance like function, launch the training.
        """
        if self._mode == "classification":
            self._classification
        elif self._mode == "object_detection":
            self._object_detection

    @property
    def _classification(self):
        """
        Property that classifies images using a camera.
        """
        vid = cv.VideoCapture(0) 

        frame_width = 480
        frame_height = 640
        vid.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
        vid.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

        prev_frame_time = 0
        new_frame_time = 0

        while(True):
            _, frame = vid.read()
            new_frame_time = time.time()

            with torch.no_grad():
                x = self._compose(Image.fromarray(frame)).unsqueeze(0).to(self._device)

                y_pred = self._model(x)
                label, objectiveness = self._model_tools(y_pred)
                cv.putText(frame, f"{self._categories[label]}: {round(objectiveness, 3)}", (473, 605), cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv.LINE_AA)

                new_frame_time = time.time()
                fps = 1 / (new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time

                cv.putText(frame, f"{int(fps)}", (7, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                cv.imshow(f"{self._name} ({self._mode})", frame)

                if cv.waitKey(1) & 0xFF == ord('q'): 
                    break
        
        vid.release() 
        cv.destroyAllWindows()

    @property
    def _object_detection(self):
        """
        Property that detects objets using a camera.
        """
        vid = cv.VideoCapture(0) 

        frame_width = 640
        frame_height = 480
        vid.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
        vid.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

        prev_frame_time = 0
        new_frame_time = 0

        while(True):
            _, frame = vid.read()
            new_frame_time = time.time()

            height, width,_ = frame.shape

            with torch.no_grad():
                x = self._compose(Image.fromarray(frame))
                x = x.unsqueeze(0)
                x = x.to(self._device)

                y_pred = self._model(x)
                pred_bboxes = self._model_tools(y_pred)
                for label, bboxes in enumerate(pred_bboxes):
                    for box in bboxes:
                        objectiveness = round(float(box[1]), 3)
                        x1 = int(box[2] * width)
                        y1 = int(box[3] * height)
                        x2 = int((box[2] + box[4]) * width)
                        y2 = int((box[3] + box[5]) * height)  
                        cv.rectangle(frame, (x1, y1),(x2, y2),(0, 255, 0), 3)
                        cv.putText(frame, f"{self._categories[label]}: {objectiveness}", (x1 - 5, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv.LINE_AA)

                new_frame_time = time.time()
                fps = 1 / (new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time

                cv.putText(frame, f"{int(fps)}", (7, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                cv.imshow(f"{self._name} ({self._mode})", frame)

                if cv.waitKey(1) & 0xFF == ord('q'): 
                    break
        
        vid.release() 
        cv.destroyAllWindows()

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Realtime', add_help=add_help)
    parser.add_argument("--weights_path", default="None", type=str, help='Path of the weights')
    parser.add_argument("--iou_threshold_overlap", default=None, type=float, help='IoU Threshold overlap between bouding boxes')
    parser.add_argument("--confidence_threshold", default=None, type=float, help='Threshold for reliability probability of predictions according the model')
    parser.add_argument("--normalize", default=False, type=bool, help="Normalize images")

    return parser