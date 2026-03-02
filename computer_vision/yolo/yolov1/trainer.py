from .yolov1 import YOLOV1
from .loss import YOLOV1Loss
from .yolo_tools import get_bboxes
from torch.utils.data import Dataset, DataLoader
from .prepare_dataset import Compose, YOLOV1Dataset
from typing import List, Tuple, Union
from ...trainer import Trainer

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

dirname = os.path.dirname(__file__)

class YOLOV1Trainer(Trainer):
    """
    Class that train YOLOV1 model.
    """
    def __init__(self, dataset_name, dataset_path, epochs, size, batch_size, learning_rate = 0.00001, milestones = None, detail= False, no_measure = False, save = False, save_metric = "loss", box_format ="xywh", delete = False, weights_path = None, load_all = False, experiment_name = None, no_verbose = False):
        super().__init__(dataset_name, dataset_path, epochs, size, batch_size, learning_rate, milestones, detail, no_measure, save, save_metric, box_format, delete, weights_path, load_all, experiment_name, no_verbose)
        self._batch_size = batch_size

    def _define_model(self) -> YOLOV1:
        """
        Method that defines YOLOV1 model.

        :return: YOLOV1 model.
        :rtype YOLOV1
        """
        if isinstance(self._size, tuple):
            model = YOLOV1(in_channels=3, img_size=self._size[0], num_classes=self._num_classes, B=2, batch_normalization=True, mode=self._mode)
        else:
            model = YOLOV1(in_channels=3, img_size=self._size, num_classes=self._num_classes, B=2, batch_normalization=True, mode=self._mode)

        if self._weights_path is not None:
            model.load(self._weights_path, self._load_all)

        model.to("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")
        return model

    def _define_loss(self):
        """
        Method that defines YOLOV1 loss.

        :return: YOLOV1 loss.
        :rtype YOLOV1
        """
        if self._mode == "classification":
            loss = nn.CrossEntropyLoss(weight=torch.tensor(self._weights).to("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            loss = YOLOV1Loss(self._model.num_classes, self._model.S, self._model.B)
        return loss
        
    def _define_optimizer(self):
        """
        Method that defines optimizer and scheduler.

        :return: Optimizer and scheduler.
        """
        if self._milestones is None:
            optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
            scheduler = None
        else:
            optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self._milestones)

        return optimizer, scheduler
    
    def _define_dirname(self) -> None:
        """
        Method that defines current dirname.
        """
        self._dirname = os.path.dirname(__file__)
    
    def _define_model_dataloader(self, train: Dataset, validation: Dataset) -> Tuple[DataLoader, DataLoader]:
        """
        Method that defines the dataloaders for training set and validation set.

        :param Dataset **train**: Training set.
        :param Dataset **validation**: Validation set.

        :return: DataLoaders for training set and validation set.
        :rtype: Tuple[DataLoader, DataLoader]
        """
        compose = Compose([transforms.Resize(self._size), transforms.ToTensor()])

        train_dataset = YOLOV1Dataset(dataset=train, num_classes=self._model.num_classes, S=self._model.S, B=self._model.B, mode=self._mode, compose=compose)

        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=self._batch_size,
                                shuffle=True,
                                num_workers=os.cpu_count(),
                                pin_memory=True,
                                drop_last=True)
        
        validation_dataset = YOLOV1Dataset(dataset=validation, num_classes=self._model.num_classes, S=self._model.S, B=self._model.B, mode=self._mode, compose=compose)
        validation_loader = DataLoader(dataset=validation_dataset,
                                batch_size=self._batch_size,
                                shuffle=True,
                                num_workers=os.cpu_count(),
                                pin_memory=True,
                                drop_last=True)
        
        return train_loader, validation_loader
    
    def _model_tools(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> Tuple[List[List[float | int]], List[List[float | int]]]:
        """
        Method that transforms and reshapes predictions and ground_truths in order to be measure.

        :param torch.Tensor **predictions**: Model's predictions.
        :param torch.Tensor **ground_truths**: Ground truths.
        :return: Predictions and ground truths sorted according confidence threshold and IoU overlap threshold.
        :rtype: Tuple[List[List[float | int]], List[List[float | int]]]
        """
        return get_bboxes(self._model.S, self._model.B, self._model.num_classes, predictions=predictions, ground_truths=ground_truths, iou_threshold=self._iou_threshold_overlap, confidence_threshold=self._confidence_threshold)