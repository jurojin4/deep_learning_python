from .yolov1 import YOLOV1
from .loss import YOLOV1Loss
from ...trainer import Trainer
from .yolo_tools import get_bboxes
from typing import List, Literal, Tuple, Union
from torch.utils.data import DataLoader
from .prepare_dataset import Compose, YOLOV1Dataset, collate_fn

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

class YOLOV1Trainer(Trainer):
    """
    Trainer class for YOLOV1 model.
    """
    def __init__(self, dataset_name, dataset_path, epochs, image_size, batch_size, iou_threshold_overlap = None, confidence_threshold = None, warmup_epoch = 1, learning_rate = 0.00001, milestones = None, detail = False, no_measure = False, save = False, save_metric = "loss", box_format = "xywh", data_aug = False, delete = False, weights_path = None, load_all = False, experiment_name = None, no_verbose = False):
        assert isinstance(weights_path, str) or weights_path == None, f"weights_path has to be a {str} instance or equals to {None}, not {type(weights_path)}."
        self._train_loader, self._validation_loader = self._define_model_dataloader(batch_size, dataset_name=dataset_name, dataset_path=dataset_path, image_size=image_size, box_format=box_format, data_aug=data_aug)
        super().__init__(dataset_name, dataset_path, epochs, image_size, batch_size, iou_threshold_overlap, confidence_threshold, warmup_epoch, learning_rate, milestones, detail, no_measure, save, save_metric, box_format, data_aug, delete, weights_path, load_all, experiment_name, no_verbose)

    def _define_model_dataloader(self, batch_size: int, dataset_name: str, dataset_path: str, image_size: Union[int, Tuple[int, int]], box_format: Literal["xyxy", "xywh", "xcycwh"] = "xywh", data_aug: bool = False):
        """
        Method that defines the dataloaders for training set and validation set.
        
        :param int **batch_size**: Batch size.
        :param str **dataset_name**: Name of the dataset.
        :param str **dataset_path**: Path of the dataset
        :param Union[int, Tuple[int, int]] **image_size**: Image size.
        :param Literal["xyxy", "xywh", "xcycwh"] **box_format**: Format of bounding boxes. Set to `xywh`.
        :param bool **data_aug**: Boolean that allows to use data augmentation on the choosen dataset. Set to `False`.
        :return: DataLoaders for training set and validation set.
        :rtype: Tuple[DataLoader, DataLoader]
        """
        print(f"Dataset:")

        transformations = Compose([transforms.ToTensor()])

        trainset = YOLOV1Dataset(dataset_name=dataset_name,
                                 dataset_path=dataset_path, 
                                 S=7,
                                 B=2, 
                                 width=image_size[1],
                                 height=image_size[0],
                                 transformations=transformations,
                                 model_normalized=True,
                                 box_format=box_format,
                                 type_set="train",
                                 data_aug=data_aug)
        
        self._weights, self._num_classes, self._categories, self._dataset_name, self._mode = trainset.weights, trainset.num_classes, trainset.categories, trainset.dataset_name, trainset.mode

        validationset = YOLOV1Dataset(dataset_name=dataset_name,
                                 dataset_path=dataset_path, 
                                 S=7,
                                 B=2, 
                                 width=image_size[1],
                                 height=image_size[0],
                                 transformations=transformations,
                                 model_normalized=True,
                                 box_format=box_format,
                                 type_set="validation")    
        
        if self._mode == "classification":
            train_loader = DataLoader(dataset=trainset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=os.cpu_count(),
                                    pin_memory=True,
                                    drop_last=True)            
            validation_loader = DataLoader(dataset=validationset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=os.cpu_count(),
                                    pin_memory=True,
                                    drop_last=True)
        else:
            train_loader = DataLoader(dataset=trainset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=os.cpu_count(),
                                    pin_memory=True,
                                    drop_last=True,
                                    collate_fn=collate_fn)
            
            validation_loader = DataLoader(dataset=validationset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=os.cpu_count(),
                                    pin_memory=True,
                                    drop_last=True,
                                    collate_fn=collate_fn)
        return train_loader, validation_loader

    def _define_model(self) -> YOLOV1:
        """
        Method that defines YOLOV1 model.

        :return: YOLOV1 model.
        :rtype YOLOV1
        """
        model = YOLOV1(in_channels=3, image_width=self._image_size[0], image_height=self._image_size[1], num_classes=self._num_classes, B=2, mode=self._mode)

        if self._weights_path is not None:
            model.load(self._weights_path, self._load_all)
            
        return model

    def _define_loss(self) -> Union[nn.CrossEntropyLoss, YOLOV1Loss]:
        """
        Method that defines YOLOV1 loss.

        :return: YOLOV1 loss.
        :rtype YOLOV1
        """
        if self._mode == "classification":
            loss = nn.CrossEntropyLoss(weight=torch.tensor(self._weights).to("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            if isinstance(self._model, nn.DataParallel):
                loss = YOLOV1Loss(self._num_classes, self._model.module.S, self._model.module.B)
            else:
                loss = YOLOV1Loss(self._num_classes, self._model.S, self._model.B)
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
    
    def _model_tools(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> Tuple[List[List[float | int]], List[List[float | int]]]:
        """
        Method that transforms and reshapes predictions and ground_truths in order to be evaluate.

        :param torch.Tensor **predictions**: Model's predictions.
        :param torch.Tensor **ground_truths**: Ground truths.
        :return: Predictions and ground truths sorted according confidence threshold and IoU overlap threshold.
        :rtype: Tuple[List[List[float | int]], List[List[float | int]]]
        """
        if isinstance(self._model, nn.DataParallel):
            return get_bboxes(predictions=predictions, ground_truths=ground_truths, S=self._model.module.S, B=self._model.module.B, num_classes=self._num_classes, iou_threshold=self._iou_threshold_overlap, confidence_threshold=self._confidence_threshold)
        else:
            return get_bboxes(predictions=predictions, ground_truths=ground_truths, S=self._model.S, B=self._model.B, num_classes=self._num_classes, iou_threshold=self._iou_threshold_overlap, confidence_threshold=self._confidence_threshold)