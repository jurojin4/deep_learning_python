from tqdm import tqdm
from .yolov3 import YOLOV3
from .loss import YOLOV3Loss
from ...trainer import Trainer
from typing import Dict, List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from .prepare_dataset import Compose, YOLOV3Dataset
from .yolo_tools import get_bboxes, generate_bboxes_prior

import os
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

class YOLOV3Trainer(Trainer):
    """
    Class that train YOLOV3 model.
    """
    def __init__(self, dataset_name, dataset_path, epochs, size, batch_size, learning_rate = 0.00001, milestones = None, detail = False, no_measure = False, save = False, save_metric = "loss", box_format = "xywh", delete = False, weights_path = None, load_all = False, experiment_name = None, no_verbose = False):
        super().__init__(dataset_name, dataset_path, epochs, size, batch_size, learning_rate, milestones, detail, no_measure, save, save_metric, box_format, delete, weights_path, load_all, experiment_name, no_verbose)
        self._batch_size = batch_size
        
    def _define_model(self) -> YOLOV3:
        """
        Method that defines YOLOV3 model.

        :return: YOLOV3 model.
        :rtype YOLOV3
        """
        model = YOLOV3(in_channels=3, num_classes=self._num_classes, img_size=self._size[0], batch_normalization=True, mode=self._mode)

        if self._weights_path is not None:
            model.load(self._weights_path, self._load_all)

        model.to("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")
        return model

    def _define_loss(self):
        """
        Method that defines YOLOV3 loss.

        :return: YOLOV3 loss.
        :rtype YOLOV3
        """
        if self._mode == "classification":
            loss = nn.CrossEntropyLoss(weight=torch.tensor(self._weights).to("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            loss = YOLOV3Loss(self._model._num_classes)
        return loss
        

    def _define_optimizer(self):
        """
        Method that defines optimizer and scheduler.

        :return: Optimizer and scheduler.
        """
        if self._milestones is None:
            optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate, weight_decay=5e-4)
            scheduler = None
        else:
            optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self._milestones)

        return optimizer, scheduler
    
    def _define_dirname(self) -> None:
        """
        Method that defines current dirname.
        """
        self._dirname = os.path.dirname(__file__)
    
    def _define_model_dataloader(self, train: Dataset, validation: Dataset, num_boxes_prior_per_scale: int = 3) -> Tuple[DataLoader, DataLoader]:
        """
        Method that defines the dataloaders for training set and validation set.

        :param Dataset **train**: Training set.
        :param Dataset **validation**: Validation set.
        :param int **num_boxes_prior_per_scale**: Number of boxes prior per scale.
        :return: DataLoaders for training set and validation set.
        :rtype: Tuple[DataLoader, DataLoader]
        """
        compose = Compose([transforms.Resize(self._size), transforms.ToTensor()])

        self._bboxes_prior = generate_bboxes_prior(train, num_boxes_prior_per_scale, self._box_format) if self._mode == "object_detection" else None

        train_dataset = YOLOV3Dataset(dataset=train, num_classes=self._model._num_classes, S=[self._size[0] // 32, self._size[0] // 16, self._size[0] // 8], bboxes_prior=self._bboxes_prior, 
                                      mode=self._mode, box_format=self._box_format, compose=compose)

        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=self._batch_size,
                                shuffle=True,
                                num_workers=os.cpu_count(),
                                pin_memory=True,
                                drop_last=True)
        
        validation_dataset = YOLOV3Dataset(dataset=validation, num_classes=self._model._num_classes, S=[self._size[0] // 32, self._size[0] // 16, self._size[0] // 8], bboxes_prior=self._bboxes_prior, 
                                      mode=self._mode, box_format=self._box_format, compose=compose)
        validation_loader = DataLoader(dataset=validation_dataset,
                                batch_size=self._batch_size,
                                shuffle=True,
                                num_workers=os.cpu_count(),
                                pin_memory=True,
                                drop_last=True)
        
        if self._mode == "object_detection":
            self._bboxes_prior = (self._bboxes_prior * torch.tensor([self._size[0] // 32, self._size[0] // 16, self._size[0] // 8]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(self._device_model)

        if self._save and self._mode == "object_detection":
            with open(os.path.join(self._save_path, "bboxes_prior.pickle"), "wb") as handle:
                pickle.dump(self._bboxes_prior.tolist(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return train_loader, validation_loader
    
    def _train(self, epoch: int) -> Union[Tuple[Dict[str, float], Dict[str, Dict[int, List[float]]]], Dict[str, float]]:
        """
        Method that trains the model.

        :param int **epoch**: Current training epoch.
        :return: Dictionaries containing the global average metrics and average metrics per class for the epoch.
        :rtype: Union[Tuple[Dict[str, float], Dict[str, Dict[int, List[float]]]], Dict[str, float]]
        """
        self._model.train()
        metrics = dict([(metric_to_use.name, []) for metric_to_use in self._metrics_to_use])
        metrics["loss"] = []

        if self._detail:
            metrics_per_class = {}
        else:
            metrics_per_class = None

        if not self._no_verbose:
            loop = tqdm(self._train_loader, leave=True)
        else:
            loop = self._train_loader

        for x, y in loop:
            self._optimizer.zero_grad()

            if self._mode == "classification":
                x, y = x.to(self._device_model), y.to(self._device_model)
            else:
                x = x.to(self._device_model)
                y0, y1, y2 = (y[0].to(self._device_model), y[1].to(self._device_model), y[2].to(self._device_model))

            y_pred = self._model(x)

            if self._mode == "classification":
                loss = self._loss(y_pred, y)
            else:
                loss = (self._loss(y_pred[0], y0, self._bboxes_prior[0]) + self._loss(y_pred[1], y1, self._bboxes_prior[1]) + self._loss(y_pred[2], y2, self._bboxes_prior[2]))
            metrics["loss"].append(loss.item())

            if epoch != 0:
                loss.backward()
                self._optimizer.step()

            if not self._no_measure or self._epochs == epoch:
                if self._mode == "object_detection":
                    y_pred, y = get_bboxes(y_pred, y, self._bboxes_prior, self._num_classes, self._iou_threshold_overlap, self._confidence_threshold)

                for metric in self._metrics_to_use:
                    measure = metric(y_pred, y)
                    if isinstance(measure, tuple):
                        metrics[metric.name].append(measure[0])
                    else:
                        metrics[metric.name].append(measure)

                    if self._detail and isinstance(measure, tuple):
                        if metric.name not in metrics_per_class.keys():
                            metrics_per_class[metric.name] = dict([(label, []) for label in range(self._num_classes)])

                        for label, value in measure[1].items():
                            metrics_per_class[metric.name][label].append(value)

            if not self._no_verbose:
                loop.set_postfix(self._set_postfix(metrics, "train"))

        self._average(metrics, metrics_per_class)

        if self._detail:
            return metrics, metrics_per_class
        else:
            return metrics
                
    def _validation(self, epoch: int) -> Union[Tuple[Dict[str, float], Dict[str, Dict[int, List[float]]]], Dict[str, float]]:
        """
        Method that evaluates the model.

        :param int **epoch**: Current training epoch.
        :return: Dictionaries containing the global average metrics and average metrics per class for the epoch.
        :rtype: Union[Tuple[Dict[str, float], Dict[str, Dict[int, List[float]]]], Dict[str, float]]
        """
        self._model.eval()
        metrics = dict([(metric_to_use.name, []) for metric_to_use in self._metrics_to_use])
        metrics["loss"] = []

        if self._detail:
            metrics_per_class = {}
        else:
            metrics_per_class = None

        if not self._no_verbose:
            loop = tqdm(self._validation_loader, leave=True)
        else:
            loop = self._validation_loader

        for x, y in loop:
            with torch.no_grad():
                if self._mode == "classification":
                    x, y = x.to(self._device_model), y.to(self._device_model)
                else:
                    x = x.to(self._device_model)
                    y0, y1, y2 = (y[0].to(self._device_model), y[1].to(self._device_model), y[2].to(self._device_model))

                y_pred = self._model(x)
                if self._mode == "classification":
                    loss = self._loss(y_pred, y)
                else:
                    loss = (self._loss(y_pred[0], y0, self._bboxes_prior[0]) + self._loss(y_pred[1], y1, self._bboxes_prior[1]) + self._loss(y_pred[2], y2, self._bboxes_prior[2]))
                metrics["loss"].append(loss.item())

                if not self._no_measure or self._epochs == epoch:
                    if self._mode == "object_detection":
                        y_pred, y = get_bboxes(y_pred, y, self._bboxes_prior, self._num_classes, self._iou_threshold_overlap, self._confidence_threshold)

                        for metric in self._metrics_to_use:
                            measure = metric(y_pred, y)
                            if isinstance(measure, tuple):
                                metrics[metric.name].append(measure[0])
                            else:
                                metrics[metric.name].append(measure)

                            if self._detail and isinstance(measure, tuple):
                                if metric.name not in metrics_per_class.keys():
                                    metrics_per_class[metric.name] = dict([(label, []) for label in range(self._num_classes)])

                                for label, value in measure[1].items():
                                    metrics_per_class[metric.name][label].append(value)

                if not self._no_verbose:
                    loop.set_postfix(self._set_postfix(metrics, "validation"))
        
        self._average(metrics, metrics_per_class)
        
        if self._detail:
            return metrics, metrics_per_class
        else:
            return metrics