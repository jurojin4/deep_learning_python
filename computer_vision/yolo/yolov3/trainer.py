from tqdm import tqdm
from .yolov3 import YOLOV3
from .loss import YOLOV3Loss
from ...trainer import Trainer
from torch.utils.data import DataLoader
from typing import Dict, List, Literal, Tuple, Union
from .yolo_tools import get_bboxes, generate_bboxes_prior
from .prepare_dataset import Compose, YOLOV3Dataset, collate_fn

import os
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

class YOLOV3Trainer(Trainer):
    """
    Trainer class for YOLOV3 model.
    """
    def __init__(self, dataset_name, dataset_path, epochs, image_size, batch_size, iou_threshold_overlap = None, confidence_threshold = None, warmup_epoch = 1, learning_rate = 0.00001, milestones = None, detail = False, no_measure = False, save = False, save_metric = "loss", box_format = "xywh", data_aug = False, delete = False, weights_path = None, load_all = False, experiment_name = None, no_verbose = False):
        assert isinstance(weights_path, str) or weights_path == None, f"weights_path has to be a {str} instance or equals to {None}, not {type(weights_path)}."
        self._train_loader, self._validation_loader = self._define_model_dataloader(batch_size, dataset_name=dataset_name, dataset_path=dataset_path, image_size=image_size, weights_path=weights_path, box_format=box_format, data_aug=data_aug)
        super().__init__(dataset_name, dataset_path, epochs, image_size, batch_size, iou_threshold_overlap, confidence_threshold, warmup_epoch, learning_rate, milestones, detail, no_measure, save, save_metric, box_format, data_aug, delete, weights_path, load_all, experiment_name, no_verbose)

        if self._mode == "object_detection":
            if self._weights_path is None:
                self._bboxes_prior = (self._bboxes_prior * torch.tensor([self._image_size[0] // 32, self._image_size[0] // 16, self._image_size[0] // 8]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(self._device_model)
            else:
                self._bboxes_prior = self._bboxes_prior.to(self._device_model)

        if self._save and self._mode == "object_detection":
            with open(os.path.join(self._save_path, "bboxes_prior.pickle"), "wb") as handle:
                pickle.dump(self._bboxes_prior.tolist(), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Bboxes prior saved")

    def _define_model_dataloader(self, batch_size: int, dataset_name: str, dataset_path: str, image_size: Union[int, Tuple[int, int]], weights_path: str, box_format: Literal["xyxy", "xywh", "xcycwh"] = "xywh", data_aug: bool = False) -> Tuple[DataLoader, DataLoader]:
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
        
        trainset = YOLOV3Dataset(dataset_name=dataset_name,
                                 dataset_path=dataset_path, 
                                 S=[image_size[0] // 32, image_size[0] // 16, image_size[0] // 8], 
                                 width=image_size[1],
                                 height=image_size[0],
                                 transformations=transformations,
                                 model_normalized=True,
                                 box_format=box_format,
                                 type_set="train",
                                 data_aug=data_aug)        
        
        self._weights, self._num_classes, self._categories, self._dataset_name, self._mode = trainset.weights, trainset.num_classes, trainset.categories, trainset.dataset_name, trainset.mode

        if weights_path:
            bboxes_prior_path = os.path.join(os.path.dirname(weights_path), "bboxes_prior.pickle")
            if os.path.exists(bboxes_prior_path):
                with open(bboxes_prior_path, "rb") as file:
                    self._bboxes_prior = torch.tensor(pickle.load(file))
                print("Bboxes prior loaded.")
        else:
            self._bboxes_prior = generate_bboxes_prior(trainset._annotations, normalize=not trainset._dataset_normalized, box_format=trainset.dataset_box_format) if self._mode == "object_detection" else None
            print("Bboxes prior generated")
        validationset = YOLOV3Dataset(dataset_name=dataset_name,
                                 dataset_path=dataset_path, 
                                 S=[image_size[0] // 32, image_size[0] // 16, image_size[0] // 8], 
                                 width=image_size[1],
                                 height=image_size[0],
                                 transformations=transformations,
                                 model_normalized=True,
                                 box_format=box_format,
                                 type_set="validation")
        
        if self._mode == "object_detection":
            trainset._set_bboxes_prior(self._bboxes_prior.clone())
            validationset._set_bboxes_prior(self._bboxes_prior.clone())

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
        
    def _define_model(self) -> YOLOV3:
        """
        Method that defines YOLOV3 model.

        :return: YOLOV3 model.
        :rtype YOLOV3
        """
        model = YOLOV3(in_channels=3, num_classes=self._num_classes, image_height=self._image_size[0], image_width=self._image_size[1], mode=self._mode)

        if self._weights_path is not None:
            model.load(self._weights_path, self._load_all)

        return model

    def _define_loss(self) -> Union[nn.CrossEntropyLoss, YOLOV3Loss]:
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
                y0, y1, y2 = (y[0][0].to(self._device_model), y[0][1].to(self._device_model), y[0][2].to(self._device_model))
                absolute_y = y[1]

            y_pred = self._model(x)

            if self._mode == "classification":
                loss = self._loss(y_pred, y)
            else:
                loss = (self._loss(y_pred[0], y0, self._bboxes_prior[0]) + self._loss(y_pred[1], y1, self._bboxes_prior[1]) + self._loss(y_pred[2], y2, self._bboxes_prior[2]))
            metrics["loss"].append(loss.item())

            if epoch != 0:
                loss.backward()
                self._optimizer.step()

            if (not self._no_measure or self._epochs == epoch) and epoch > self._warmup_epoch:
                if self._mode == "object_detection":
                    y_pred, absolute_y = get_bboxes(y_pred, absolute_y.tolist(), self._bboxes_prior, self._num_classes, self._iou_threshold_overlap, self._confidence_threshold)

                for metric in self._metrics_to_use:
                    if self._mode == "classification":
                        measure = metric(y_pred, y)
                    else:
                        measure = metric(y_pred, absolute_y)
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

        with torch.no_grad():
            for x, y in loop:
                if self._mode == "classification":
                    x, y = x.to(self._device_model), y.to(self._device_model)
                else:
                    x = x.to(self._device_model)
                    y0, y1, y2 = (y[0][0].to(self._device_model), y[0][1].to(self._device_model), y[0][2].to(self._device_model))
                    absolute_y = y[1]

                y_pred = self._model(x)
                if self._mode == "classification":
                    loss = self._loss(y_pred, y)
                else:
                    loss = (self._loss(y_pred[0], y0, self._bboxes_prior[0]) + self._loss(y_pred[1], y1, self._bboxes_prior[1]) + self._loss(y_pred[2], y2, self._bboxes_prior[2]))
                metrics["loss"].append(loss.item())

                if (not self._no_measure or self._epochs == epoch) and epoch > self._warmup_epoch:
                    if self._mode == "object_detection":
                        y_pred, absolute_y = get_bboxes(y_pred, absolute_y.tolist(), self._bboxes_prior, self._num_classes, self._iou_threshold_overlap, self._confidence_threshold)

                    for metric in self._metrics_to_use:
                        if self._mode == "classification":
                            measure = metric(y_pred, y)
                        else:
                            measure = metric(y_pred, absolute_y)
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