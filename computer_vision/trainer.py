from tqdm import tqdm
from typing import Literal
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, List, Literal, Tuple, Union
from .metrics import Accuracy, Precision, Recall, F1_Score, Mean_Average_Precision
from .dataset import prepare_cifar10, prepare_facedetection, prepare_vocdetection, prepare_tiny_imagenet

import os
import time
import torch
import random
import pickle
import numpy as np
import torch.nn as nn

class Trainer:
    """
    Class that train models.
    """
    def __init__(self, dataset_name: Literal["cifar10", "facedetection", "pascalvoc2007", "pascalvoc2012", "tiny-imagenet200"], dataset_path: str, epochs: int, size: int, batch_size: int, learning_rate: float = 1e-5, milestones: Union[List[int], None] = None, detail: bool = False, no_measure: bool = False, save: bool = False, save_metric: Literal["accuracy", "loss", "mAP", "precision", "recall", "f1_score"] = "loss", box_format: Literal['xyxy', 'xywh', 'xcycwh'] = "xywh", delete: bool = False, weights_path: Union[str, None] = None, load_all: bool = False, experiment_name: Union[str, None] = None, no_verbose: bool = False):
        """
        Initializes the Trainer class.

        :param Literal["cifar10", "facedetection", "pascalvoc2007", "pascalvoc2012", "tiny-imagenet200"] **dataset_name**: Name of the dataset.
        :param str **dataset_path**: Path of the dataset, where it will be stored.
        :param int **epochs**: Number of iterations during the training.
        :param int **size**: Image size.
        :param int **batch_size**: Batch size.
        :param float **learning_rate**: Coefficient to apply during gradient descent. Set to `1e-5`.
        :param Union[List[int], None] **milestones**: Integers list of epochs. Set to `None`.
        :param bool **detail**: Boolean that allows to have metrics for each class. Set to `False`.
        :param bool **no_measure**: Boolean that allows to not measure the model. Set to `False`.
        :param bool **save**: Boolean that saves metrics. Set to `False`
        :param Literal["accuracy", "loss", "mAP", "precision", "recall", "f1_score"] **save_metric**: Saves the model according the given metric. Set to `loss`.
        :param Literal["xyxy", "xywh", "xcycwh"] **box_format**: Format of bounding boxes. Set to `xywh`.
        :param bool **delete**: Boolean that allows to delete the downloaded dataset. Set to `False`.
        :param Union[str, None] **weights_path**: Path of the weights. Set to `None`.
        :param bool **load_all**: Boolean that allows to load partially or completely the model. Set to `False`
        :param Union[str, None] **experiment_name**: Name of the experiment. Set to `None`.
        :param bool **no_verbose**: Set to `False`.
        """
        assert dataset_name in ["cifar10", "facedetection", "pascalvoc2007", "pascalvoc2012", "tiny-imagenet200"], f"dataset_name has to be in ['cifar10', 'facedetection', 'pascalvoc2007', 'pascalvoc2012', 'tiny-imagenet200'], and not equal to {dataset_name}."
        assert isinstance(dataset_path, str), f"dataset_path has to be a {str}, not {type(dataset_path)}."
        assert isinstance(epochs, int), f"epochs has to be an {int} instance, not {type(epochs)}."
        assert isinstance(size, int) or isinstance(size, tuple), f"size has to be an {int} or {tuple} of {int} instance, not {type(size)}."
        assert isinstance(batch_size, int), f"batch_size has to be an {int}, not {type(batch_size)}."
        assert isinstance(learning_rate, float), f"learning_rate has to be a {float} instance, not {type(learning_rate)}."
        assert isinstance(detail, bool), f"detail has to be a {bool} instance, not {type(detail)}."
        assert isinstance(no_measure, bool), f"measure has to be a {bool} instance, not {type(no_measure)}."
        assert isinstance(save, bool), f"save has to be a {bool} instance, not {type(save)}."
        assert isinstance(save_metric, str) or save_metric == None, f"save_metric has to be a {str} instance (or None), not {type(save_metric)}."
        assert box_format in ["xyxy", "xywh", "xcycwh"], f"box_format has to be in [\"xyxy\", \"xywh\", \"xcycwh\"], not {box_format}."
        assert isinstance(no_verbose, bool), f"verbose has to be a {bool} instance, not {type(no_verbose)}."

        super().__init__()

        self._dataset_name = dataset_name
        self._dataset_path = dataset_path
        self._epochs = epochs
        
        if isinstance(size, int):
            self._size = (size, size)

        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._milestones = milestones
        self._detail = detail
        self._no_measure = no_measure
        self._save = save
        self._box_format = box_format
        self._delete = delete
        self._weights_path = weights_path
        self._load_all = load_all
        self._no_verbose = no_verbose

        train, validation, self._weights, self._num_classes, self._categories, dataset_name, self._mode = self._define_dataset(box_format)

        self._model = self._define_model()
        self._device_model = next(self._model.parameters()).device

        assert isinstance(self._model, (nn.Module, nn.Sequential)), f"model has to be a {nn.Module} or {nn.Sequential} instance, not {type(self._model)}."

        self._loss = self._define_loss()

        optimizer, scheduler = self._define_optimizer()

        assert isinstance(optimizer, Optimizer), f"optimizer has to be a {Optimizer} instance, not {type(optimizer)}."

        if experiment_name is None:
            experiment_name = "".join([chr(random.randint(97, 122)) for _ in range(5)] + [str(random.randint(0, 1000))])

        self._optimizer = optimizer
        self._scheduler = scheduler

        if self._mode == "classification":
            self._metrics = {Accuracy.__name__.lower(): Accuracy(), Precision.__name__.lower(): Precision(self._num_classes, detail), Recall.__name__.lower(): Recall(self._num_classes, detail), F1_Score.__name__.lower(): F1_Score(self._num_classes, detail)}
        else:
            self._metrics = {}

            boolean = True
            while boolean:
                try:
                    num_threshold = int(input("Choose how many IoU threshold you want it to measure the model (strictly Postive Integer only): "))
                    assert isinstance(num_threshold, int) and num_threshold >= 1, f"Only stricly positive integer not {num_threshold}."

                    self._iou_threshold_overlap = float(input("IoU threshold between two boxes to determine whether they are overlaid. (between 0. and 1.): "))
                    assert isinstance(self._iou_threshold_overlap, float) and self._iou_threshold_overlap >= 0. and self._iou_threshold_overlap <= 1., f"IoU threshold (overlap) has to be between 0. and 1., not {self._iou_threshold_overlap}."                    

                    self._confidence_threshold = float(input("Confidence threshold: "))
                    assert isinstance(self._confidence_threshold, float) and self._confidence_threshold >= 0. and self._confidence_threshold <= 1., f"Confidence threshold (overlap) has to be between 0. and 1., not {self._confidence_threshold}"
        
                    for i in range(num_threshold):
                        _boolean = True
                        while _boolean:
                            try:
                                iou = float(input(f"IoU threshold {i+1} (between 0. and 1.): "))
                                assert iou < 1 and iou > 0., "IoU threshold should be 0.<= IoU threshold <= 1."
                                mAP = Mean_Average_Precision(num_classes=self._num_classes, batch_size=self._batch_size, iou_threshold=iou, detail=detail)
                                self._metrics[mAP.name] = mAP
                                _boolean = False
                            except Exception as e:
                                print(e)
                                continue

                        boolean = False
                except Exception as e:
                    print(e)
                    continue

        self._metrics_to_use = [item for _, item in self._metrics.items()]

        if save:
            self._define_dirname()
            if not os.path.exists(os.path.join(self._dirname, "model_saves", dataset_name)):
                os.mkdir(os.path.join(self._dirname, "model_saves", dataset_name))
            
            localtime = time.localtime()
            self._save_path = os.path.join(self._dirname, "model_saves", dataset_name, "_".join([self._model.__class__.__name__, self._mode, str(localtime.tm_year), str(localtime.tm_yday), str(localtime.tm_hour), str(localtime.tm_min), str(localtime.tm_sec)]))
            os.mkdir(self._save_path)
            self._save_log_path = os.path.join(self._save_path, "checkpoint_log.txt")
            
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                break

            model_parameters = sum([p.numel() for p in self._model.parameters() if p.requires_grad])
            with open(self._save_log_path, "a") as file:
                file.write(f"experiment name:{experiment_name}|dataset type:{self._mode}|dataset:{dataset_name}|classes:{self._num_classes}|image size:{self._size}|model parameters:{model_parameters}|epochs:{epochs}|batch:{batch_size}|optimizer:{optimizer.__class__.__name__}|learning rate: {lr}|loss:{self._loss.__class__.__name__}")
                if scheduler is not None:
                    file.write(f"|scheduler:{scheduler.__class__.__name__}|milestones:{scheduler.milestones}")
                if self._mode == "object_detection":
                    file.write(f"|iou threshold overlap:{self._iou_threshold_overlap}|confidence threshold:{self._confidence_threshold}")

            with open(os.path.join(self._save_path, f"categories.pickle"), 'wb') as handle:
                pickle.dump(self._categories, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self._train_loader, self._validation_loader = self._define_model_dataloader(train, validation)

        assert isinstance(self._train_loader, DataLoader), f"train_loader has to be a {DataLoader} instance, not {type(self._train_loader)}."
        assert isinstance(self._validation_loader, DataLoader) or self._validation_loader == None, f"validation_loader has to be a {DataLoader} instance (or None), not {type(self._validation_loader)}."

        self._save_metric = save_metric
        if self._save_metric is not None:
            if self._save_metric not in list(self._metrics.keys()) + ["loss"]:
                self._save_metric = "loss"
        else:
            self._save_metric = "accuracy"

        if self._mode == "classification":
            self._fp = 3
        else:
            self._fp = 5

    def _define_dataset(self, box_format: Literal["xyxy", "xywh", "xcycwh"] = "xywh") -> Tuple[dict[str, List[np.ndarray]], dict[str, List[int]], list[float], int, Dict[str, int], str, str]:
        """
        Method that defines the dataset chosen for the model's training.

        :param Literal["xyxy", "xywh", "xcycwh"] **box_format**: Format of bounding boxes. Set to `xywh`.
        :return: The dataset chosen.
        :rtype: Tuple[dict[str, List[np.ndarray]], dict[str, List[int]], list[float], int, Dict[str, int], str, str]
        """
        if self._dataset_name == "cifar10":
            return prepare_cifar10(path=self._dataset_path, delete=self._delete)
        elif self._dataset_name == "facedetection":
            return prepare_facedetection(path=self._dataset_path, size=self._size, box_format=box_format)
        elif self._dataset_name == "tiny-imagenet200":
            return prepare_tiny_imagenet(path=self._dataset_path, delete=self._delete)
        elif self._dataset_name == "pascalvoc2007":
            return prepare_vocdetection(path=self._dataset_path, year="2007", box_format=box_format)
        else:
            return prepare_vocdetection(path=self._dataset_path, year="2012", box_format=box_format)
        
    def _define_model(self):
        raise NotImplementedError(f'Trainer [{type(self).__name__}] is missing the required "_define_model" method.')
    
    def _define_loss(self):
        raise NotImplementedError(f'Trainer [{type(self).__name__}] is missing the required "_define_loss" method.')

    def _define_optimizer(self):
        raise NotImplementedError(f'Trainer [{type(self).__name__}] is missing the required "_define_optimizer" method.')
    
    def _define_dirname(self):
        raise NotImplementedError(f'Trainer [{type(self).__name__}] is missing the required "_define_dirname" method.')

    def _define_model_dataloader(self, train, validation = None):
        raise NotImplementedError(f'Trainer [{type(self).__name__}] is missing the required "_define_model_dataloader" method.')
    
    def _display_current_lr(self):
        for param_group in self._optimizer.param_groups:
            current_lr = param_group['lr']
            break
        print(f"Learning Rate: {current_lr}")
    
    def __call__(self) -> Union[nn.Module, nn.Sequential]:
        """
        Built-in Python method that allows to call an instance like function, launch the training.

        :return: Model trained.
        :rtype: Union[nn.Module, nn.Sequential]
        """
        train_metrics = dict([(metric_to_use.name, []) for metric_to_use in self._metrics_to_use])
        train_metrics["loss"] = []
        train_metrics_per_class = {}

        if self._validation_loader is not None:
            validation_metrics = dict([(metric_to_use.name, []) for metric_to_use in self._metrics_to_use])
            validation_metrics["loss"] = []
            validation_metrics_per_class = {}

        if self._save_metric == "loss":
            best_metric = torch.inf
        else:
            best_metric = 0

        for epoch in range(0, self._epochs + 1):
            if not self._no_verbose:
                print(f"Epoch: {epoch}/{self._epochs}")
            
            if self._detail:
                _train_metrics, _train_metrics_per_class = self._train(epoch)
            else:
                _train_metrics = self._train(epoch)

            for key in _train_metrics:
                train_metrics[key].append(_train_metrics[key])

            if self._detail:
                for key in _train_metrics_per_class:
                    if key not in train_metrics_per_class:
                        train_metrics_per_class[key] = dict([(label, []) for label in range(self._num_classes)])
                    for label, value in _train_metrics_per_class[key].items():
                        train_metrics_per_class[key][label].append(value)

            if not self._no_verbose:
                self._display_metrics(_train_metrics, "Train")

            if self._validation_loader is not None:
                if self._detail:
                    _validation_metrics, _validation_metrics_per_class = self._validation(epoch)
                else:
                    _validation_metrics = self._validation(epoch)

                for key in _validation_metrics:
                    validation_metrics[key].append(_validation_metrics[key])

                if self._detail:
                    for key in _validation_metrics_per_class:
                        if key not in validation_metrics_per_class:
                            validation_metrics_per_class[key] = dict([(label, []) for label in range(self._num_classes)])
                        for label, value in _validation_metrics_per_class[key].items():
                            validation_metrics_per_class[key][label].append(value)

                if not self._no_verbose:
                    self._display_metrics(_validation_metrics, "Validation")

            if self._scheduler is not None and epoch != 0:
                self._scheduler.step()

            if not self._no_verbose:
                print("\n")

            if self._save and epoch != 0:
                if self._validation_loader is not None:
                    if self._save_metric == "loss":
                        if best_metric > _validation_metrics[self._save_metric]:
                            self._save_model(epoch, _validation_metrics[self._save_metric], self._save_metric, "validation")
                            best_metric = _validation_metrics[self._save_metric]
                    else:
                        if best_metric < _validation_metrics[self._save_metric]:
                            self._save_model(epoch, _validation_metrics[self._save_metric], self._save_metric, "validation")
                            best_metric = _validation_metrics[self._save_metric]
                else:
                    if self._save_metric == "loss":
                        if best_metric > _train_metrics[self._save_metric]:
                            self._save_model(epoch, _train_metrics[self._save_metric], self._save_metric, "train")
                            best_metric = _train_metrics[self._save_metric]
                    else:
                        if best_metric < _train_metrics[self._save_metric]:
                            self._save_model(epoch, _train_metrics[self._save_metric], self._save_metric, "train")
                            best_metric = _train_metrics[self._save_metric]

            if self._save:
                self._save_metrics(metrics=train_metrics, stage="train")
                self._save_metrics(metrics=train_metrics_per_class, stage="train", name="pc")

                if self._validation_loader is not None:
                    self._save_metrics(metrics=validation_metrics, stage="validation")
                    self._save_metrics(metrics=validation_metrics_per_class, stage="validation", name="pc")

        self._save_model(epoch, None, None, None)

        return self._model

    def _display_metrics(self, metrics: Dict[str, float], mode: Literal["Train", "Validation"]) -> None:
        """
        Method that displays metrics into the terminal.

        :param Dict[str, float] **metrics**: Dictionary containing as keys metrics name and as items theirs values.
        :param Literal["Train", "Validation"] **mode**: String that can be equal to `Train` or `Validation`.
        """
        for metric_name, measure in metrics.items():
            print(f"{mode} {metric_name}: {round(measure, self._fp)}", end="|")
        print("\n")

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

            x, y = x.to(self._device_model), y.to(self._device_model)
            y_pred = self._model(x)

            if self._mode == "object_detection":
                y_pred = y_pred.reshape(y.shape)

            loss = self._loss(y_pred, y)
            metrics["loss"].append(loss.item())

            if epoch != 0:
                loss.backward()
                self._optimizer.step()

            if not self._no_measure or self._epochs == epoch:
                y_pred, y = self._prepare_gts_and_preds(y_pred, y)

                if self._mode == "object_detection":
                    y_pred, y = self._model_tools(y_pred, y)
                elif y.shape[-1] != 1 and self._mode == "classification":
                    y, y_pred = y.argmax(axis=-1), y_pred.argmax(axis=-1)

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
                x, y = x.to(self._device_model), y.to(self._device_model)
                y_pred = self._model(x)

                if self._mode == "object_detection":
                    y_pred = y_pred.reshape(y.shape)

                loss = self._loss(y_pred, y)
                metrics["loss"].append(loss.item())

                if not self._no_measure or self._epochs == epoch:
                    y_pred, y = self._prepare_gts_and_preds(y_pred, y)

                    if self._mode == "object_detection":
                        y_pred, y = self._model_tools(y_pred, y)
                    elif y.shape[-1] != 1 and self._mode == "classification":
                        y, y_pred = y.argmax(axis=-1), y_pred.argmax(axis=-1)

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
        
    def _model_tools(self, predictions: torch.Tensor, ground_truths: torch.Tensor):
        raise NotImplementedError(f'Trainer [{type(self).__name__}] is missing the required "_model_tools" method.')
    
    def _set_postfix(self, metrics, stage: Literal["train", "validation"]):
        metrics_postfix = []
        for name, values in metrics.items():
            if len(values) != 0:
                if stage == "train":
                    metrics_postfix.append(("Train " + name, values[-1]))
                else:
                    metrics_postfix.append(("Validation " + name, values[-1]))
        
        return metrics_postfix

    def _average(self, metrics: Dict[str, List[float]], metrics_per_class: Union[Dict[str, Dict[int, List[float]]], None] = None) -> None:
        """
        Method that averages the metrics.

        :param Dict[str, List[float]] **metrics**: Metrics used during the training.
        """
        for metric_name, values in metrics.items():
            if values is None:
                metrics[metric_name] = 0
            elif len(values) != 0:
                metrics[metric_name] = sum(values) / len(values)
            else:
                metrics[metric_name] = 0

        if self._detail:
            for metric_name in metrics_per_class:
                for label, values in metrics_per_class[metric_name].items():
                    if len(metrics_per_class[metric_name][label]) != 0:
                        metrics_per_class[metric_name][label] = sum(metrics_per_class[metric_name][label]) / len(metrics_per_class[metric_name][label])
                    else:
                        metrics_per_class[metric_name][label] = 0

    def _prepare_gts_and_preds(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method that prepares the ground truths and the predictions in order to have same size for the measurements.

        :param Tensor **predictions**: Tensor that contains model predictions of the given batch.
        :param Tensor **ground_truths**: Tensor that contains ground truths of the given batch.
        :return: Ground Truths and model predictions for the given batch.
        :rtype: Tuple[Tensor, Tensor]
        """
        if len(ground_truths.shape) < len(predictions.shape):
            ground_truths = ground_truths.unsqueeze(-1)

        if ground_truths.shape[-1] == predictions.shape[-1]:
            if predictions.dtype == bool:
                return ground_truths, predictions.type(int)
            else:
                return predictions, ground_truths
        else:
            preds = predictions.argmax(axis=-1, keepdims=True)

            shape = ground_truths.shape
            new_predictions = torch.zeros(shape).to(ground_truths.device)
            if shape[-1] == 1:
                for i in range(ground_truths.shape[0]):
                    new_predictions[i] = int(preds[i].item())
            else:
                for i in range(ground_truths.shape[0]):
                    new_predictions[i][int(preds[i].item())] = 1

            return new_predictions, ground_truths
        
    def _save_model(self, epoch, metric_value: Union[float, None], metric_name: Union[str, None], stage: Union[str, None]):
        """
        Method that saves the model during the training.

        :param int **epoch**: Iteration.
        :param Union[float, None] **metric_value**: Value of the metric.
        :param Union[str, None] **metric_name**: The metric that is used to determine the save.
        :param Union[str, None] **stage**: Stage (train, validation) that produces the metric.
        """
        self._model.save(self._save_path, "model_checkpoint")

        if metric_value is not None and metric_name is not None and stage is not None:
            with open(self._save_log_path, "a") as file:
                file.write(f"\nmodel_checkpoint_{epoch}: {metric_value:.4f} of {metric_name} for {stage} stage.")
        print("checkpoint " + str(epoch) + " saved!")

    def _save_metrics(self, metrics: Union[Tuple[Dict[str, float], Dict[str, Dict[int, List[float]]]], Dict[str, float]], stage: str, name: Union[str, None] = None) -> None:
        """
        Method that saves metrics into a pickle file.

        :param Union[Tuple[Dict[str, float], Dict[str, Dict[int, List[float]]]], Dict[str, float]] **metrics**: metrics to save.
        :param str **stage**: Stage (train, validation) that produces the metrics.
        :param str **name**: Extra string for the name file.
        """
        if name is None:
            name = ""
        else:
            name = f"_{name}"
    
        with open(os.path.join(self._save_path, f"{stage}{name}.pickle"), 'wb') as handle:
            pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)