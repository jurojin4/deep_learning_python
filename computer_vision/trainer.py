from tqdm import tqdm
from time import time
from typing import Any, Literal
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, List, Literal, Tuple, Union
from .metrics import Accuracy, Precision, Recall, F1_Score, Mean_Average_Precision

import os
import torch
import random
import pickle
import torch.nn as nn

def integer_to_string(integer):
    string = str(integer)
    new_int = ""
    for i in range(1, len(string)+1):
        new_int = string[-i] + new_int
        if i % 3 == 0:
            new_int = " " + new_int

    return new_int

class Trainer:
    """
    Class that train models.
    """
    def __init__(self, dataset_name: Literal["cifar10", "coco2017", "modified_coco2017", "facedetection", "pascalvoc2007", "pascalvoc2012", "tiny-imagenet200"], dataset_path: str, epochs: int, image_size: int, batch_size: int, warmup_epoch: int = 1, learning_rate: float = 1e-4, milestones: Union[List[int], None] = None, detail: bool = False, no_measure: bool = False, save: bool = False, save_metric: Literal["accuracy", "loss", "mAP", "precision", "recall", "f1_score"] = "loss", box_format: Literal['xyxy', 'xywh', 'xcycwh'] = "xywh", data_aug: bool = False, delete: bool = False, weights_path: Union[str, None] = None, load_all: bool = False, experiment_name: Union[str, None] = None, no_verbose: bool = False):
        """
        Initializes the Trainer class.

        :param Literal["cifar10", "coco2017", "modified_coco2017", "facedetection", "pascalvoc2007", "pascalvoc2012", "tiny-imagenet200"] **dataset_name**: Name of the dataset.
        :param str **dataset_path**: Path of the dataset, where it will be stored.
        :param int **epochs**: Number of iterations during the training.
        :param int **image_size**: Image size.
        :param int **batch_size**: Batch size.
        :param int **warmup_epoch**: Number of epochs where model is not measured. Set to `1`.
        :param float **learning_rate**: Coefficient to apply during gradient descent. Set to `1e-4`.
        :param Union[List[int], None] **milestones**: Integers list of epochs. Set to `None`.
        :param bool **detail**: Boolean that allows to have metrics for each class. Set to `False`.
        :param bool **no_measure**: Boolean that allows to not measure the model. Set to `False`.
        :param bool **save**: Boolean that saves metrics. Set to `False`
        :param Literal["accuracy", "loss", "mAP", "precision", "recall", "f1_score"] **save_metric**: Saves the model according the given metric. Set to `loss`.
        :param Literal["xyxy", "xywh", "xcycwh"] **box_format**: Format of bounding boxes. Set to `xywh`.
        :param bool **data_aug**: Boolean that allows data augmentation.
        :param bool **delete**: Boolean that allows to delete the downloaded dataset. Set to `False`.
        :param Union[str, None] **weights_path**: Path of the weights. Set to `None`.
        :param bool **load_all**: Boolean that allows to load partially or completely the model. Set to `False`
        :param Union[str, None] **experiment_name**: Name of the experiment. Set to `None`.
        :param bool **no_verbose**: Set to `False`.
        """
        self.__getattribute__("_train_loader")
        self.__getattribute__("_validation_loader")
        self.__getattribute__("_weights")
        self.__getattribute__("_num_classes")
        self.__getattribute__("_categories")
        self.__getattribute__("_dataset_name")
        self.__getattribute__("_mode")
        assert dataset_name in ["cifar10", "coco2017", "modified_coco2017", "facedetection", "pascalvoc2007", "pascalvoc2012", "tiny-imagenet200"], f"dataset_name has to be in ['cifar10', 'facedetection', 'pascalvoc2007', 'pascalvoc2012', 'tiny-imagenet200'], and not equal to {dataset_name}."
        assert isinstance(dataset_path, str), f"dataset_path has to be a {str}, not {type(dataset_path)}."
        assert isinstance(epochs, int), f"epochs has to be an {int} instance, not {type(epochs)}."
        assert isinstance(image_size, int) or isinstance(image_size, tuple), f"image_size has to be an {int} or {tuple} of {int} instance, not {type(image_size)}."
        assert isinstance(batch_size, int) and batch_size > 0, f"batch_size has to be an {int}, not {type(batch_size)}."
        assert isinstance(warmup_epoch, int) and warmup_epoch >= -1, f"warmup_epoch has to be an {int}, not {type(warmup_epoch)}."
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
        
        if isinstance(image_size, int):
            self._image_size = (image_size, image_size)
        else:
            self._image_size = image_size

        self._batch_size = batch_size
        self._warmup_epoch = warmup_epoch
        self._learning_rate = learning_rate
        self._milestones = milestones
        self._detail = detail
        self._no_measure = no_measure
        self._save = save
        self._box_format = box_format
        self._data_aug = data_aug
        self._delete = delete
        self._weights_path = weights_path
        self._load_all = load_all
        self._no_verbose = no_verbose

        self._model = self._define_model()

        if torch.cuda.device_count() > 1:
            self._model = nn.DataParallel(self._model)
        elif torch.cuda.device_count() == 1:
            self._model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using {torch.cuda.device_count()} GPU(s)")
        print(f"Model parameters: {integer_to_string(sum([p.numel() for p in self._model.parameters() if p.requires_grad]))}")

        self._device_model = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert isinstance(self._model, (nn.Module, nn.Sequential)), f"model has to be a {nn.Module} or {nn.Sequential} instance, not {type(self._model)}."

        self._loss = self._define_loss()

        optimizer, scheduler = self._define_optimizer()
        assert isinstance(optimizer, Optimizer), f"optimizer has to be a {Optimizer} instance, not {type(optimizer)}."

        if experiment_name is None:
            experiment_name = f"{"".join([chr(random.randint(97, 122)) for _ in range(3)])}-{str(random.randint(0, 10000))}"

        if self._save:
            print(f"Experiment Name: {experiment_name}")

        self._optimizer = optimizer
        self._scheduler = scheduler

        if self._mode == "classification":
            self._metrics = {Accuracy.__name__.lower(): Accuracy(), Precision.__name__.lower(): Precision(self._num_classes, detail), Recall.__name__.lower(): Recall(self._num_classes, detail), F1_Score.__name__.lower(): F1_Score(self._num_classes, detail)}
        else:
            self._metrics = {}

            boolean = True
            while boolean:
                try:
                    self._iou_threshold_overlap = float(input("IoU threshold between two boxes to determine whether they are overlaid. (between 0. and 1.): "))
                    assert isinstance(self._iou_threshold_overlap, float) and self._iou_threshold_overlap >= 0. and self._iou_threshold_overlap <= 1., f"IoU threshold (overlap) has to be between 0. and 1., not {self._iou_threshold_overlap}."                    

                    self._confidence_threshold = float(input("Confidence threshold: "))
                    assert isinstance(self._confidence_threshold, float) and self._confidence_threshold >= 0. and self._confidence_threshold <= 1., f"Confidence threshold (overlap) has to be between 0. and 1., not {self._confidence_threshold}"
        
                    for iou in [i / 100 for i in range(50, 100, 5)]:
                        mAP = Mean_Average_Precision(num_classes=self._num_classes, iou_threshold=iou, detail=detail, box_format=box_format)
                        self._metrics[mAP.name] = mAP
                        boolean = False
                except Exception as e:
                    print(e)
                    continue

        self._metrics_to_use = [item for _, item in self._metrics.items()]

        if save:
            self._define_dirname()
            if not os.path.exists(os.path.join(self._dirname, "model_saves", dataset_name)):
                os.mkdir(os.path.join(self._dirname, "model_saves", dataset_name))
            
            self._save_path = os.path.join(self._dirname, "model_saves", dataset_name, experiment_name)
            if not os.path.exists(self._save_path):
                os.mkdir(self._save_path)
            for param_group in self._optimizer.param_groups:
                lr = param_group['lr']
                break

            _has_been_chosen = True
            if save_metric is not None:
                if save_metric not in list(self._metrics.keys()) + ["loss"]:
                    _has_been_chosen = False
                else:
                    self._save_metric = save_metric
            
            if not _has_been_chosen:
                if self._mode == "classification":
                    self._save_metric = "accuracy"
                else:
                    self._save_metric = "mAP@50"

            model_parameters = sum([p.numel() for p in self._model.parameters() if p.requires_grad])

            self._logs = {"batch_size": batch_size, "data_augmentaion": data_aug, "dataset_name": dataset_name, "dataset_type": self._mode, "epochs": epochs, "experiment_name": experiment_name, "image_size": self._image_size, "learning_rate": lr, "loss": self._loss.__class__.__name__, "num_classes": self._num_classes, "num_parameters": model_parameters, "optimizer": optimizer.__class__.__name__, "save_metric": self._save_metric, "warmup_epoch": self._warmup_epoch}

            if scheduler is not None:
                self._logs["scheduler"] = scheduler.__class__.__name__
                try:
                    self._logs["milestones"] = scheduler.milestones
                except:
                    pass

            if self._mode == "object_detection":
                self._logs["iou_threshold_overlap"] = self._iou_threshold_overlap
                self._logs["confidence_threshold"] = self._confidence_threshold

            try:
                self.__getattribute__("_config")
                self._logs["config"] = self._config["name"]
                print(f"Model has the configuration: {self._config["name"]}")
            except:
                print("Model has no configuration.")

            self._save_pickle(os.path.join(self._save_path, "logs.pickle"), self._logs)
            self._save_pickle(os.path.join(self._save_path, "categories.pickle"), self._categories)

        assert isinstance(self._train_loader, DataLoader), f"train_loader has to be a {DataLoader} instance, not {type(self._train_loader)}."
        assert isinstance(self._validation_loader, DataLoader) or self._validation_loader == None, f"validation_loader has to be a {DataLoader} instance (or None), not {type(self._validation_loader)}."

        if self._mode == "classification":
            self._float_point = 3
        else:
            self._float_point = 5
        
        self._train_metrics = dict([(metric_to_use.name, []) for metric_to_use in self._metrics_to_use])
        self._train_metrics["loss"] = []
        if self._detail:
            self._train_metrics_per_class = {}

        if self._validation_loader is not None:
            self._validation_metrics = dict([(metric_to_use.name, []) for metric_to_use in self._metrics_to_use])
            self._validation_metrics["loss"] = []

            if self._detail:
                self._validation_metrics_per_class = {}

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
        if self._save:
            if "loss" in self._save_metric:
                best_metric = torch.inf
            else:
                best_metric = 0

        self._start_time = time()
        for epoch in range(1, self._epochs + 1):
            if not self._no_verbose:
                lr = self._optimizer.param_groups[0]["lr"]
                print(f"Epoch: {epoch}/{self._epochs} (Learning rate: {lr})")
            
            if self._detail:
                _train_metrics, _train_metrics_per_class = self._train(epoch)
            else:
                _train_metrics = self._train(epoch)

            for key in _train_metrics:
                self._train_metrics[key].append(_train_metrics[key])

            if self._detail:
                for key in _train_metrics_per_class:
                    if key not in self._train_metrics_per_class:
                        self._train_metrics_per_class[key] = dict([(label, []) for label in range(self._num_classes)])
                    for label, value in _train_metrics_per_class[key].items():
                        self._train_metrics_per_class[key][label].append(value)

            if not self._no_verbose:
                self._display_metrics(_train_metrics, "Train")

            if self._validation_loader is not None:
                if self._detail:
                    _validation_metrics, _validation_metrics_per_class = self._validation(epoch)
                else:
                    _validation_metrics = self._validation(epoch)

                for key in _validation_metrics:
                    self._validation_metrics[key].append(_validation_metrics[key])

                if self._detail:
                    for key in _validation_metrics_per_class:
                        if key not in self._validation_metrics_per_class:
                            self._validation_metrics_per_class[key] = dict([(label, []) for label in range(self._num_classes)])
                        for label, value in _validation_metrics_per_class[key].items():
                            self._validation_metrics_per_class[key][label].append(value)

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
                            self._save_model(epoch)
                            best_metric = _validation_metrics[self._save_metric]
                    else:
                        if best_metric < _validation_metrics[self._save_metric]:
                            self._save_model(epoch)
                            best_metric = _validation_metrics[self._save_metric]
                else:
                    if self._save_metric == "loss":
                        if best_metric > _train_metrics[self._save_metric]:
                            self._save_model(epoch)
                            best_metric = _train_metrics[self._save_metric]
                    else:
                        if best_metric < _train_metrics[self._save_metric]:
                            self._save_model(epoch)
                            best_metric = _train_metrics[self._save_metric]

            if self._save:
                self._save_metrics(metrics=self._train_metrics, stage="train")
                if self._detail:
                    self._save_metrics(metrics=self._train_metrics_per_class, stage="train", name="pc")

                if self._validation_loader is not None:
                    self._save_metrics(metrics=self._validation_metrics, stage="validation")
                    if self._detail:
                        self._save_metrics(metrics=self._validation_metrics_per_class, stage="validation", name="pc")

        self._save_model(epoch, last=True)

        return self._model

    def _display_metrics(self, metrics: Dict[str, float], mode: Literal["Train", "Validation"]) -> None:
        """
        Method that displays metrics into the terminal.

        :param Dict[str, float] **metrics**: Dictionary containing as keys metrics name and as items theirs values.
        :param Literal["Train", "Validation"] **mode**: String that can be equal to `Train` or `Validation`.
        """
        for metric_name, measure in metrics.items():
            print(f"{mode} {metric_name}: {round(measure, self._float_point)}", end="\n")

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
                x, y, absolute_y = x.to(self._device_model), y[0].to(self._device_model), y[1]

            y_pred = self._model(x)

            if self._mode == "object_detection":
                y_pred = y_pred.reshape(y.shape)

            loss = self._loss(y_pred, y)
            metrics["loss"].append(loss.item())

            if epoch != 0:
                loss.backward()
                self._optimizer.step()

            if (not self._no_measure or self._epochs == epoch) and epoch > self._warmup_epoch:
                y_pred, y = self._prepare_preds_gts(y_pred, y)
                if self._mode == "object_detection":
                    y_pred, absolute_y = self._model_tools(y_pred, absolute_y.tolist())
                elif y.shape[-1] != 1 and self._mode == "classification":
                    y, y_pred = y.argmax(axis=-1), y_pred.argmax(axis=-1)

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
                    x, y, absolute_y = x.to(self._device_model), y[0].to(self._device_model), y[1]
                
                y_pred = self._model(x)

                if self._mode == "object_detection":
                    y_pred = y_pred.reshape(y.shape)

                loss = self._loss(y_pred, y)
                metrics["loss"].append(loss.item())

                if (not self._no_measure or self._epochs == epoch) and epoch > self._warmup_epoch:
                    y_pred, y = self._prepare_preds_gts(y_pred, y)
                    if self._mode == "object_detection":
                        y_pred, absolute_y = self._model_tools(y_pred, absolute_y.tolist())
                    elif y.shape[-1] != 1 and self._mode == "classification":
                        y, y_pred = y.argmax(axis=-1), y_pred.argmax(axis=-1)

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
        
    def _model_tools(self, predictions: torch.Tensor, ground_truths: torch.Tensor):
        raise NotImplementedError(f'Trainer [{type(self).__name__}] is missing the required "_model_tools" method.')
    
    def _set_postfix(self, metrics: Dict[str, float], stage: Literal["train", "validation"]) -> List[str]:
        """
        Method that sets tqdm bar with current metrics values.

        :param Dict[str, float] **metrics**: Current metrics.
        :param Literal["train", "validation"] **stage**: Current training stage.
        :return:
        :rtype: List[str]

        """
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

    def _prepare_preds_gts(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
    def _save_model(self, epoch: int, last: bool = False) -> None:
        """
        Method that saves the model during the training.

        :param int **epoch**: Iteration.
        :param bool **last**: Boolean that characterizes if the checkpoint is from the last epoch. Set to `False`.
        """
        if last:
            self._model.save(self._save_path, "model_last_epoch")
        else:
            self._model.save(self._save_path, "model_checkpoint")

        self._logs["checkpoint_epoch"] = epoch
        self._logs["training_time_checkpoint"] = time() - self._start_time
        self._save_pickle(os.path.join(self._save_path, "logs.pickle"), self._logs)

        print("checkpoint " + str(epoch) + " saved!", end="\n\n")

    def _save_pickle(self, path: str, object: Any) -> None:
        """
        Method that saves into a pickle file an object.

        :param str **path**: Path of the object to save.
        :param Any **object**: Object to save.
        :rtype: None
        """
        with open(path, "wb") as handle:
            pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

        self._save_pickle(os.path.join(self._save_path, f"{stage}{name}.pickle"), metrics)