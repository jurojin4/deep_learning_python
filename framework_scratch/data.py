from typing import Literal
from random import randint
from module import Activation_Module
from typing import Any, Dict, List, Tuple, Union
from sklearn.datasets import make_blobs, make_classification, make_circles, make_regression

import nn
import math
import random as rd
import tensyx as ts

def choose_dataset(dataset_type: Literal["blobs", "classification", "circles", "linear_regression", "non_linear_regession"], random_state: int = 0, last_module: Union[Activation_Module, None] = None, **kwargs) -> Tuple[ts.np.ndarray, ts.np.ndarray]:
    """
    Methods that allows to chose a specific dataset.

    :param Literal["blobs", "classification", "circles", "linear_regression", "non_linear_regession"] **dataset_type**: Type dataset.
    :param int, optional **random_state**: Set to `0`.
    :param Activation_Module | None, optional **last_module**: Set to `None`, otherwise the last model module, in general an activation function, to categorical the label array according the activation function.
    :param ****kwargs**: Arbitrary keyword arguments for dataset parameters.\n
        - int **n_samples**: Set to `1000`.
        - int **n_features**: Set to `2`.
        - int **classes**: Set to `2`.
        - int **n_clusters_per_class**: Set to `2`.
        - float **biais**: Set to `ts.np.pi`.        

    :return: Dataset normalized.
    :rtype: Tuple[ts.np.ndarray, ts.np.ndarray]
    """
    assert dataset_type in ["blobs", "classification", "circles", "linear_regression", "non_linear_regression"], "dataset_type has to be in [\"blobs\", \"classification\", \"circles\", \"linear_regression\", \"non_linear_regression\"]"

    keys = kwargs.keys()
    if "n_samples" in keys:
        n_samples = kwargs["n_samples"]
    else:
        n_samples = 1000

    if "n_features" in keys:
        n_features = kwargs["n_features"]
    else:
        n_features = 2

    if "classes" in keys:
        classes = kwargs["classes"]
    else:
        classes = 2

    if "n_clusters_per_class" in keys:
        n_clusters_per_class = kwargs["n_clusters_per_class"]
    else:
        n_clusters_per_class = 2
    
    if "biais" in keys:
        biais = kwargs["biais"]
    else:
        biais = float(ts.np.pi)

    if dataset_type == "blobs":
        X, Y = make_blobs(n_samples=n_samples, n_features=n_features, centers=classes, random_state=random_state)
        for i in range(n_features):
            rd = randint(0, 2)
            if rd == 0:
                X[..., i] = ts.cos(X[..., i])
            elif rd == 1:
                X[..., i] = ts.sin(X[..., i])
            else:
                X[..., i] = ts.cos(X[..., i]) + ts.sin(X[..., i])
        X = normalization(X)
    elif dataset_type == "classification":
        X, Y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=classes, n_clusters_per_class=n_clusters_per_class)
        X = normalization(X)
    elif dataset_type == "circles":
        X, Y = make_circles(n_samples=n_samples, random_state=random_state)
        X = normalization(X)
    elif dataset_type == "linear_regression":
        X, Y = make_regression(n_samples=n_samples, n_features=n_features, bias=biais, random_state=random_state)
        return X, Y
    elif dataset_type == "non_linear_regression":
        X = ts.np.expand_dims(ts.np.linspace(-2 * ts.np.pi, 2 * ts.np.pi, n_samples), -1)
        Y = ts.np.sin(X) + biais
        return X, Y
    if isinstance(last_module, (nn.Sigmoid, nn.TanH, nn.Identity)):
        classes = 1
    elif isinstance(last_module, nn.Linear):
        if last_module.get_parameters["W"].shape[1] == 1:
            classes = 1
        
    Y = to_categorical(Y, classes)

    return X, Y

def normalization(X: ts.np.ndarray, method: Literal["min-max"]="min-max") -> ts.np.ndarray:
    """
    Normalizes the ndarray X dataset.

    :param ndarray **X**: Dataset to normalize.
    :param Literal["min-max"] **method**: Normalization method to use.

    :return: Dataset normalized
    :rtype: ts.np.ndarray
    """
    assert method in ["min-max"], f"method has to be in [\"min-max\"]."
    if method == "min-max":
        minimum = X.min()
        maximum = X.max()
        X_new = (X - minimum) / (maximum - minimum)
        return X_new
    
def one_hot_encoding(Y: ts.np.ndarray, classes: int) -> ts.np.ndarray:
    """
    Transforms a dataset multi-labels/multi-classes of shape (samples, 1) to a shape (samples, classes).

    :param ndarray **Y**: Dataset multi-labels/multi-classes.
    :param int **classes**: Number of classes in the dataset.
    :return: The dataset with the shape (samples, classes).
    :rtype: ts.np.ndarray
    """
    Y_new = ts.np.zeros((Y.shape[0], classes))
    for i in range(Y.shape[0]):
        Y_new[i][int(Y[i])] = 1

    return Y_new

def to_categorical(Y: ts.np.ndarray, classes: int) -> ts.np.ndarray:
    """
    Transforms a dataset multi-labels/multi-classes of shape (samples, 1) to a shape (samples, classes).

    :param ndarray **Y**: Dataset multi-labels/multi-classes.
    :param int **classes**: Number of classes in the dataset.
    :return: Categorized Dataset.
    :rtype: ts.np.ndarray
    """
    if classes >= 2:
        return one_hot_encoding(Y, classes)
    elif classes == 1:
        return ts.np.expand_dims(Y, -1)
    
def to_uncategorical(Y: ts.Tensor) -> ts.np.ndarray:
    """
    Transforms a dataset multi-labels/multi-classes of shape (samples, classes) to a shape (samples, 1).

    :param ndarray **Y**: Dataset multi-labels/multi-classes.
    :param int **classes**: Number of classes in the dataset.
    :return: Uncategorized Dataset.
    :rtype: ts.np.ndarray
    """
    if len(Y.shape) > 1:
        if Y.shape[1] == 1:
            return ts.np.squeeze(Y.tensor, axis=-1)
        else:
            return Y.argmax(axis=-1).tensor
    else:
        return Y.tensor

class Dataset:
    """
    Base class for dataset.
    """
    def __init__(self, X: ts.Tensor, Y: ts.Tensor):
        """
        Initializes the Dataset class.

        :param ts.Tensor **X**: Tensor containing the samples.
        :param ts.Tensor **Y**: Tensor containing the output to predict.
        """
        assert isinstance(X, ts.Tensor), f"X has to be an {ts.Tensor} instance."
        assert isinstance(Y, ts.Tensor), f"Y has to be an {ts.Tensor} instance."
        assert X.shape[0] == Y.shape[0], "X and Y does not have the same number of samples."
        self.X = X
        self.Y = Y

    @property
    def get_dataset(self) -> Tuple[ts.Tensor, ts.Tensor]:
        """
        Property that gets the dataset.

        :return: The dataset.
        :rtype: Tuple[ts.Tensor, ts.Tensor]
        """
        return self.X, self.Y

    def __len__(self) -> int:
        """
        Built-in Python method that returns the dataset length.

        :return: Number of samples in the dataset.
        :rtype: int
        """
        return len(self.X)
    
    def __getitem__(self, index: int) -> Tuple[ts.Tensor, ts.Tensor]:
        """
        Get the specified item according the index in the dataset.

        :param int **index**: Sample index.
        :return: The specified item.
        :rtype: Tuple[ts.Tensor, ts.Tensor]
        """
        return self.X[index], self.Y[index]
    
class DataLoader:
    """
    Base class for Dataloader.
    """
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: Union[bool, int, float, str, bytes, bytearray] = True, shuffle_iteration: bool = True, drop_last: bool = True):
        """
        Initializes the DataLoader class.

        :param Dataset **dataset**: Dataset used.
        :param int **batch_size**: Number of element in a batch.
        :param bool | int | float | str | bytes | byterray, optional **shuffle**: shuffle during the iteration.
        :param bool, optional **shuffle_iteration**: shuffle during each iteration.
        :param bool, optional **drop_last**: drop the last partial batch.
        """
        assert isinstance(dataset, Dataset), f"dataset has to be a {Dataset} instance, not {type(dataset)}."
        assert isinstance(batch_size, int), f"batch_size has to be an {int} instance, not {type(dataset)}."
        assert isinstance(shuffle, (bool, int, float, str, bytes, bytearray)), f"shuffle has to be an instance of [{bool}, {int}, {float}, {str}, {bytes}, {bytearray}], not {type(shuffle)}."
        assert isinstance(shuffle_iteration, bool), f"shuffle_iteration has to be a {bool} instance, not {type(shuffle_iteration)}."
        assert isinstance(drop_last, bool), f"drop_last has to be a {bool} instance, not {type(drop_last)}."

        self.dataset = dataset
        self.batch_size = batch_size

        self.shuffle_iteration = shuffle_iteration
        self.drop_last = drop_last
        self._idx = 0
        
        self._end = len(dataset) / batch_size
        if not drop_last:
            self._end = int(self._end)

        self._ids = [i for i in range(math.ceil(self._end))]

        if isinstance(shuffle, (int, float, str, bytes, bytearray)) and not isinstance(shuffle, bool):
            self._id_shuffle = shuffle
            self._shuffle()
        else:
            if shuffle:
                self._id_shuffle = rd.randint(0, int(1e9))
                self._shuffle()

    def _shuffle(self) -> None:
        """
        Method that shuffles the indices during the iteration.
        """
        rd.seed(self._id_shuffle)
        rd.shuffle(self._ids)

    def __len__(self) -> int:
        """
        Built-in Python method that returns the number of batch(s) during the iteration.

        :return: Number of batch(s) during the iteration.
        :rtype: int
        """
        self._end = len(self.dataset) / self.batch_size
        if not self.drop_last:
            self._end = int(self._end)
        return math.ceil(self._end)
    
    def __iter__(self):
        """
        Built-in Python method that returns the dataloader iterator.

        :return: A dataloader iterator.
        :rtype: Dataloader
        """
        return self
    
    def __next__(self) -> Tuple[ts.Tensor, ts.Tensor]:
        """
        Built-in Python method that returns the next batch in the dataloader.

        :return: Next batch in the dataloader.
        :rtype Tuple[ts.Tensor, ts.Tensor]
        """
        if self._idx < self._end:
            element = self.dataset[self._ids[self._idx] * self.batch_size:(self._ids[self._idx] + 1) * self.batch_size]
            self._idx += 1
            return element
        else:
            self._idx = 0
            if self.shuffle_iteration:
                self._id_shuffle += 1
                self._shuffle()

            raise StopIteration

class Data_Metrics:
    """
    Data_Metrics is a storage class for metrics during a training.
    """
    def __init__(self, stages: Tuple[str, Tuple[ts.Tensor, ts.Tensor]] | List[Tuple[str, Tuple[ts.Tensor, ts.Tensor]]], metrics: List[str]):
        """
        Initializes the Data_Metrics class.

        :param Tuple[str, Tuple[ts.Tensor, ts.Tensor]] | List[Tuple[str, Tuple[ts.Tensor, ts.Tensor]]] **stages**: In this context, stages can be `train` and/or `validation` nay `test`.
        :param List[str] **metrics**: Metrics names used during the stages.
        """
        super().__init__()
        assert isinstance(stages, (list, tuple)), f"stagess has to be a {list} or {tuple} instance, not {type(stages)}."
        assert isinstance(metrics, list), f"metrics has to be a {list} instance, not {type(metrics)}."
        if isinstance(stages[0], str):
            self.stages = [stages[0]]
            self.datasets = dict([(stages[0], stages[1])])
        else:
            self.stages = [stage[0] for stage in stages]
            self.datasets = dict([(stage[0], stage[1]) for stage in stages])

        self._metrics = metrics
        self.metrics_by_stages = dict([(stage, self._init_metrics_dictionnary) for stage in self.stages])

        self.model_parameters = []

    @property
    def _init_metrics_dictionnary(self) -> Dict[str, List[Any]]:
        """
        Property that initializes the dictionnary metrics.

        :return: Dictionary with string keys and their empty list items.
        :rtype Dict[str, List[Any]]
        """
        return dict([(metric, []) for metric in self._metrics])

    def __len__(self) -> int:
        """
        Built-in Python method that returns the metrics number.

        :return: Number of metrics.
        :rtype: int
        """
        if len(self._metrics) == 0:
            return 0
        else:
            n = None
            for metric_name, metric_values in self._metrics:
                if n is None:
                    n = len(metric_values)
                else:
                    if n != len(metric_values):
                        raise TypeError(f"For {metric_name}, the length is different with others metrics")
            return n

    def update(self, stage: str, metrics: Dict[str, int]) -> None:
        """
        Method that updates the different metrics list for a specified stage.

        :param str **stage**: Stage name.
        :param Dict[str, int] **metrics**: Dictionary that contains the values.
        """
        for metric, value in metrics.items():
            self.metrics_by_stages[stage][metric].append(value)

    def add_stages(self, stages: Union[Tuple[str, Tuple[ts.Tensor, ts.Tensor]], List[tuple[str, Tuple[ts.Tensor, ts.Tensor]]]]) -> None:
        """
        Method that adds a stage with its specified dataset.

        :param Tuple[str, Tuple[ts.Tensor, ts.Tensor]] | List[tuple[str, Tuple[ts.Tensor, ts.Tensor]]] **stages**: Storage whose items contain the stage name and their datasets.
        """
        if isinstance(stages[0], str):
            self.stages.append(stages[0])
            self.datasets[stages[0]] = stages[1]
            self.metrics_by_stages[stages[0]] = self._init_metrics_dictionnary
        else:
            self.stages.extend([stage[0] for stage in stages])
            for stage in stages:
                self.datasets[stage[0]] = stage[1]
                self.metrics_by_stages[stage[0]] = self._init_metrics_dictionnary

    @property
    def _verify_size(self) -> bool:
        """
        Property that verifies the size.

        :return: `True` if all metrics list and model's parameters list have the same size otherwise `False`.
        :rtype: bool
        """
        boolean = True
        for stage in self.metrics_by_stages:
            for metric in self.metrics_by_stages[stage]:
                boolean = len(self.model_parameters) == len(self.metrics_by_stages[stage][metric])
                if not boolean:
                    raise TypeError(f"Different size between model's parameters list and {metric} list in {stage} stage:\nlen(self.model_parameters) = {len(self.model_parameters)} != {len(self.metrics_by_stages[stage][metric])} = len(self.metrics_by_stages[{stage}][{metric}])")
        return boolean
    
    @property
    def get_number_epochs(self) -> int:
        """
        Property that gets the number of epochs.

        :return: The number of epochs during stages.
        :rtype: int
        """
        if self._verify_size:
            return len(self.model_parameters)