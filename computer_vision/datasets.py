from PIL import Image
from random import randint
from .data_augmentation import DataAugmentation
from torch.utils.data import Dataset as dataset
from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import os
import json
import torch
import xml.etree.ElementTree as ET

class Compose(object):
    """
    Base class for compose: Applies different transformations to datasets objets.
    """
    def __init__(self, transforms: List[Callable]) -> None:
        """
        Initializes the Compose class.
        
        :param List[Callable] **transforms**: Transformations to apply to objects (images).
        """
        self._transforms = transforms

    def __call__(self, obj: Union[Image.Image]) -> torch.Tensor:
        """
        Built-in Python method that allows to call an instance of the class. Applies the transformations.
        
        :param Union[Image.Image] **obj**: Object to transform.
        :return: Object transformed.
        :rtype: Tensor
        """
        for transform in self._transforms:
            obj = transform(obj)
        return obj

class Dataset(dataset):
    """
    Base dataset class.
    """
    def __init__(self, dataset_name: str, dataset_path: str, height: int, width: int, transformations: Compose, box_format: Literal["xyxy", "xywh", "xcycwh"] = "xywh", model_normalized: bool = True, type_set: Literal["train", "validation"] = "train", data_aug: bool = False, **kwargs):
        """
        Initializes the Dataset class.

        :param str **dataset_name**: Path of the dataset.
        :param str **dataset_path**: Path of the dataset.
        :param int **height**: Height for images.
        :param int **width**: Width for images.
        :param Compose **transformations**: Transformations to apply to images.
        :param Literal["xyxy", "xywh", "xcycwh"] **box_format**: Box format for bounding boxes. Set to `xywh`.
        :param bool **model_normalized**: Boolean that specifies if the model needs normalized input. Set to `True`.
        :param str **type_set**: Type of the set. Set to `train`.
        :param bool **data_aug**: Data augmentation. Set to `False`.
        """
        assert isinstance(dataset_name, str), f"dataset_name has to be a {str} instance, not {type(dataset_name)}."
        assert isinstance(dataset_path, str), f"dataset_path has to be a {str} instance, not {type(dataset_path)}."
        assert isinstance(height, int) and height > 0, f"height has to be a positive integer, not {type(height)}."
        assert isinstance(width, int) and width > 0, f"width has to be a positive integer, not {type(width)}."
        assert isinstance(transformations, Compose), f"transformation has to be a {Compose} instance, not {type(transformations)}."
        assert box_format in ["xyxy", "xywh", "xcycwh"], f'box_format has to be in ["xyxy", "xywh", "xcycwh"], not equal to {box_format}.'
        assert isinstance(model_normalized, bool), f"dataset_normalized has to be a {bool} instance, not {type(model_normalized)}."
        assert isinstance(type_set, str) and type_set in ["train", "validation"], f"type_set has to be a {str} instance and belong to ['train', 'validation'], not {type(type_set)}."
        super().__init__()
        
        self._base10 = True
        self._images_selection = False
        self._model_normalized = model_normalized

        self._dataset_path, self._annotations, self.weights, self.num_classes, self.categories, self.dataset_name, self.dataset_box_format, self.mode, self._dataset_normalized = self._get_dataset(dataset_path, dataset_name, type_set, **kwargs)
        
        if self._images_selection:
            if self._base10:
                self._images_files = [file for file in os.listdir(self._dataset_path) if int(file.split(".")[0]) in self._annotations["bboxes"].keys()]
            else:
                self._images_files = [file for file in os.listdir(self._dataset_path) if file.split(".")[0] in self._annotations["bboxes"].keys()]
        else:
            self._images_files = os.listdir(self._dataset_path)

        self._height = height
        self._width = width
        self._transformations = transformations
        self._box_format = box_format

        self._data_aug = data_aug
        if data_aug:
            self._data_aug_factor = 2
            self._dataugmentation = DataAugmentation(box_format=self._box_format, 
                                                     brightness=0.25,
                                                     contrast=randint(20, 200) / 100,
                                                     saturation=randint(35, 200) / 100)
            self._indexes = dict([(i, j) for i, j in zip(range(len(self._images_files), int(len(self._images_files) * self._data_aug_factor)), range(len(self._images_files)))])

    def _get_dataset(self, dataset_path: str, dataset_name: str, type_set: Literal["train", "validation"], **kwargs) -> Tuple[str, Dict[int, List[List[float | int]]], List[float], int, Dict[str, int], str, Literal["xyxy", "xywh", "xcycwh"], Literal["classification", "object_detection"]]:
        """
        Method that gets a chosen dataset.

        :param str **dataset_path**: Path of the dataset.
        :param str **dataset_name**: Name of the dataset.
        :param str **type_set**: Type of the set.
        :return: Dataset information.
        :rtype: Tuple[str, Dict[int, List[List[float | int]]], List[float], int, Dict[int, str], str, Literal["xyxy", "xywh", "xcycwh"], Literal["classification", "object_detection"]]
        """
        if dataset_name == "coco2017":
            self._images_selection = True

            if "categories_to_keep" in kwargs.keys():
                categories_to_keep = kwargs["categories_to_keep"]
            else:
                categories_to_keep = None

            if "limit_images" in kwargs.keys():
                limit_images = kwargs["limit_images"]
            else:
                limit_images = -1
            return coco2017(dataset_path, type_set, categories_to_keep, limit_images)
        elif dataset_name == "facedetection":
            self._base10 = False
            return facedetection(dataset_path, type_set)
        elif dataset_name == "modified_coco2017":
            self._images_selection = True

            categories_to_keep = ["person"]
            if "limit_images" in kwargs.keys():
                limit_images = kwargs["limit_images"]
            else:
                limit_images = -1
            return coco2017(dataset_path, type_set, categories_to_keep, limit_images)
        elif dataset_name == "pascalvoc2007":
            self._base10 = False
            self._images_selection = True
            return vocdetection(dataset_path, type_set, year="2007")
        elif dataset_name == "pascalvoc2012":
            self._base10 = False
            self._images_selection = True
            return vocdetection(dataset_path, type_set, year="2012")

    def __len__(self) -> int:
        """
        Built-in Python method that returns the length of the object.

        :return: Number of images in the dataset.
        :rtype: int
        """
        if self._data_aug:
            self._dataset_length = len(self._images_files)
            return int(len(self._images_files) * self._data_aug_factor)
        else:
            return len(self._images_files)
    
    def _verify_value(self, value: float) -> float:
        """
        Method that verifies value range.

        :param float **value**: Value to verify.
        :return: Float in [0;1]
        :rtype: float
        """
        if value > 0. and value < 1.:
            return value
        elif value >= 1.:
            return 0.99
        else:
            return 0.01
    
    def _normalization(self, bbox_label: int, bbox: List[float | int]) -> List[float | int]:
        """
        Method that normalizes a bounding box.

        :param int **bbox_label**: Bounding box label.
        :param int **bbox**: Bounding box label.
        :return: Bounding box normalized.
        :rtype: List[float | int]
        """
        if bbox == []:
            return bbox

        x, y, w, h = bbox
        if self._model_normalized:
            if self._box_format == "xyxy":
                return [bbox_label, x, y, (x + w), (y + h)]
            elif self._box_format == "xywh":
                return [bbox_label, x, y, w, h]
            else:
                return [bbox_label, (x + (w / 2)), (y + (h / 2)), w, h]
        else:
            if self._box_format == "xyxy":
                return [bbox_label, x * self._width, y * self._height, (x + w) * self._width, (y + h) * self._height]
            elif self._box_format == "xywh":
                return [bbox_label, x * self._width, y * self._height, w * self._width, h * self._height]
            else:
                return [bbox_label, (x + (w / 2)) * self._width, (y + (h / 2)) * self._height, w * self._width, h * self._height]
    
    def _bbox_transformation(self, bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
        """
        Method that transforms bounding boxes to a define box format.

        :param List[float] **bbox**: Bounding boxes.
        :param int **image_size**: Size of the bounding box image.
        :return: Bounding boxes with a specified format.
        :rtype: List[float]
        """
        bbox_label = bbox[0]
        bbox = bbox[1:]
        if self.dataset_box_format == "xyxy":
            x = bbox[0]
            y = bbox[1]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        elif self.dataset_box_format == "xywh":
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
        elif self.dataset_box_format == "xcycwh":
            x = bbox[0] - (bbox[2] / 2)
            y = bbox[1] - (bbox[3] / 2)
            w = bbox[2]
            h = bbox[3]

        if self._dataset_normalized:
            bbox = [self._verify_value(x), self._verify_value(y), self._verify_value(w), self._verify_value(h)]
        else:
            bbox = [self._verify_value(x / image_size[0]), self._verify_value(y / image_size[1]), self._verify_value(w / image_size[0]), self._verify_value(h / image_size[1])]

        if self._box_format == "xyxy":
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        elif self._box_format == "xywh":
            bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
        elif self._box_format == "xcycwh":
            bbox = [bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3] / 2), bbox[2], bbox[3]]
        return self._normalization(bbox_label, bbox)
                
    def _specified_getitem(self, bboxes: List[List[float | int]]) -> Any:
        """
        Method that generates if necessary a specific item for the model.

        :param bboxes: List[List[float | int]] **bboxes**: Bounding boxes.
        :return: Model's ground_truths.
        :rtype: Any
        """
        return None
                
    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], Union[Tuple[torch.Tensor, torch.Tensor, List[List[float | int]]], Tuple[torch.Tensor, None, List[List[float | int]]]]]:
        """
        Built-in Python method that allows to access an element from the object.

        :param int **index**:
        :return: In mode "classification", a tensor and a label. In mode "object_detection, a tuple of tensor of size 3.
        :rtype: Union[Tuple[torch.Tensor, int], Union[Tuple[torch.Tensor, torch.Tensor, List[List[float | int]]], Tuple[torch.Tensor, None, List[List[float | int]]]]]
        """
        _data_aug = False
        if self._data_aug:
            if index >= self._dataset_length:
                index = self._indexes[index]
                _data_aug = True

        image = Image.open(os.path.join(self._dataset_path, self._images_files[index])).resize(size=(self._width, self._height)).convert("RGB")

        if self._base10:
            idx = int(self._images_files[index].split(".")[0])
        else:
            idx = self._images_files[index].split(".")[0]

        if self.mode == "classification":
            return image, self._annotations[idx]
        elif self.mode == "object_detection":
            bboxes = []
            _bboxes = self._annotations["bboxes"][idx]
            image_size = self._annotations["sizes"][idx]

            if _data_aug:
                image, _bboxes = self._dataugmentation(image, _bboxes)
            else:
                image = self._transformations(image)

            for bbox in _bboxes:
                bboxes.append(self._bbox_transformation(bbox, image_size))
            
            targets = self._specified_getitem(bboxes)
            return image, targets, bboxes
    
def collate_fn(batch: Any) -> Any:
    """
    Specific collate function for Dataset class.

    :param Any **batch**:
    :return: Apdated Batch.
    :rtype: Any
    """
    images, targets, _batch_bboxes = zip(*batch)
    batch_bboxes = []
    for i, bboxes in enumerate(_batch_bboxes):
        for bbox in bboxes:
            batch_bboxes.append([i] + bbox)

    images = torch.stack(images, dim=0)

    if targets[0] is None:
        return images, torch.tensor(batch_bboxes, dtype=torch.float32)
    else:
        return images, (torch.stack(targets, dim=0), torch.tensor(batch_bboxes, dtype=torch.float32))
    
def coco2017(path: str, group: Literal["train", "validation"], categories_to_keep: Union[List[str], None] = None, limit_images: int = -1) -> Tuple[str, Dict[int, List[List[float | int]]], List[float], int, Dict[str, int], Literal["coco2017", "modified_coco2017"], Literal["xywh"], Literal["object_detection"], bool]:
    """
    Method that prepares "coco2017" dataset on a given set (train or validation).

    :param str **path**: Directory path that contains the dataset.
    :param Literal["train", "validation"] **group**: Set to take from the dataset.
    :param Union[List[str], None] **categories_to_keep**: Classes to keep for object detection task. Set to `None`.
    :param int **limit_images**: Limit of images to take. Set to `-1`.
    :return: Dataset.
    :rtype: Tuple[str, Dict[int, List[List[float | int]]], List[float], int, Dict[str, int], Literal["coco2017", "modified_coco2017"], Literal["xywh"], Literal["object_detection"], bool]
    """
    def _load_json(path: str, filename: str):
        with open(os.path.join(path, filename), "rb") as file:
            stage = json.load(file)
        return stage
    
    def _transform_to_filename(image_id: int):
        image_id = str(image_id)
        image_id = "0" * (12 - len(image_id)) + image_id
        return f"{image_id}.jpg"

    categories = {}
    normalized_categories = {}
    weights = [0] * 80 if categories_to_keep is None else [0] * len(categories_to_keep)

    if group == "validation":
        group = "val"

    dataset = _load_json(os.path.join(path, "coco2017/annotations"), f"instances_{group}2017.json")

    images_path = os.path.join(path, f"coco2017/{group}2017/")

    if categories_to_keep is None:
        categories = dict([(i, category["name"]) for i, category in enumerate(dataset["categories"])])
        normalized_categories = dict([(category["id"], i) for i, category in enumerate(dataset["categories"])])
    else:
        __categories = [category["name"] for category in dataset["categories"]]
        for category in categories_to_keep:
            if category not in __categories:
                raise Exception(f"{category} not in {__categories}")
        i = 0
        _categories = []
        _normalized_categories = []
        for category in dataset["categories"]:
            if category["name"] in categories_to_keep:
                _categories.append((i, category["name"]))
                _normalized_categories.append((category["id"], i))
                i += 1

        categories = dict(_categories)
        normalized_categories = dict(_normalized_categories)

    if limit_images != -1:
        images_per_class = dict([(name, 0) for _, name in categories.items()])

    ids = {}
    annotations = {"bboxes": {}, "sizes": {}}

    for annotation in dataset["annotations"]:
        if annotation["category_id"] in normalized_categories:
            if limit_images != -1:
                if images_per_class[categories[normalized_categories[annotation["category_id"]]]] >= limit_images:
                    if annotation["image_id"] not in ids:
                        images_per_class[categories[normalized_categories[annotation["category_id"]]]] += 1
                        ids.add(annotation["image_id"])
            
            weights[normalized_categories[annotation["category_id"]]] += 1

            if annotation["image_id"] not in annotations["bboxes"]:
                annotations["bboxes"][annotation["image_id"]] = [[normalized_categories[annotation["category_id"]]] + annotation["bbox"]]
            else:
                annotations["bboxes"][annotation["image_id"]].append([normalized_categories[annotation["category_id"]]] + annotation["bbox"])

            if annotation["image_id"] not in annotations["sizes"]:
                annotations["sizes"][annotation["image_id"]] = Image.open(os.path.join(images_path, _transform_to_filename(annotation["image_id"]))).size
            
    weights = [weight / sum(weights) for weight in weights]

    if len(categories) != 80:
        dataset_name = "modified_coco2017_" + "_".join([category for _, category in categories.items()])
    else:
        dataset_name = "coco2017"

    return os.path.join(path, f"coco2017/{group}2017"), annotations, weights, len(categories), dict([(value, key) for key, value in categories.items()]), dataset_name, "xywh", "object_detection", False

def facedetection(path: str, group: Literal["train", "validation"]) -> Tuple[str, Dict[int, List[List[float | int]]], List[float], int, Dict[str, int], Literal["facedetection"], Literal["xcycwh"], Literal["object_detection"], bool]:
    """
    Method that prepares the "object detection" Face Detection dataset.

    :param str **path**: Path where the dataset is stored.
    :param Literal["train", "validation"] **group**: Set to take from the dataset.
    :return: Dataset.
    :rtype: Tuple[str, Dict[int, List[List[float | int]]], List[float], int, Dict[str, int], Literal["facedetection"], Literal["xcycwh"], Literal["object_detection"], bool]
    """
    categories = {"face": 0}

    def _positive(value: float):
        if value <= 0.001:
            return 0.01
        elif value >= 1.:
            return 0.99
        else:
            return value

    annotations = {"bboxes": {}, "sizes": {}}

    images_path = os.path.join(path, f"facedetection/images/{group}/")
    labels_path = os.path.join(path, f"facedetection/labels/{group}/")
    image_files = sorted(os.listdir(images_path))
    for image_file in image_files:

        image_id = image_file.split(".")[0]

        size = Image.open(os.path.join(images_path, image_file)).size

        with open(os.path.join(labels_path, f"{image_id}.txt"), "r") as file:
            lines = file.readlines()
            if lines != []:
                for elements in lines:
                    elements = elements.split(" ")
                    xc = _positive(float(elements[1]))
                    yc = _positive(float(elements[2]))
                    w = _positive(float(elements[3]))
                    h = _positive(float(elements[4][:-1]))

                    if image_id not in annotations["bboxes"]:
                        annotations["bboxes"][image_id] = [[0, xc, yc, w, h]]
                        annotations["sizes"][image_id] = size
                    else:
                        annotations["bboxes"][image_id].append([0, xc, yc, w, h])
            else:
                annotations["bboxes"][image_id] = []
                annotations["sizes"][image_id] = size
    return images_path, annotations, [1], len(categories), categories, "facedetection", "xcycwh", "object_detection", True

def vocdetection(path: str, group: Literal["train", "validation"] = "train", year: Literal["2007", "2012"] = "2007") -> Tuple[str, Dict[int, List[List[float | int]]], List[float], int, Dict[str, int], Literal["pascalvoc2007", "pascalvoc2012"], Literal["xywh", "xyxy"], Literal["object_detection"], bool]:
    """
    Method that prepares the "classification" tiny-imagenet200 dataset.

    :param str **path**: Path where the dataset is stored.
    :param Literal["train", "validation"] **group**: Set to take from the dataset.
    :param Literal["2007", "2012"] **year**: Version of the dataset.
    :return: Dataset
    :rtype: Tuple[str, Dict[int, List[List[float | int]]], List[float], int, Dict[str, int], Literal["pascalvoc2007", "pascalvoc2012"], Literal["xywh", "xyxy"], Literal["object_detection"], bool]
    """
    assert year in ["2007", "2012"], f"year has to be in [\"2007\", \"2012\"]."

    if group == "validation":
        group = "val"

    if year == "2007":
        with open(os.path.join(path, f"pascalvoc2007/pascal_voc/pascal_{group}2007.json"), "r") as file:
            data = json.load(file)

        categories = dict([(c["name"], c["id"]) for c in data["categories"]])
        weights = [0] * len(categories)

        def id_to_filename(idx: int):
            idx = str(idx)
            return "0" * (6-(len(idx))) + idx + ".jpg"

        annotations = {"bboxes": {}, "sizes": {}}
        for annotation_data in data["annotations"]:
            if annotation_data["image_id"] not in annotations.keys():
                annotations["bboxes"][annotation_data["image_id"]] = [[annotation_data["category_id"]-1] + annotation_data["bbox"]]
                annotations["sizes"][annotation_data["image_id"]] = Image.open(os.path.join(path, "pascalvoc2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/", id_to_filename(annotation_data["image_id"]))).size
            else:
                annotations["bboxes"][annotation_data["image_id"]].append([annotation_data["category_id"]-1] + annotation_data["bbox"])

            weights[annotation_data["category_id"]-1] += 1

        return os.path.join(path, "pascalvoc2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/"), annotations, weights, len(categories), categories, f"pascalvoc{year}", "xywh", "object_detection", False
    else:
        with open(os.path.join(path, f"pascalvoc2012/ImageSets/Main/{group}.txt"), "r") as file:
            data = file.readlines()

        data = [file.replace("\n", "") for file in data]

        categories = {}
        weights = [0] * 20
        annotations = {"bboxes": {}, "sizes": {}}
        
        for file in os.listdir(os.path.join(path, "pascalvoc2012/Annotations/")):
            idx = file.split(".")[0]
            if idx in data:
                tree = ET.parse(os.path.join(path, "pascalvoc2012/Annotations", file))
                root = tree.getroot()

                size = (int(root.findall("size")[0].find("width").text), int(root.findall("size")[0].find("height").text))
                for obj in root.findall("object"):
                    label = obj.find("name").text
                    if label not in categories.keys():
                        categories[label] = len(categories)

                    bbox = obj.find("bndbox")
                    xmin = float(bbox.find("xmin").text)
                    ymin = float(bbox.find("ymin").text)
                    xmax = float(bbox.find("xmax").text)
                    ymax = float(bbox.find("ymax").text)

                    bbox = [xmin, ymin , xmax, ymax]

                    
                    if idx not in annotations["bboxes"].keys():
                        annotations["bboxes"][idx] = [[categories[label]] + bbox]
                        annotations["sizes"][idx] = size
                    else:
                        annotations["bboxes"][idx].append([categories[label]] + bbox)                    
                    
                    weights[categories[label]] += 1

        return os.path.join(path, "pascalvoc2012/JPEGImages/"), annotations, weights, len(categories), categories, f"pascalvoc{year}", "xyxy", "object_detection", False