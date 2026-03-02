from tqdm import tqdm
from PIL import Image
from typing import  Dict, List, Literal, Tuple, Union

import os
import json
import pickle
import subprocess
import torchvision
import numpy as np
import xml.etree.ElementTree as ET

def prepare_tiny_imagenet(path: str, delete: bool = False) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[int]], list[float], int, Dict[str, int], Literal['tiny-imagenet200'], Literal['classification']]:
    """
    Method that prepares the "classification" tiny-imagenet200 dataset.

    :param str **path**: Path where the dataset will be stored after the download.
    :param bool **delete**: Boolean that allows to delete the dataset at the end of the preparation. Set to `False`.
    :return: Train set, validation set, weights, number of classes, classes with their id, dataset name and dataset type.
    :rtype: Tuple[Dict[str, List[np.ndarray]], Dict[str, List[int]], list[float], int, Dict[str, int], Literal['tiny-imagenet200'], Literal['classification']]
    """
    assert isinstance(delete, bool), f"delete has to be an {bool} instance, not {type(delete)}."
    if not os.path.exists(os.path.join(path, "tiny-imagenet-200")):
        zip_path = os.path.join(path, "tiny-imagenet-200.zip")

        os.makedirs(path, exist_ok=True)

        subprocess.run(["wget", "-O", zip_path, "http://cs231n.stanford.edu/tiny-imagenet-200.zip"])
        subprocess.run(["unzip", zip_path, "-d", path])

        os.remove(zip_path)

    dataset_path = os.path.join(path, "tiny-imagenet-200")

    _categories = {}
    with open(os.path.join(dataset_path, "wnids.txt"), "r") as file:
        for enu, line in enumerate(file.readlines()):
            _categories[line.split("\n")[0]] = enu

    categories = {}
    with open(os.path.join(dataset_path, "words.txt"), "r") as file:
        for line in file.readlines():
            split = line.split("	")
            label_id, label_name = split[0], split[1]
            if label_id in _categories:
                categories[label_name.replace("\n", "")] = _categories[label_id]

    # Train
    _n_total_imgs = 0

    weights = []
    train = {"images": [], "labels": []}
    for enu, category in enumerate(tqdm(os.listdir(os.path.join(dataset_path, "train")), leave=True)):
        with open(os.path.join(dataset_path, "train", category, f"{category}_boxes.txt"), "r") as file:
            _bboxes_img = file.readlines()

        weights.append(0)
        for bbox_img in _bboxes_img:
            image_name = bbox_img.replace("\t", " ").split(" ")[0]
            image = Image.open(os.path.join(dataset_path, "train", category, "images", image_name), mode="r").convert("RGB")
            train["images"].append(np.array(image))
            train["labels"].append(_categories[category])

            weights[enu] += 1
            _n_total_imgs += 1
    
    for i in range(len(weights)):
        weights[i] /= _n_total_imgs

    # Validation
    validation = {"images": [], "labels": []}
    with open(os.path.join(dataset_path, "val", "val_annotations.txt"), "r") as file:
        _bboxes_img = file.readlines()

    for bbox_img in tqdm(_bboxes_img, leave=True):
        _info = bbox_img.replace("\t", " ").split(" ")
        image_name, category = _info[0], _info[1]
        image = Image.open(os.path.join(dataset_path, "val", "images", image_name), mode="r").convert("RGB")
        validation["images"].append(np.array(image))
        validation["labels"].append(_categories[category])
            
    if delete:
        os.remove(os.path.join(path, "tiny-imagenet-200"))

    return train, validation, weights, len(categories), categories, "tiny-imagenet200", "classification"

def prepare_cifar10(path: str, delete: bool = False) -> Tuple[dict[str, List[np.ndarray]], dict[str, List[int]], list[float], int, Dict[str, int], Literal['cifar-10'], Literal['classification']]:
    """
    Method that prepares the "classification" tiny-imagenet200 dataset.

    :param str **path**: Path where the dataset will be stored after the download.
    :param bool **delete**: Boolean that allows to delete the dataset at the end of the preparation. Set to `False`.
    :return: Train set, validation set, weights, number of classes, classes with their id, dataset name and dataset type.
    :rtype: Tuple[dict[str, List[np.ndarray]], dict[str, List[int]], list[float], int, Dict[str, int], Literal['cifar-10'], Literal['classification']]
    """
    assert isinstance(delete, bool), f"delete has to be an {bool} instance, not {type(delete)}."

    torchvision.datasets.CIFAR10(root=path, download=True)
    try:
        os.remove(os.path.join(path, "cifar-10-python.tar.gz"))
    except:
        pass

    train = {"images": [], "labels": []}
    validation = {"images": [], "labels": []}
    
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding="bytes")
        
        return dict

    dataset_path = os.path.join(path, "cifar-10-batches-py")

    categories = dict([(category.decode("utf-8"), enu) for enu, category in enumerate(unpickle(os.path.join(dataset_path, "batches.meta"))[b"label_names"])])
    num_classes = len(unpickle(os.path.join(dataset_path, "batches.meta"))[b"label_names"])

    _n_total_imgs = 0

    for file in tqdm(os.listdir(dataset_path), leave=True):
        add_train = True

        if file.find(".") != -1:
            continue

        if file.find("test") != -1:
            add_train = False
            
        d = unpickle(os.path.join(dataset_path, file))

        weights = [0] * num_classes

        _n_total_imgs += 1
            
        for data, label in zip(d[bytes("data", encoding="utf-8")], d[bytes("labels", encoding="utf-8")]):
            R = data[:1024].reshape(32, 32, 1)
            G = data[1024:1024*2].reshape(32, 32, 1)
            B = data[1024*2:1024*3].reshape(32, 32, 1)

            image = np.concat((R, G, B), axis=-1)
            if add_train:
                train["images"].append(image)
                train["labels"].append(label)
                weights[label] += 1
            else:
                validation["images"].append(image)
                validation["labels"].append(label)

    for i in range(len(weights)):
        weights[i] /= _n_total_imgs

    if delete:
        os.remove(os.path.join(path, "cifar-10-batches-py"))
    return train, validation, weights, num_classes, categories, "cifar-10", "classification"

def prepare_vocdetection(path: str, year: Literal["2007", "2012"] = "2007", box_format: Literal["xyxy", "xywh", "xcycwh"] = "xywh") -> Union[Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]], List[float], int, Dict[str, int], Literal['pascalvoc2007'], Literal['object_detection']], Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]], List[float], int, Dict[str, int], Literal['pascalvoc2012'], Literal['object_detection']]]:
    """
    Method that prepares the "classification" tiny-imagenet200 dataset.

    :param str **path**: Path where the dataset is stored.
    :param Literal["2007", "2012"] **year**: Version of the dataset.
    :param Literal["xyxy", "xywh", "xcycwh"] **box_format**: Format of bounding boxes. Set to `"xywh"`.
    :return: Train set, validation set, weights, number of classes, classes with their id, dataset name and dataset type.
    :rtype: Union[Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]], List[float], int, Dict[str, int], Literal['pascalvoc2007'], Literal['object_detection']], Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]], List[float], int, Dict[str, int], Literal['pascalvoc2012'], Literal['object_detection']]]
    """
    assert year in ["2007", "2012"], f"year has to be in [\"2007\", \"2012\"]."
    assert box_format in ["xyxy", "xywh", "xcycwh"], f"box_format has to be in [\"xyxy\", \"xywh\", \"xcycwh\"]."

    if year == "2007":
        with open(os.path.join(path, "pascalvoc2007/pascal_voc/pascal_train2007.json"), "r") as file:
            train_data = json.load(file)
        with open(os.path.join(path, "pascalvoc2007/pascal_voc/pascal_val2007.json"), "r") as file:
            val_data = json.load(file)

        categories = dict([(c["name"], c["id"]) for c in train_data["categories"]])
        weights = [0] * len(categories)

        def _treatment_2007(data, update_weights: bool = True):
            images = {}
            bboxes = {}
            labels = {}
            for annotation_data in data["annotations"]:
                x, y, w, h = annotation_data["bbox"]

                if box_format == "xyxy":
                    bbox = [x, y, x + w, y + h]
                elif box_format == "xywh":
                    bbox = [x, y, w, h]
                elif box_format == "xcycwh":
                    bbox = [x + (w / 2), y + (h / 2), w, h]

                if annotation_data["image_id"] not in bboxes.keys():
                    bboxes[annotation_data["image_id"]] = [bbox]
                    labels[annotation_data["image_id"]] = [annotation_data["category_id"]-1]
                else:
                    bboxes[annotation_data["image_id"]].append(bbox)
                    labels[annotation_data["image_id"]].append(annotation_data["category_id"]-1)

                if update_weights:
                    weights[annotation_data["category_id"]-1] += 1

            images_path = os.path.join(path, "pascalvoc2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/")
            for img_id in bboxes:
                n = len(str(img_id))
                file = f"{"0" * (6 - n)}{img_id}.jpg"
                image = np.array(Image.open(os.path.join(images_path, file)))
                images[img_id] = image
                H, W, _ = image.shape

                boxes = bboxes[img_id]
                new_boxes = []
                for box in boxes:
                    new_boxes.append([box[0] / W, box[1] / H, box[2] / W, box[3] / H])

                bboxes[img_id] = new_boxes
                
            return {"images": images, "bboxes": bboxes, "labels": labels}
        return _treatment_2007(train_data), _treatment_2007(val_data, False), weights, len(categories), categories, f"pascalvoc{year}", "object_detection"
    else:
        with open(os.path.join(path, "pascalvoc2012/ImageSets/Main/train.txt"), "r") as file:
            train_data = file.readlines()
        with open(os.path.join(path, "pascalvoc2012/ImageSets/Main/val.txt"), "r") as file:
            val_data = file.readlines()

        train_data = [file.replace("\n", "") for file in train_data]
        val_data = [file.replace("\n", "") for file in val_data]

        categories = {}
        weights = [0] * 20
        def _treatment_2012(data, update_weights: bool = True):
            images = {}
            bboxes = {}
            labels = {}
            
            for file in tqdm(os.listdir(os.path.join(path, "pascalvoc2012/Annotations/")), leave=True):
                if file.split(".")[0] in data:
                    tree = ET.parse(os.path.join(path, "pascalvoc2012/Annotations", file))
                    root = tree.getroot()

                    width = int(root.find("size/width").text)
                    height = int(root.find("size/height").text)

                    for obj in root.findall("object"):
                        label = obj.find("name").text
                        if label not in categories.keys():
                            categories[label] = len(categories)

                        bbox = obj.find("bndbox")
                        xmin = float(bbox.find("xmin").text)
                        ymin = float(bbox.find("ymin").text)
                        xmax = float(bbox.find("xmax").text)
                        ymax = float(bbox.find("ymax").text)

                        if box_format == "xyxy":
                            bbox = [xmin / width, ymin / height, xmax / width, ymax / height]
                        elif box_format == "xywh":
                            bbox = [xmin / width, ymin / height, (xmax - xmin) / width, (ymax - ymin) / height]
                        elif box_format == "xcycwh":
                            w = xmax - xmin
                            h = ymax - ymin
                            bbox = [(xmin + (w / 2)) / width, (ymin + (h / 2)) / height, w / width, h / height]

                        id = file.split(".")[0].split("_")[1]
                        if id not in bboxes.keys():
                            bboxes[id] = [bbox]
                            labels[id] = [categories[label]]
                        else:
                            bboxes[id].append(bbox)
                            labels[id].append(categories[label])
                        
                        if update_weights:
                            weights[categories[label]] += 1

                    image = np.array(Image.open(os.path.join(path, "pascalvoc2012/JPEGImages", f"{file.split(".")[0]}.jpg")))
                    images[id] = image
            return {"images": images, "bboxes": bboxes, "labels": labels}

        return _treatment_2012(train_data), _treatment_2012(val_data, False), weights, len(categories), categories, f"pascalvoc{year}", "object_detection"

def prepare_facedetection(path: str, size: Union[int, Tuple[int, int]], box_format: Literal["xyxy", "xywh", "xcycwh"] = "xywh") -> Union[Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]], List[float], int, Dict[str, int], Literal['pascalvoc2007'], Literal['object_detection']], Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]], List[float], int, Dict[str, int], Literal['facedetection'], Literal['object_detection']]]:
    """
    Method that prepares the "object detection" Face Detection dataset.

    :param str **path**: Path where the dataset is stored.
    :param Union[int, Tuple[int, int]] **size**: Size of images.
    :param Literal["xyxy", "xywh", "xcycwh"] **box_format**: Format of bounding boxes. Set to `"xywh"`.
    :return: Train set, validation set, weights, number of classes, classes with their id, dataset name and dataset type.
    :rtype: Union[Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]], List[float], int, Dict[str, int], Literal['pascalvoc2007'], Literal['object_detection']], Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]], List[float], int, Dict[str, int], Literal['facedetection'], Literal['object_detection']]]
    """
    assert isinstance(size, int) or isinstance(size, tuple), f"size has to be an {int} or {tuple} instance, not {type(size)}."
    if isinstance(size, int):
        size = (size, size)

    categories = {"face": 0}

    def _positive(value: float):
        if value <= 0.:
            return 0.01
        elif value >= 1.:
            return 0.99
        else:
            return value

    def _treatment(set: str):
        images_path = os.path.join(path, f"facedetection/images/{set}")
        labels_path = os.path.join(path, f"facedetection/labels/{set}")
        images = {}
        bboxes = {}
        labels = {}

        images_files = sorted(os.listdir(images_path))
        labels_files = sorted(os.listdir(labels_path))
        for image_file, label_file in tqdm(zip(images_files, labels_files), total=len(os.listdir(images_path)), leave=True):
            image_id = str(os.path.basename(image_file).split(".")[0])
            image = np.array(Image.open(os.path.join(images_path, image_file)).resize(size=size).convert("RGB"))
            images[image_id] = image

            with open(os.path.join(labels_path, label_file), "r") as file:
                lines = file.readlines()
                if lines != []:
                    for elements in lines:
                        elements = elements.split(" ")
                        xc = _positive(float(elements[1]))
                        yc = _positive(float(elements[2]))
                        w = _positive(float(elements[3]))
                        h = _positive(float(elements[4][:-1]))

                        if box_format == "xyxy":
                            bbox = [xc - (w / 2), yc - (h / 2), xc + (w / 2), yc + (h / 2)]
                        elif box_format == "xywh":
                            bbox = [xc - (w / 2), yc - (h / 2), w, h]
                        elif box_format == "xcycwh":
                            bbox = [xc, yc, w, h]

                        if image_id not in bboxes.keys():
                            bboxes[image_id] = [bbox]
                            labels[image_id] = [int(elements[0])]
                        else:
                            bboxes[image_id].append(bbox)
                            labels[image_id].append(int(elements[0]))
                else:
                    bboxes[image_id] = []
                    labels[image_id] = [None]

        return {"images": images, "bboxes": bboxes, "labels": labels}
                
    return _treatment("train"), _treatment("validation"), {0: 1}, len(categories), categories, "facedetection", "object_detection"