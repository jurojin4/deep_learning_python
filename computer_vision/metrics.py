from collections import Counter
from typing import Dict, List, Literal, Tuple, Union

import torch
import torch.nn as nn

eps = 1e-9

def intersection_over_union(bboxes_preds: torch.Tensor, bboxes_gts: torch.Tensor, box_format: Literal["xyxy", "xywh", "xcycwh"] = "xywh") -> torch.Tensor:
    """
    Method that calculates IoU between bounding boxes of predictions and ground_truths.

    :param Tensor **bboxes_preds**: Predictions bounding boxes of shapes (B, 4).
    :param Tensor **bboxes_gts**: Ground Truths bounding boxes of shapes (B, 4).
    :param Literal["xyxy", "xywh", "xcycwh"] **box_format**: Format of bounding boxes. Set to `xywh`.
    :return: Intersection over Union of shapes (B, 1).
    :rtype: Tensor
    """
    if box_format == "xyxy":
        preds_x1 = bboxes_preds[..., 0:1]
        preds_y1 = bboxes_preds[..., 1:2]
        preds_x2 = bboxes_preds[..., 2:3]
        preds_y2 = bboxes_preds[..., 3:4]

        ground_truths_x1 = bboxes_gts[..., 0:1]
        ground_truths_y1 = bboxes_gts[..., 1:2]
        ground_truths_x2 = bboxes_gts[..., 2:3]
        ground_truths_y2 = bboxes_gts[..., 3:4]

    elif box_format == "xywh":
        preds_x1 = bboxes_preds[..., 0:1]
        preds_y1 = bboxes_preds[..., 1:2]
        preds_x2 = bboxes_preds[..., 0:1] + bboxes_preds[..., 2:3]
        preds_y2 = bboxes_preds[..., 1:2] + bboxes_preds[..., 3:4]

        ground_truths_x1 = bboxes_gts[..., 0:1]
        ground_truths_y1 = bboxes_gts[..., 1:2]
        ground_truths_x2 = bboxes_gts[..., 0:1] + bboxes_gts[..., 2:3]
        ground_truths_y2 = bboxes_gts[..., 1:2] + bboxes_gts[..., 3:4]
    elif box_format == "xcycwh":
        preds_x1 = bboxes_preds[..., 0:1] - (bboxes_preds[..., 2:3] / 2)
        preds_y1 = bboxes_preds[..., 1:2] - (bboxes_preds[..., 3:4] / 2)
        preds_x2 = bboxes_preds[..., 2:3] + preds_x1
        preds_y2 = bboxes_preds[..., 3:4] + preds_y1

        ground_truths_x1 = bboxes_gts[..., 0:1] - (bboxes_gts[..., 2:3] / 2)
        ground_truths_y1 = bboxes_gts[..., 1:2] - (bboxes_gts[..., 3:4] / 2)
        ground_truths_x2 = bboxes_gts[..., 2:3] + ground_truths_x1
        ground_truths_y2 = bboxes_gts[..., 3:4] + ground_truths_y1

    x1 = torch.max(preds_x1, ground_truths_x1)
    y1 = torch.max(preds_y1, ground_truths_y1)
    x2 = torch.min(preds_x2, ground_truths_x2)
    y2 = torch.min(preds_y2, ground_truths_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    preds_area = abs((preds_x2 - preds_x1) * (preds_y2 - preds_y1))
    ground_truths_area = abs((ground_truths_x2 - ground_truths_x1) * (ground_truths_y2 - ground_truths_y1))

    return intersection / (preds_area + ground_truths_area - intersection + eps)    

class Metric(nn.Module):
    """
    Base metric class.
    """
    name: str
    def __init__(self):
        """
        Initiliazes the Metric class.
        """
        super().__init__()

    def _prepare_gts_and_preds(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method that prepares ground truths and predictions to have same shape for loss calculation.

        :param Tensor **predictions**: Tensor containing the model predictions.
        :param Tensor **ground_truths**: Tensor containing the values to predict.
        :return: Predictions and ground truths with same shape.
        :rtype: Tensor
        """
        if len(ground_truths.shape) < len(predictions.shape):
            ground_truths = ground_truths.unsqueeze(-1)

        if ground_truths.shape[-1] == predictions.shape[-1]:
            if predictions.dtype == bool:
                return predictions.type(int), ground_truths
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
    
class Accuracy(Metric):
    """
    Accuracy metric class.
    """
    def __init__(self):
        """
        Initializes the Accuracy class.
        """
        super().__init__()
        super().__setattr__("name", "accuracy")

    def forward(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> float:
        """
        Method that calculates accuracy between predictions and ground_truths.

        :param Tensor **predictions**: Model predictions.
        :param Tensor **ground_truths**: Ground Truths.
        :return: Accuracy, value between 0 to 1.
        :rtype: float
        """
        predictions, ground_truths = self._prepare_gts_and_preds(predictions, ground_truths)

        n = ground_truths.shape[0]
        return (ground_truths == predictions).sum().item() / n

class Precision(Metric):
    """
    Precision metric class.
    """
    def __init__(self, num_classes: int, detail: bool = False):
        """
        Initializes the Precision class.

        :param int **num_classes**: Number of classes.
        :param bool **detail**: Boolean that adds precision per class. Set to `False.
        """
        super().__init__()
        super().__setattr__("name", "precision")
        self.num_classes = num_classes
        self.detail = detail

    def forward(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> Union[Tuple[Tuple[float, Dict[int, Union[float, None]]], float], float]:
        """
        Method that calculates precision between predictions and ground_truths. if detail is `True` then precision per class is added.

        :param Tensor **predictions**: Model predictions.
        :param Tensor **ground_truths**: Ground Truths.
        :return: Average precision (value between 0 to 1) and if detail is `True` precision per class.
        :rtype: Union[Tuple[Tuple[float, Dict[int, Union[float, None]]], float], float]
        """ 
        if self.detail:
            return self._forward_d(predictions, ground_truths)
        else:
            return self._forward_nd(predictions, ground_truths)

    def _forward_nd(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> float:
        """
        Method that calculates precision between predictions and ground_truths.

        :param Tensor **predictions**: Model predictions.
        :param Tensor **ground_truths**: Ground Truths.
        :return: Average precision (value between 0 to 1) and precision per class.
        :rtype: float
        """ 
        predictions, ground_truths = self._prepare_gts_and_preds(predictions, ground_truths)

        classes_present = ground_truths.unique()

        average_precision = 0

        for class_label in classes_present:
            true_positive = ((predictions == class_label.item()) & (ground_truths == class_label.item())).sum().item()
            predicted_positive = (predictions == class_label.item()).sum().item()

            average_precision += true_positive / predicted_positive if predicted_positive != 0 else 0
        
        average_precision /= len(classes_present)
        return average_precision
    
    def _forward_d(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> Tuple[float, Dict[int, Union[float, None]]]:
        """
        Method that calculates precision between predictions and ground_truths in a detail way.

        :param Tensor **predictions**: Model predictions.
        :param Tensor **ground_truths**: Ground Truths.
        :return: Average precision (value between 0 to 1) and precision per class.
        :rtype: Tuple[float, Dict[int, Union[float, None]]]
        """
        predictions, ground_truths = self._prepare_gts_and_preds(predictions, ground_truths)

        classes_present = ground_truths.unique()

        precision_per_class = dict([(int(label.item()), None) for label in ground_truths.unique()])

        for class_label in classes_present:
            true_positive = ((predictions == class_label.item()) & (ground_truths == class_label.item())).sum().item()
            predicted_positive = (predictions == class_label.item()).sum().item()

            precision = true_positive / predicted_positive if predicted_positive != 0 else 0
            precision_per_class[int(class_label)] = precision
        
        average_precision = sum([item for _, item in precision_per_class.items() if item is not None]) / len(classes_present)
        return average_precision, precision_per_class

class Recall(Metric):
    """
    Recall metric class.
    """
    def __init__(self, num_classes: int, detail: bool = False):
        """
        Initializes the Recall class.

        :param int **num_classes**: Number of classes.
        :param bool **detail**: Boolean that adds precision per class. Set to `False.
        """
        super().__init__()
        super().__setattr__("name", "recall")
        self.num_classes = num_classes
        self.detail = detail

    def forward(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> Union[Tuple[Tuple[float, Dict[int, Union[float, None]]], float], float]:
        """
        Method that calculates recall between predictions and ground_truths. if detail is `True` then recall per class is added.

        :param Tensor **predictions**: Model predictions.
        :param Tensor **ground_truths**: Ground Truths.
        :return: Average recall (value between 0 to 1) and if detail is `True` precision per class.
        :rtype: Union[Tuple[Tuple[float, Dict[int, Union[float, None]]], float], float]
        """ 
        if self.detail:
            return self._forward_d(predictions, ground_truths)
        else:
            return self._forward_nd(predictions, ground_truths)

    def _forward_nd(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> float:
        """
        Method that calculates recall between predictions and ground_truths.

        :param Tensor **predictions**: Model predictions.
        :param Tensor **ground_truths**: Ground Truths.
        :return: Average recall (value between 0 to 1).
        :rtype: float
        """
        predictions, ground_truths = self._prepare_gts_and_preds(predictions, ground_truths)
        
        classes_present = ground_truths.unique()

        recall = 0.

        for class_label in classes_present:
            true_positive = ((predictions == class_label.item()) & (ground_truths == class_label.item())).sum().item()
            actual_positive = (ground_truths == class_label.item()).sum().item()
            
            recall += true_positive / actual_positive if actual_positive != 0 else 0

        recall /= len(classes_present)
        return recall

    def _forward_d(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> Tuple[float, Dict[int, Union[float, None]]]:
        """
        Method that calculates recall between predictions and ground_truths in a detail way.

        :param Tensor **predictions**: Model predictions.
        :param Tensor **ground_truths**: Ground Truths.
        :return: Average recall (value between 0 to 1) and recall per class.
        :rtype: Tuple[float, Dict[int, Union[float, None]]]
        """
        predictions, ground_truths = self._prepare_gts_and_preds(predictions, ground_truths)

        recall_per_class = dict([(int(label.item()), None) for label in ground_truths.unique()])

        classes_present = ground_truths.unique()

        for class_label in classes_present:
            true_positive = ((predictions == class_label.item()) & (ground_truths == class_label.item())).sum().item()
            actual_positive = (ground_truths == class_label.item()).sum().item()
            recall = true_positive / actual_positive if actual_positive != 0 else 0
            recall_per_class[int(class_label)] = recall

        average_recall = sum([item for _, item in recall_per_class.items() if item is not None]) / len(classes_present)
        return average_recall, recall_per_class

class F1_Score(Metric):
    """
    F1-Score metric class.
    """
    def __init__(self, num_classes: int, detail: bool = False):
        """
        Initializes the F1_score class.

        :param int **num_classes**: Number of classes.
        :param bool **detail**: Boolean that adds precision per class. Set to `False.
        """
        super().__init__()
        super().__setattr__("name", "f1_score")
        self.num_classes = num_classes
        self.precision = Precision(num_classes, detail=detail)
        self.recall = Recall(num_classes, detail=detail)
        self.detail = detail

    def forward(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> Union[Tuple[Tuple[float, Dict[int, Union[float, None]]], float], float]:
        """
        Method that calculates f1-score between predictions and ground_truths. if detail is `True` then f1-score per class is added.
        
        :param Tensor **predictions**: Model predictions.
        :param Tensor **ground_truths**: Ground Truths.
        :return: Average f1-score (value between 0 to 1) and if detail is `True` precision per class.
        :rtype: Union[Tuple[Tuple[float, Dict[int, Union[float, None]]], float], float]
        """ 
        if self.detail:
            return self._forward_d(predictions, ground_truths)
        else:
            return self._forward_nd(predictions, ground_truths)

    def _forward_nd(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> float:
        """
        Method that calculates f1-score between predictions and ground_truths.

        :param Tensor **predictions**: Model predictions.
        :param Tensor **ground_truths**: Ground Truths.
        :return: Average F1-score (value between 0 to 1).
        :rtype: float
        """
        predictions, ground_truths = self._prepare_gts_and_preds(predictions, ground_truths)

        avg_prec = self.precision(predictions, ground_truths)
        avg_rec = self.recall(predictions, ground_truths)  

        average_f1 = 2 * (avg_prec * avg_rec) / (avg_prec + avg_rec) if avg_prec != 0 and avg_rec != 0 else 0

        return average_f1
        
    def _forward_d(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> Tuple[float, Dict[int, Union[float, None]]]:
        """
        Method that calculates f1-score between predictions and ground_truths in a detail way.

        :param Tensor **predictions**: Model predictions.
        :param Tensor **ground_truths**: Ground Truths.
        :return: Average F1-score (value between 0 to 1) and F1-score per class.
        :rtype: Tuple[float, Dict[int, Union[float, None]]]
        """
        predictions, ground_truths = self._prepare_gts_and_preds(predictions, ground_truths)

        avg_prec, prec_per_class = self.precision(predictions, ground_truths)
        avg_rec, rec_per_class = self.recall(predictions, ground_truths)
        average_f1 = 2 * (avg_prec * avg_rec) / (avg_prec + avg_rec) if avg_prec != 0 and avg_rec != 0 else 0

        f1_per_class = {}
        for (key, prec), (_, rec) in zip(prec_per_class.items(), rec_per_class.items()):
            f1_per_class[key] = (2 * (prec * rec) / (prec + rec)) if (prec + rec) != 0 else 0

        return average_f1, f1_per_class

class Mean_Average_Precision(Metric):
    """
    Mean Average Precision metric class.
    """
    def __init__(self, num_classes: int, batch_size: int, iou_threshold: float = 0.5, detail: bool = False):
        """
        Initializes Mean Average Precision class.

        :param int **num_classes**: Number of classes.
        :param int **batch**: Size of the batch.
        :param float **iou_threshold**: Threshold for IoU. Set to `0.5`.
        :param bool **detail**: Boolean that adds precision per class. Set to `False.
        """
        super().__init__()
        super().__setattr__("name", f"mAP@{int(iou_threshold * 100)}")
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._iou_threshold = iou_threshold
        self.detail = detail

    def forward(self, predictions: List[List[List[torch.Tensor]]], ground_truths: List[List[List[torch.Tensor]]]) -> Union[Tuple[Tuple[float, Dict[int, Union[float, None]]], float], float]:
        """
        Method that calculates mAP between predictions and ground_truths. if detail is `True` then f1-score per class is added.
        
        :param List[List[List[torch.Tensor]]] **predictions**: Model predictions.
        :param List[List[List[torch.Tensor]]] **ground_truths**: Ground Truths.
        :return: Average mAP (value between 0 to 1) and if detail is `True` mAP per class.
        :rtype: Union[Tuple[Tuple[float, Dict[int, Union[float, None]]], float], float]
        """ 
        if self.detail:
            return self._forward_d(predictions, ground_truths)
        else:
            return self._forward_nd(predictions, ground_truths)

    def _forward_nd(self, predictions: List[List[List[torch.Tensor]]], ground_truths: List[List[List[torch.Tensor]]]) -> float:
        """
        Method that calculates mAP between predictions and ground_truths.

        :param List[List[List[torch.Tensor]]] **predictions**: Bounding boxes of predictions.
        :param List[List[List[torch.Tensor]]] **ground_truths**: Bounding boxes of ground_truths.
        :return: Average mAP (value between 0 to 1).
        :rtype: float
        """
        mean_average_precision = []

        for preds_bboxes, gts_bboxes in zip(predictions, ground_truths):
            if len(gts_bboxes) == 0:
                print("continue")
                continue
            
            amount_bboxes = Counter([gt[0] for gt in gts_bboxes])
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)

            if self._num_classes > 1:              
                preds_bboxes.sort(key=lambda box: box[2], reverse=True)
            else:
                preds_bboxes.sort(key=lambda box: box[1], reverse=True)

            true_positive = torch.zeros((len(preds_bboxes)))
            false_positive = torch.zeros((len(preds_bboxes)))

            for pred_idx, pred_bbox in enumerate(preds_bboxes):
                best_iou = 0

                for gt_idx, gt in enumerate(gts_bboxes):
                    if self._num_classes > 1:
                        if pred_bbox[1] != gt[1]:
                            continue
                    iou = intersection_over_union(pred_bbox[3:], gt[3:]) if self._num_classes > 1 else intersection_over_union(pred_bbox[2:], gt[2:])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou > self._iou_threshold:
                    if amount_bboxes[best_gt_idx] == 0:
                        true_positive[pred_idx] = 1
                        amount_bboxes[best_gt_idx] = 1
                    else:
                        false_positive[pred_idx] = 1
                else:
                    false_positive[pred_idx] = 1

            true_positive_cumsum = true_positive.cumsum(dim=0)
            false_positive_cumsum = false_positive.cumsum(dim=0)

            recalls = true_positive_cumsum / (len(gts_bboxes) + 1e-16)
            precisions = true_positive_cumsum / (true_positive_cumsum + false_positive_cumsum + 1e-16)

            mean_average_precision.append(torch.trapz(precisions, recalls).item())

        if len(mean_average_precision) != 0:
            mean_average_precision = sum(mean_average_precision) / len(mean_average_precision)
        else:
            mean_average_precision = 0.

        return mean_average_precision

    def _forward_d(self, predictions: List[List[List[torch.Tensor]]], ground_truths: List[List[List[torch.Tensor]]]) -> Tuple[float, Dict[int, Union[float, None]]]:
        """
        Method that calculates mAP between predictions and ground_truths.

        :param List[List[List[torch.Tensor]]] **predictions**: Bounding boxes of predictions.
        :param List[List[List[torch.Tensor]]] **ground_truths**: Bounding boxes of ground_truths.
        :return: Average mAP (value between 0 to 1) and mAP per class.
        :rtype: Tuple[float, Dict[int, Union[float, None]]]
        """
        average_precision_per_class = {}

        for label, (preds_bboxes, gts_bboxes) in enumerate(zip(predictions, ground_truths)):
            amount_bboxes = Counter([gt[0] for gt in gts_bboxes])
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)
                            
            if self._num_classes > 1:              
                preds_bboxes.sort(key=lambda box: box[2], reverse=True)
            else:
                preds_bboxes.sort(key=lambda box: box[1], reverse=True)
            
            true_positive = torch.zeros((len(preds_bboxes)))
            false_positive = torch.zeros((len(preds_bboxes)))
            total_gt_bboxes = len(gts_bboxes)     

            if total_gt_bboxes == 0:
                continue

            for pred_idx, pred_bbox in enumerate(preds_bboxes):
                best_iou = 0

                for gt_idx, gt in enumerate(gts_bboxes):
                    if self._num_classes > 1:
                        if pred_bbox[1] != gt[1]:
                            continue
                    iou = intersection_over_union(pred_bbox[3:], gt[3:]) if self._num_classes > 1 else intersection_over_union(pred_bbox[2:], gt[2:])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou > self._iou_threshold:
                    if amount_bboxes[best_gt_idx] == 0:
                        true_positive[pred_idx] = 1
                        amount_bboxes[best_gt_idx] = 1
                    else:
                        false_positive[pred_idx] = 1
                else:
                    false_positive[pred_idx] = 1

            true_positive_cumsum = true_positive.cumsum(dim=0)
            false_positive_cumsum = false_positive.cumsum(dim=0)

            recalls = true_positive_cumsum / (total_gt_bboxes + 1e-9)
            precisions = true_positive_cumsum / (true_positive_cumsum + false_positive_cumsum + 1e-9)

            if label in average_precision_per_class.keys():
                average_precision_per_class[label].append(torch.trapz(precisions, recalls).item())
            else:
                average_precision_per_class[label] = [torch.trapz(precisions, recalls).item()]
            
            if label in average_precision_per_class.keys():
                if average_precision_per_class[label] != 0:
                    average_precision_per_class[label] = sum(average_precision_per_class[label]) / len(average_precision_per_class[label])
                else:
                    average_precision_per_class[label] = 0

        if len(average_precision_per_class) != 0:
            mean_average_precision = sum([item for _, item in average_precision_per_class.items()]) / len(average_precision_per_class)
        else:
            mean_average_precision = 0.

        return mean_average_precision, average_precision_per_class