from typing import List, Tuple, Union
from ...utils import non_max_suppression
from ...metrics import intersection_over_union

import torch

def get_bboxes(predictions: torch.Tensor, ground_truths: List[List[float]], S: int, B: int, num_classes: int, iou_threshold: float, confidence_threshold: float, alternative: bool = False) -> Tuple[List[List[List[float]]], List[List[List[float]]]]:
    """
    Method that converts and sorts bounding boxes with NMS method.

    :param Tensor **predictions**: Predictions bounding boxes of shapes (batch, S, S, num_classes + 6 * B).
    :param Tensor **ground_truths**: Ground Truths bounding boxes of shapes (batch, S, S, num_classes + 6 * B).
    :param int **S**: Cells number along height and width.
    :param int **B**: Bounding boxes number in each cell.
    :param int **num_classes**: Number of classes to predict.
    :param float **iou_threshold**: IoU threshold between two boxes to determine whether they are overlaid.
    :param float **confidence_threshold**: Threshold that bounding-box confidence scores must exceed.
    :param bool **alternative**: Boolean that adds a condition during NMS (label difference).
    :return: Bounding boxes converted and sorted.
    :rtype: Tuple[List[List[List[float]]], List[List[List[float]]]]
    """
    preds_bboxes = cellboxes_to_boxes(predictions, S, B, num_classes)

    num_classes = num_classes if num_classes > 1 else 1
    all_preds_bboxes = [[] for _ in range(num_classes)]
    all_gts_bboxes = [[] for _ in range(num_classes)]

    batch_size = predictions.shape[0]
    for idx in range(batch_size):
        nms_boxes = non_max_suppression(preds_bboxes[idx], num_classes, iou_threshold=iou_threshold, confidence_threshold=confidence_threshold, alternative=alternative)
        for nms_box in nms_boxes:
            if num_classes > 1:
                all_preds_bboxes[int(nms_box[0])].append([idx] + nms_box)
            else:
                all_preds_bboxes[0].append([idx] + nms_box)

    for gt_bbox in ground_truths:
        all_gts_bboxes[int(gt_bbox[1])].append([gt_bbox[0]] + gt_bbox[2:])

    return all_preds_bboxes, all_gts_bboxes

def get_bboxes_preds(predictions: torch.Tensor, S: int, B: int, num_classes: int, iou_threshold: float, confidence_threshold: float, alternative: bool = False) -> List[List[List[float]]]:
    """
    Method that converts and sorts bounding boxes with NMS method.

    :param Tensor **predictions**: Predictions bounding boxes of shapes (batch, S, S, num_classes + 6 * B).
    :param int **S**: Cells number along height and width.
    :param int **B**: Bounding boxes number in each cell.
    :param int **num_classes**: Number of classes to predict.
    :param float **iou_threshold**: IoU threshold between two boxes to determine whether they are overlaid.
    :param float **confidence_threshold**: Threshold that bounding-box confidence scores must exceed.
    :param bool **alternative**: Boolean that adds a condition during NMS (label difference).
    :return: Bounding boxes converted and sorted.
    :rtype: List[List[List[float]]]
    """
    num_classes = num_classes if num_classes > 1 else 0
    predictions = predictions.reshape((1, S, S, num_classes + (B * 5)))
    preds_bboxes = cellboxes_to_boxes(predictions, S, B, num_classes)
    num_classes = num_classes if num_classes > 1 else 1

    batch_size = predictions.shape[0]
    all_preds_bboxes = [[] for _ in range(num_classes)]
    for idx in range(batch_size):
        nms_boxes = non_max_suppression(preds_bboxes[idx], num_classes, iou_threshold=iou_threshold, confidence_threshold=confidence_threshold, alternative=alternative)
        for nms_box in nms_boxes:
            if num_classes > 1:
                all_preds_bboxes[int(nms_box[0])].append([idx] + nms_box)
            else:
                all_preds_bboxes[0].append([idx] + nms_box)
                
    return all_preds_bboxes

def cellboxes_to_boxes(bboxes: torch.Tensor, S: int, B: int, num_classes: int) -> List[List[List[float]]]:
    """
    Method that converts bounding boxes from cell to global representation.

    :param Tensor **bboxes**: .
    :param int **S**: Cells number along height and width.
    :param int **B**: Bounding boxes number in each cell.
    :param int **num_classes**: Number of classes to predict.
    :return: Bounding boxes converted.
    :rtype: List[List[List[float]]]
    """
    bboxes = bboxes.to("cpu")
    batch_size = bboxes.shape[0]

    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)

    boundingboxes = []
    for i in range(num_classes, num_classes + 5 * B, 5):
        score = bboxes[..., i].unsqueeze(-1)
        x = (bboxes[..., i+1:i+2] + cell_indices) * (1 / S)
        y = (bboxes[..., i + 2:i + 3] + cell_indices.permute(0, 2, 1, 3)) * (1 / S)
        w = (bboxes[..., i + 3:i + 4]) * (1 / S)
        h = (bboxes[..., i + 4:i + 5]) * (1 / S)

        converted_bboxes = torch.cat((x, y, w, h), dim=-1)

        if num_classes > 1:
            label = bboxes[..., :num_classes].argmax(-1).unsqueeze(-1)
            converted_bboxes = torch.cat((label, score, converted_bboxes), dim=-1)
        else:
            converted_bboxes = torch.cat((score, converted_bboxes), dim=-1)

        boundingboxes.append(converted_bboxes)

    boundingboxes = torch.cat(boundingboxes, dim=-1)
    
    if num_classes > 1:
        boundingboxes = boundingboxes.reshape((boundingboxes.shape[0], S * S * B , 6))
    else:
        boundingboxes = boundingboxes.reshape((boundingboxes.shape[0], S * S * B , 5))

    all_bboxes = []
    for batch_idx in range(boundingboxes.shape[0]):
        all_bboxes.append(boundingboxes[batch_idx].tolist())
    return all_bboxes