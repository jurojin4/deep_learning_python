from typing import List, Tuple, Union
from ...metrics import intersection_over_union

import torch

def non_max_suppression(bboxes: List[torch.Tensor], num_classes: int, iou_threshold: float, confidence_threshold: float, alternative: bool = False) -> List[torch.Tensor]:
    """
    Non-Max-Supression method, sorts bounding boxes in function of the confidence and the overlay between the best box during the loop.
    
    :param List[Tensor] **bboxes**: Bounding boxes to sort, shape (batch, 6).
    :param int **num_classes**: Number of classes to predict.
    :param float **iou_threshold**: IoU threshold between two boxes to determine whether they are overlaid.
    :param float **confidence_threshold**: Threshold that bounding-box confidence scores must exceed.
    :param bool **alternative**: Boolean that adds a condition during NMS (label difference).
    :return: Bounding boxes sorted by NMS.
    :rtype: List[Tensor]
    """
    scaling = 0 if num_classes > 1 else 1
    bboxes = sorted([box for box in bboxes if box[1-scaling] > confidence_threshold], key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    while bboxes:
        best_box = bboxes.pop(0)

        if alternative and num_classes > 1:
            bboxes = [box for box in bboxes if intersection_over_union(torch.tensor(best_box[2-scaling:]), torch.tensor(box[2-scaling:])) < iou_threshold or best_box[0] != box[0]]
        else:
            bboxes = [box for box in bboxes if intersection_over_union(torch.tensor(best_box[2-scaling:]), torch.tensor(box[2-scaling:])) < iou_threshold]

        bboxes_after_nms.append(best_box)
    return bboxes_after_nms

def get_bboxes(S: int, B: int, num_classes: int, predictions: torch.Tensor, ground_truths: torch.Tensor, iou_threshold: float, confidence_threshold: float, alternative: bool = False) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
    """
    Method that converts and sorts bounding boxes with NMS method.

    :param int **S**: Cells number along height and width.
    :param int **B**: Bounding boxes number in each cell.
    :param int **num_classes**: Number of classes to predict.
    :param Tensor **predictions**: Predictions bounding boxes of shapes (batch, S, S, num_classes + 6 * B).
    :param Tensor **ground_truths**: Ground Truths bounding boxes of shapes (batch, S, S, num_classes + 6 * B).
    :param float **iou_threshold**: IoU threshold between two boxes to determine whether they are overlaid.
    :param float **confidence_threshold**: Threshold that bounding-box confidence scores must exceed.
    :param bool **alternative**: Boolean that adds a condition during NMS (label difference).
    :return: Bounding boxes converted and sorted.
    :rtype: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]
    """
    preds_bboxes = cellboxes_to_boxes(predictions, S, B, num_classes)
    gts_bboxes = cellboxes_to_boxes(ground_truths, S, B, num_classes)

    all_preds = [[]] * predictions.shape[0]
    all_ground_truths = [[]] * predictions.shape[0]

    scaling = 0 if num_classes > 1 else 1
    for idx in range(predictions.shape[0]):
        nms_boxes = non_max_suppression(preds_bboxes[idx], num_classes, iou_threshold=iou_threshold, confidence_threshold=confidence_threshold, alternative=alternative)
        for nms_box in nms_boxes:
            all_preds[idx].append(torch.cat((torch.tensor([idx]), torch.tensor(nms_box))))

        for box in gts_bboxes[idx]:
            if box[1-scaling] == 1:
                all_ground_truths[idx].append(torch.cat((torch.tensor([idx]), torch.tensor(box))))
    
    del gts_bboxes
    del preds_bboxes

    return all_preds, all_ground_truths

def get_bboxes_preds(S: int, B: int, num_classes: int, predictions: torch.Tensor, iou_threshold: float, confidence_threshold: float, alternative: bool = False) -> List[List[torch.Tensor]]:
    """
    Method that converts and sorts bounding boxes with NMS method.

    :param int **S**: Cells number along height and width.
    :param int **B**: Bounding boxes number in each cell.
    :param int **num_classes**: Number of classes to predict.
    :param Tensor **predictions**: Predictions bounding boxes of shapes (batch, S, S, num_classes + 6 * B).
    :param float **iou_threshold**: IoU threshold between two boxes to determine whether they are overlaid.
    :param float **confidence_threshold**: Threshold that bounding-box confidence scores must exceed.
    :param bool **alternative**: Boolean that adds a condition during NMS (label difference).
    :return: Bounding boxes converted and sorted.
    :rtype: List[List[torch.Tensor]]
    """
    predictions = predictions.reshape((1, S, S, num_classes + (B * 5)))
    preds_bboxes = cellboxes_to_boxes(predictions, S, B, num_classes)

    all_preds = [[]] * predictions.shape[0]
    for idx in range(predictions.shape[0]):
        nms_boxes = non_max_suppression(preds_bboxes[idx], num_classes, iou_threshold=iou_threshold, confidence_threshold=confidence_threshold, alternative=alternative)
        for nms_box in nms_boxes:
            all_preds[idx].append(torch.cat((torch.tensor([idx]), torch.tensor(nms_box))))

    del preds_bboxes
    
    return all_preds

def cellboxes_to_boxes(bboxes: torch.Tensor, S: int, B: int, num_classes: int) -> List[torch.Tensor]:
    """
    Method that converts bounding boxes from cell to global representation.

    :param Tensor **bboxes**: .
    :param int **S**: Cells number along height and width.
    :param int **B**: Bounding boxes number in each cell.
    :param int **num_classes**: Number of classes to predict.
    :return: Bounding boxes converted.
    :rtype: List[torch.Tensor]
    """
    bboxes = bboxes.to("cpu")
    batch_size = bboxes.shape[0]

    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)

    boundingboxes = []
    for i in range(num_classes, num_classes + 5 * B, 5):
        label = bboxes[..., :num_classes].argmax(-1).unsqueeze(-1)
        score = bboxes[..., i].unsqueeze(-1)
        x = (bboxes[..., i+1:i+2] + cell_indices) * (1 / S)
        y = (bboxes[..., i + 2:i + 3] + cell_indices.permute(0, 2, 1, 3)) * (1 / S)
        w = (bboxes[..., i + 3:i + 4]) * (1 / S)
        h = (bboxes[..., i + 4:i + 5]) * (1 / S)

        converted_bboxes = torch.cat((x, y, w, h), dim=-1)

        if num_classes > 1:
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