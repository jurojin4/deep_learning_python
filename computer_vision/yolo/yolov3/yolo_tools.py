from tqdm import tqdm
from sklearn.cluster import KMeans
from ...utils import non_max_suppression
from typing import Any, Dict, List, Literal, Tuple

import torch

def generate_bboxes_prior(annotations: Dict[str, Any], num_bboxes_prior_per_scale: int = 3, normalize: bool = False, box_format: Literal['xyxy', 'xywh', 'xcycwh'] = "xywh") -> torch.Tensor:
    """
    Method that generates bboxes prior for a given dataset.

    :param Dict[str, Any] **annotations**: Dictionary containing bounding boxes.
    :param int **num_bboxes_prior_per_scale**: Number of bboxes prior per scale. Set to `3`.
    :param bool **normalize**: Boolean that specifies if values must be normalized. Set to `False`.
    :param Literal["xyxy", "xywh", "xcycwh"] **box_format**: Format of bounding boxes. Set to `xywh`.
    :return: Bounding Boxes prior.
    :rtype: Tensor
    """
    bboxes = []
    for (_, _bboxes), (_, sizes) in tqdm(zip(annotations["bboxes"].items(), annotations["sizes"].items()), total=len(annotations["bboxes"])):
        for bbox in _bboxes:
            bbox = bbox[1:]
            if normalize:
                bboxes.append([bbox[0] / sizes[0], bbox[1] / sizes[1], bbox[2] / sizes[0], bbox[3] / sizes[1]])
            else:
                bboxes.append(bbox)

    bboxes = torch.tensor(bboxes)

    if box_format == "xyxy":
        widths = bboxes[..., 2] - bboxes[..., 0]
        heights = bboxes[..., 3] - bboxes[..., 1]
    elif box_format == "xywh" or box_format == "xcycwh":
        widths = bboxes[..., 2]
        heights = bboxes[..., 3]

    whs = torch.stack((widths, heights), dim=1)
    kmeans = KMeans(n_clusters=3 * num_bboxes_prior_per_scale)
    kmeans.fit(whs)

    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    bboxes_prior = centers[torch.argsort(centers[..., 0] * centers[..., 1], descending=True)].reshape(3, num_bboxes_prior_per_scale, 2)
    return bboxes_prior

def intersection_over_union_bboxes_prior(boxes: torch.Tensor, bboxes_prior: torch.Tensor):
    """
    Method that calculates IoU between bounding boxes and bounding boxes prior.

    :param Tensor **bboxes**: Bounding boxes of shapes (1, 2) where the two elements are width and height.
    :param Tensor **bboxes_prior**: Bounding boxes prior of shapes (1, 2) where the two elements are width and height.
    :return: Intersection over Union.
    :rtype: Tensor
    """
    intersection = torch.min(boxes[..., 0], bboxes_prior[..., 0]) * torch.min(boxes[..., 1], bboxes_prior[..., 1])
    union = (boxes[..., 0] * boxes[..., 1] + bboxes_prior[..., 0] * bboxes_prior[..., 1] - intersection)

    return intersection / union

def get_bboxes_preds(predictions: List[torch.Tensor], bboxes_prior: List[torch.Tensor], num_classes: int, iou_threshold: float, confidence_threshold: float) -> List[List[List[float]]]:
    """
    Method that gets bounding boxes for predictions only.

    :param List[Tensor] **predictions**: Model's predictions.
    :param List[Tensor] **bboxes_prior**: Bounding boxes prior.
    :param int **num_classes**: Number of classes.
    :param int **iou_threshold**: IoU threshold between two bounding boxes to consider as overlap.
    :param int **confidence_threshold**: Confidence threshold to consider a bounding as valide according the model.
    :return: Predictions Bounding boxes.
    :rtype: List[List[List[float]]]
    """
    num_classes = num_classes if num_classes > 1 else 1
    batch_size = predictions[0].shape[0]
    all_preds_bboxes = [[] for _ in range(num_classes)]
    preds_bboxes = [[] for _ in range(batch_size)]

    for i in range(3):
        S = predictions[i].shape[2]
        boxes_scale = cells_to_bboxes(predictions[i], bboxes_prior[i], S, num_classes)
        for idx in range(batch_size):
            preds_bboxes[idx] += boxes_scale[idx]

    for idx in range(batch_size):
        nms_boxes = non_max_suppression(preds_bboxes[idx], num_classes, iou_threshold, confidence_threshold)
        for nms_box in nms_boxes:
            if num_classes > 1:
                all_preds_bboxes[int(nms_box[0])].append([idx] + nms_box)
            else:
                all_preds_bboxes[0].append([idx] + nms_box)
        
    return all_preds_bboxes

def get_bboxes(predictions: List[torch.Tensor], ground_truths: List[List[float]], bboxes_prior: List[torch.Tensor], num_classes: int, iou_threshold: float, confidence_threshold: float) -> Tuple[List[List[List[float]]], List[List[List[float]]]]:
    """
    Method that gets bounding boxes for predictions only.

    :param List[Tensor] **predictions**: Model's predictions.
    :param List[List[float]] **ground_truths**: Ground truths.
    :param List[Tensor] **bboxes_prior**: Bounding boxes prior.
    :param int **num_classes**: Number of classes.
    :param int **iou_threshold**: IoU threshold between two bounding boxes to consider as overlap.
    :param int **confidence_threshold**: Confidence threshold to consider a bounding as valide according the model.
    :return: Predictions Bounding boxes and ground truths bounding boxes.
    :rtype: Tuple[List[List[List[float]]], List[List[List[float]]]]
    """
    num_classes = num_classes if num_classes > 1 else 1
    batch_size = predictions[0].shape[0]

    all_preds_bboxes = [[] for _ in range(num_classes)]
    all_gts_bboxes = [[] for _ in range(num_classes)]
    preds_bboxes = [[] for _ in range(batch_size)]

    for i in range(3):
        S = predictions[i].shape[2]
        boxes_scale = cells_to_bboxes(predictions[i], bboxes_prior[i], S, num_classes)
        for idx in range(batch_size):
            preds_bboxes[idx] += boxes_scale[idx]

    for idx in range(batch_size):
        nms_boxes = non_max_suppression(preds_bboxes[idx], num_classes, iou_threshold, confidence_threshold)
        for nms_box in nms_boxes:
            if num_classes > 1:
                all_preds_bboxes[int(nms_box[0])].append([idx] + nms_box)
            else:
                all_preds_bboxes[0].append([idx] + nms_box)

    for gt_bbox in ground_truths:
        all_gts_bboxes[int(gt_bbox[1])].append([gt_bbox[0]] + gt_bbox[2:])
    
    return all_preds_bboxes, all_gts_bboxes

def cells_to_bboxes(bboxes: torch.Tensor, bboxes_prior_on_scale: torch.Tensor, s: int, num_classes: int) -> List[List[List[float]]]:
    """
    Method that gets bounding boxes for predictions only.

    :param Tensor **bboxes**: Bounding boxes.
    :param Tensor **bboxes_prior_on_scale**: Bounding boxes prior on scale.
    :param int **s**: Number of classes.
    :param int **num_classes**: Number of classes.
    :return: Bounding boxes processed.
    :rtype: List[List[List[float]]]
    """
    bboxes = bboxes.to("cpu")
    bboxes_prior_on_scale = bboxes_prior_on_scale.to("cpu")
    num_bboxes_prior_on_scale = len(bboxes_prior_on_scale)
    boxes = bboxes[..., 1:5]

    bboxes_prior_on_scale = bboxes_prior_on_scale.reshape(1, num_bboxes_prior_on_scale, 1, 1, 2)
    boxes[..., 0:2] = torch.sigmoid(boxes[..., 0:2])
    boxes[..., 2:] = torch.exp(boxes[..., 2:]) * bboxes_prior_on_scale
    scores = torch.sigmoid(bboxes[..., 0:1])

    if num_classes > 1:
        best_class = torch.argmax(bboxes[..., 5:], dim=-1).unsqueeze(-1)

    cell_indices = (torch.arange(s).repeat(bboxes.shape[0], 3, s, 1).unsqueeze(-1).to(bboxes.device))
    x = (boxes[..., 0:1] + cell_indices) * (1 / s)
    y = (boxes[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4)) * (1 / s)
    w_h = boxes[..., 2:4] * (1 / s)

    if num_classes > 1:
        converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(bboxes.shape[0], num_bboxes_prior_on_scale * s * s, 6)
    else:
        converted_bboxes = torch.cat((scores, x, y, w_h), dim=-1).reshape(bboxes.shape[0], num_bboxes_prior_on_scale * s * s, 5)

    all_bboxes = []
    for batch_idx in range(converted_bboxes.shape[0]):
        all_bboxes.append(converted_bboxes[batch_idx].tolist())
    return all_bboxes