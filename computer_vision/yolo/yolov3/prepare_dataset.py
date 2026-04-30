from ...datasets import Compose, Dataset
from typing import Any, List, Literal, Tuple, Union
from .yolo_tools import intersection_over_union_bboxes_prior

import torch

class YOLOV3Dataset(Dataset):
    """
    YOLOV3Dataset class.
    """
    def __init__(self, dataset_name: str, dataset_path: str, S: Union[List[int], Tuple[int, int, int]], width: int, height: int, transformations: Compose, box_format: Literal["xyxy", "xywh", "xcycwh"] = "xywh", model_normalized: bool = True, type_set: Literal["train", "validation"] = "train", data_aug: bool = False, **kwargs):
        assert len(S) == 3, f"S has to be of length equal to 3, not {len(S)}."
        assert isinstance(transformations, Compose), f"transformations has to be a {Compose} instance (or None), not {type(transformations)}."
        assert box_format in ["xyxy", "xywh", "xcycwh"], f'box_format has to be in ["xyxy", "xywh", "xcycwh"], not equal to {box_format}.'
        super().__init__(dataset_name, dataset_path, width, height, transformations, box_format, model_normalized, type_set, data_aug, **kwargs)

        if self.mode == "classification":
            self.S = None
        else:
            self.S = S

    def _set_bboxes_prior(self, bboxes_prior: torch.Tensor) -> None:
        """
        Method that sets bounding boxes prior (anchors).

        :param Tensor **bboxes_prior**: Bouding boxes prior.
        """
        self._bboxes_prior = bboxes_prior
        self._num_bboxes_prior_per_scale = bboxes_prior.shape[1]
        self._bboxes_prior = bboxes_prior.reshape(9, 2) 

    def _specified_getitem(self, bboxes) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Built-in Python method that allows to access an element from the object.

        :param int **index**:
        :return: In mode "classification", a tensor and a label. In mode "object_detection, a tuple of tensor of size 2.
        :rtype: Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]
        """
        if self.mode == "classification":
            return None
        else:
            num_elements = 6 if self.num_classes > 1 else 5
            grid_label = [torch.zeros((self._num_bboxes_prior_per_scale, s, s, num_elements), dtype=torch.float32) for s in self.S]
            for bbox in bboxes:
                if self.num_classes > 1:
                    label = bbox[0]
                
                box = bbox[1:]

                if self._box_format == "xyxy":
                    x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
                elif self._box_format == "xywh":
                    x, y, w, h = box
                elif self._box_format == "xcycwh":
                    x, y, w, h = box[0] - (box[2] / 2), box[1] - (box[3] / 2), box[2], box[3]

                has_bboxes_prior = [False] * 3

                ious_prior = intersection_over_union_bboxes_prior(torch.tensor([w, h]), self._bboxes_prior)
                ious_prior_indices = ious_prior.argsort(descending=True, dim=0)

                for prior_indice in ious_prior_indices:
                    scale_indice = prior_indice // self._num_bboxes_prior_per_scale

                    prior_indice_compared_to_scale = prior_indice % self._num_bboxes_prior_per_scale

                    s = self.S[scale_indice]
                    i, j = int(s * y), int(s * x)

                    cell_taken = grid_label[scale_indice][prior_indice_compared_to_scale, i, j, 0]
                    if not cell_taken and not has_bboxes_prior[scale_indice]:
                        grid_label[scale_indice][prior_indice_compared_to_scale, i, j, 0] = 1.

                        x_cell, y_cell = s * x - j, s * y - i
                        width_cell, height_cell = (w * s, h * s)

                        box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                        grid_label[scale_indice][prior_indice_compared_to_scale, i, j, 1:5] = box_coordinates

                        if self.num_classes > 1:
                            grid_label[scale_indice][prior_indice_compared_to_scale, i, j, 5] = int(label)
                        has_bboxes_prior[scale_indice] = True

            return grid_label
        
def collate_fn(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Specific collate function for Dataset class.

    :param Any **batch**:
    :return: Apdated Batch.
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    images, targets, _batch_bboxes = zip(*batch)
    targets0, targets1, targets2 = zip(*targets)

    targets0 = torch.stack(targets0, dim=0)
    targets1 = torch.stack(targets1, dim=0)
    targets2 = torch.stack(targets2, dim=0)

    batch_bboxes = []
    for i, bboxes in enumerate(_batch_bboxes):
        for bbox in bboxes:
            batch_bboxes.append([i] + bbox)
    return torch.stack(images, dim=0), ((targets0, targets1, targets2), torch.tensor(batch_bboxes))