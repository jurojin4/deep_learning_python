from ...datasets import Compose, Dataset
from typing import Any, List, Literal, Tuple, Union

import torch

class YOLOV1Dataset(Dataset):
    """
    YOLOV1Dataset class.
    """
    def __init__(self, dataset_name: str, dataset_path: str, S: int, B: int, height: int, width: int, transformations: Compose, box_format: Literal["xyxy", "xywh", "xcycwh"] = "xywh", model_normalized: bool = True, type_set: Literal["train", "validation"] = "train", data_aug: bool = False, **kwargs):
        """
        Initializes the YOLOV1Dataset class.

        :param Union[Dict[str, List[np.ndarray]], Dict[str, List[int]], Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Union[List[int], List[List[int]]]]], Dict[str, Dict[str, Union[int, List[int]]]]]] **dataset**: Dataset used.
        :param int **num_classes**: Number of classes to predict.
        :param int **S**: Cells number along height and width.
        :param int **B**: Bounding boxes number in each cell.
        :param int **height**: Height of the image.
        :param int **width**: Width of the image.
        :param Compose **transformations**: Tool transforming images.
        :param Literal["xyxy", "xywh", "xcycwh"] **box_format**: Format of bounding boxes. Set to `xywh`.
        :param bool **model_normalized**: Boolean that indicates if outputs are normalized between 0 and 1.
        :param Literal["train", "validation"] **type_set**: Set to `train`.
        :param bool **data_aug**: Boolean that allows data augmentation. Set to `False`.
        """
        assert isinstance(S, int), f"S has to be an {int} instance, not {type(S)}."
        assert isinstance(B, int), f"B has to be an {int} instance, not {type(B)}."
        assert isinstance(height, int), f"height has to be an {int} instance, not {type(height)}."
        assert isinstance(width, int), f"width has to be an {int} instance, not {type(width)}."
        assert isinstance(transformations, Compose), f"transformations has to be a {Compose} instance (or None), not {type(transformations)}."
        assert box_format in ["xyxy", "xywh", "xcycwh"], f'box_format has to be in ["xyxy", "xywh", "xcycwh"], not equal to {box_format}.'
        super().__init__(dataset_name, dataset_path, width, height, transformations, box_format, model_normalized, type_set, data_aug, **kwargs)

        if self.mode == "classification":
            self.S = None
            self.B = None
        else:
            self.S = S
            self.B = B

        self._num_classes = self.num_classes if self.num_classes > 1 else 0

    
    def _loc(self, line: torch.Tensor, rate: int = 5) -> int:
        """
        Method that returns the location of the bounding box in the line of size num_classes + B * 2 from case num_classes.
        The method attributes one of B spot in the line, if none is empty then the value returned is superior than B * 5 + num_classes.
        
        :param Tensor **line*: 
        :param int **rate**: Number of elements in a box.
        :return: Beginning of the box.
        :rtype: int
        """
        for i in range(self._num_classes, line.shape[0], rate):
            if line[i] == 0:
                return int(i/rate)
        return int((i/rate)) + 1
    
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
            grid_label = torch.zeros(size=(self.S, self.S, self._num_classes + (5 * self.B)))

            for bbox in bboxes:
                if self._num_classes > 1:
                    label = bbox[0]
                
                box = bbox[1:]
                x, y, w, h = box

                i, j = int(y * self.S), int(x * self.S)

                x_cell, y_cell = self.S * x - j, self.S * y - i
                height_cell, width_cell = (h * self.S, w * self.S)

                pos = torch.nonzero(grid_label[i, j, :self._num_classes])
                if pos.nelement() != 0:
                    if pos[0].item() == label:
                        n = self._loc(grid_label[i,j])
                        if n < self.B:
                            grid_label[i, j, n * 5] = 1

                            box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                            grid_label[i, j, n * 5 + 1:(n+1) * 5] = box_coordinates
                        else:
                            break
                    else:
                        continue
                else:
                    grid_label[i, j, self._num_classes] = 1

                    cell_bbox = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    grid_label[i, j, self._num_classes+1:self._num_classes+5] = cell_bbox

                    if self._num_classes > 1:
                        grid_label[i, j, int(label)] = 1
                    else:
                        continue            
            return grid_label

def collate_fn(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Specific collate function for Dataset class.

    :param Any **batch**:
    :return: Apdated Batch.
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    images, targets, _batch_bboxes = zip(*batch)

    batch_bboxes = []
    for i, bboxes in enumerate(_batch_bboxes):
        for bbox in bboxes:
            batch_bboxes.append([i] + bbox)
    return torch.stack(images, dim=0), (torch.stack(targets, dim=0), torch.tensor(batch_bboxes))