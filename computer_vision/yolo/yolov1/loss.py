from typing import Literal

import torch
import torch.nn as nn

class YOLOV1Loss(nn.Module):
    """
    YOLOV1 Loss class.
    """
    def __init__(self, num_classes: int, S: int = 7, B: int = 2, lambda_coord: float = 5., lambda_noobj: float = 0.5, reduction: Literal["mean", "sum"] = "sum") -> None:
        """
        Initializes the YOLOV1Loss class.

        :param int **num_classes**: Number of classes to predict.
        :param int **S**: Cells number along height and width.
        :param int **B**: Bounding boxes number in each cell.
        :param float **lambda_coord**: Weight for boxes coordinates. Set to `5.`.
        :param float **lambda_noobj**: Weight for no object. Set to `0.5`.
        :param Literal["mean", "sum"] **reduction**: Set to `sum`.
        """
        assert isinstance(num_classes, int), f"num_classes has to be an {int} instance, not {type(num_classes)}."
        assert isinstance(S, int), f"S has to be an {int} instance, not {type(S)}."
        assert isinstance(B, int), f"B has to be an {int} instance, not {type(B)}."
        assert isinstance(lambda_coord, float), f"lambda_coord has to be a {float} instance, not {type(lambda_coord)}."
        assert isinstance(lambda_noobj, float), f"lambda_noobj has to be a {float} instance, not {type(lambda_noobj)}."
        assert reduction in ["mean", "sum"], f"reduction has to be in [\"mean\", \"sum\"], not {reduction}."
        super().__init__()
        self._mse = nn.MSELoss(reduction=reduction)

        self.num_classes = num_classes
        self.S = S
        self.B = B

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards through the loss function predictions and targets.

        :param Tensor **predictions**: Model grid predictions.
        :param Tensor **targets**: Ground truths grid.
        :return: Loss value.
        :rtype: Tensor
        """
        predictions_boxes = predictions[..., self.num_classes:].clone()
        predictions_boxes = predictions_boxes.reshape((predictions_boxes.shape[0:3]) + (self.B, 5))

        targets_boxes = targets[..., self.num_classes:].clone()
        targets_boxes = targets_boxes.reshape((targets_boxes.shape[0:3]) + (self.B, 5))

        obj = targets_boxes[..., 0] == 1
        no_obj = targets_boxes[..., 0] != 1

        predictions_boxes[..., 3:5] = torch.sign(predictions_boxes[..., 3:5]) * torch.sqrt(torch.abs(predictions_boxes[..., 3:5] + 1e-6))
        targets_boxes[..., 3:5] = torch.sign(targets_boxes[..., 3:5]) * torch.sqrt(torch.abs(targets_boxes[..., 3:5] + 1e-6))

        box_loss = self._mse(predictions_boxes[obj][..., 1:5], targets_boxes[obj][..., 1:5])

        object_loss = self._mse(predictions_boxes[obj][..., 0], targets_boxes[obj][..., 0])
        no_object_loss = self._mse(predictions_boxes[no_obj][..., 0], targets_boxes[no_obj][..., 0])

        predictions_class = predictions[..., :self.num_classes].clone()
        targets_class = targets[..., :self.num_classes].clone()

        obj = obj.any(dim=3)

        class_loss = self._mse(predictions_class[obj], targets_class[obj])

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        return loss