from typing import Literal
from ...metrics import intersection_over_union

import torch
import torch.nn as nn

class YOLOV3Loss(nn.Module):
    """
    YOLOV3 Loss class.
    """
    def __init__(self, num_classes: int, lambda_coord: float = 10., lambda_obj: float = 1., lambda_noobj: float = 10., reduction: Literal["mean", "sum"] = "mean") -> None:
        """
        Initializes the YOLOV3Loss class.

        :param int **num_classes**: Number of classes to predict.
        :param float **lambda_coord**: Weight for boxes coordinates. Set to `5*10.`.
        :param float **lambda_obj**: Weight for object. Set to `1.`.
        :param float **lambda_noobj**: Weight for no object. Set to `10.`.
        :param Literal["mean", "sum"] **reduction**: Set to `mean`.
        """
        assert isinstance(num_classes, int), f"num_classes has to be an {int} instance, not {type(num_classes)}."
        assert isinstance(lambda_coord, float), f"lambda_coord has to be a {float} instance, not {type(lambda_coord)}."
        assert isinstance(lambda_obj, float), f"lambda_obj has to be a {float} instance"
        assert isinstance(lambda_noobj, float), f"lambda_noobj has to be an {float} instance, not {type(lambda_noobj)}."
        assert reduction in ["mean", "sum"], f"reduction has to be in [\"mean\", \"sum\"], not {reduction}."
        super().__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj

        self._sigmoid = nn.Sigmoid()
        self._mse = nn.MSELoss(reduction=reduction)
        self._crossentropy = nn.CrossEntropyLoss(reduction=reduction)
        self._bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, bboxes_prior_on_scale: torch.Tensor) -> torch.Tensor:
        """
        Method that forwards through the loss function predictions and targets.

        :param Tensor **predictions**: Model grid predictions.
        :param Tensor **targets**: Ground truths grid.
        :param Tensor **boxes_prior_on_scale**: Boxes prior for the given scale.
        :return: Loss value.
        :rtype: Tensor
        """
        bboxes_prior_on_scale = bboxes_prior_on_scale.reshape(1, len(bboxes_prior_on_scale), 1, 1, 2)

        ind_obj = targets[..., 0] == 1
        ind_noobj = targets[..., 0] == 0

        noobj_loss = self._bce(predictions[..., 0:1][ind_noobj], targets[..., 0:1][ind_noobj])

        bboxes_preds = torch.cat((self._sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * bboxes_prior_on_scale), dim=-1)
        ious = intersection_over_union(bboxes_preds[ind_obj], targets[..., 1:5][ind_obj]).detach()
        obj_loss = self._mse(self._sigmoid(predictions[..., 0:1][ind_obj]), ious * targets[..., 0:1][ind_obj])

        preds = predictions.clone()
        tars = targets.clone()
        preds[..., 1:3] = self._sigmoid(preds[..., 1:3])
        tars[..., 3:5] = torch.log(1e-15 + (tars[..., 3:5] / bboxes_prior_on_scale))
        box_loss = self._mse(preds[..., 1:5][ind_obj], tars[..., 1:5][ind_obj])

        if self.num_classes > 1:
            class_loss = self._crossentropy(predictions[..., 5:][ind_obj], targets[..., 5][ind_obj].long())
            return self.lambda_noobj * noobj_loss + class_loss + self.lambda_coord * box_loss + self.lambda_obj * obj_loss
        else:
            return self.lambda_noobj * noobj_loss + self.lambda_coord * box_loss + self.lambda_obj * obj_loss