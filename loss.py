"""
Loss function is the sum of 4 quantities: 
    1. distance to center
    2. sum of width and height differences from actual 
        - we will take square root to equalize size of bounding boxes, small count the same as large
    3. probability that there is a box
    4. probability that there is NOT a box (no object in the square)

    For all of these quantities, we take the largest IOU
    To maintain convexity so that our numerical descent work, I used L2 Loss
"""
import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction = 'sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noObj = 0.5
        self.lambda_coord = 5
    
    def forward(self, predictions, target):
        """Given an 
        Input:
            1. predictions = our predictions from our model
            2. target = actual labels, has shape (batchSize, S, S, 25)
        Output:
            1. loss
        """
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5) # predictions has shape (batchSize, 7, 7, 30)
        
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) # note: ... means repeated colons
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        iouS = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim = 0)
        iou_maxes, best_box = torch.max(iouS, dim = 0) # note: best_box refers to the box that's reponsible (highest IOU)
        exists_box = target[..., 20].unsqueeze(3) # exists_box refers to the indicator function 1^obj_i (1 if there is an object in cell i)

        # =============== #
        # BOX COORDINATES #
        # =============== #
        box_predictions = exists_box * (
            (best_box * predictions[..., 26:30] + (1 - best_box) * predictions[..., 21:25])
        )
        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(
            box_predictions[..., 2:4] * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6)))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        # (N, S, S, 4) ----> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim = -2),
            torch.flatten(box_targets, end_dim = -2)
        )

        # =========== #
        # OBJECT LOSS #
        # =========== #
        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21])
        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )
        # ============== #
        # NO OBJECT LOSS #
        # ============== #
        # (N, S, S, 1) ----> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim = 1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim = 1)
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim = 1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim = 1)
        )
        # =================== #
        # CLASSIFICATION LOSS #
        # =================== #
        # (N, S, S, 20) ----> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim = 2),
            torch.flatten(exists_box * target[..., :20], end_dim = 2)
        )

        loss = (
            self.lambda_coord * box_loss 
            + object_loss 
            + self.lambda_noObj * no_object_loss
            + class_loss
        )
        return loss