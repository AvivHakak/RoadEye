import torch
import torch.nn as nn

def calculate_iou_tensor(boxes_preds, boxes_labels):
    box1_x1 = boxes_preds[:, 0:1, ...] - boxes_preds[:, 2:3, ...] / 2
    box1_y1 = boxes_preds[:, 1:2, ...] - boxes_preds[:, 3:4, ...] / 2
    box1_x2 = boxes_preds[:, 0:1, ...] + boxes_preds[:, 2:3, ...] / 2
    box1_y2 = boxes_preds[:, 1:2, ...] + boxes_preds[:, 3:4, ...] / 2

    box2_x1 = boxes_labels[:, 0:1, ...] - boxes_labels[:, 2:3, ...] / 2
    box2_y1 = boxes_labels[:, 1:2, ...] - boxes_labels[:, 3:4, ...] / 2
    box2_x2 = boxes_labels[:, 0:1, ...] + boxes_labels[:, 2:3, ...] / 2
    box2_y2 = boxes_labels[:, 1:2, ...] + boxes_labels[:, 3:4, ...] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

class RoadEyeLoss(nn.Module):
    def __init__(self):
        super(RoadEyeLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCELoss(reduction="sum")
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def forward(self, predictions, target):
        obj = target[:, 0, ...] == 1
        noobj = target[:, 0, ...] == 0

        noobj_loss = self.bce(
            predictions[:, 0, ...][noobj],
            target[:, 0, ...][noobj]
        )

        if obj.sum() > 0:
            obj_loss = self.bce(
                predictions[:, 0, ...][obj],
                target[:, 0, ...][obj]
            )

            pred_boxes = predictions[:, 1:5, ...].permute(0, 2, 3, 1)
            target_boxes = target[:, 1:5, ...].permute(0, 2, 3, 1)
            
            box_loss = self.mse(
                pred_boxes[obj],
                target_boxes[obj]
            )
        else:
            obj_loss = 0.0
            box_loss = 0.0

        total_loss = (
            self.lambda_coord * box_loss
            + obj_loss
            + self.lambda_noobj * noobj_loss
        )

        return total_loss