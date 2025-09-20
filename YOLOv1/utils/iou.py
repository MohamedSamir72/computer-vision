import torch
import numpy as np

def claculate_iou(box_pred, box_target):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (torch.Tensor): Bounding box 1 of shape (4,) in format [x1, y1, x2, y2].
        box2 (torch.Tensor): Bounding box 2 of shape (4,) in format [x1, y1, x2, y2].

    Returns:
        float: IoU value.
    """
    ### Calculate intersection coordinates
    x1 = torch.max(box_pred[0], box_target[0])
    y1 = torch.max(box_pred[1], box_target[1])
    x2 = torch.min(box_pred[2], box_target[2])
    y2 = torch.min(box_pred[3], box_target[3])

    ### To ensure the only positive values are considered for intersection area
    intersection_area: int = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    # print(f"Intersection: {intersection_area}")

    pred_area: int = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    target_area: int = (box_target[2] - box_target[0]) * (box_target[3] - box_target[1])
    union_area: int = pred_area + target_area - intersection_area
    # print(f"Union: {union_area}")

    iou: float = intersection_area / union_area 
    
    return iou.item()

def intersection_over_union(pred_box: torch.Tensor, 
                            target_box: torch.Tensor, 
                            threshold: float = 0.5, 
                            box_format: str = "midpoint"):
    """
    Initialize with two sets of bounding boxes and a threshold.
    predicitions & targets shape: [x1, y1, x2, y2, conf, class1, class2, ...]

    Args:
        pred_box (torch.Tensor): Predicted bounding box.
        target_box (torch.Tensor): Ground truth bounding box.
        threshold (float): IoU threshold to consider a match.
    """

    if box_format == "midpoint":
        pred_box = [pred_box[0] - pred_box[2] / 2,
                    pred_box[1] - pred_box[3] / 2,
                    pred_box[0] + pred_box[2] / 2,
                    pred_box[1] + pred_box[3] / 2]
        
        target_box = [target_box[0] - target_box[2] / 2,
                    target_box[1] - target_box[3] / 2,
                    target_box[0] + target_box[2] / 2,
                    target_box[1] + target_box[3] / 2]
        
        return claculate_iou(pred_box, target_box)

    elif box_format == "corner":
        return claculate_iou(pred_box, target_box)
    
if __name__ == "__main__":
    # Each row: [x_min, y_min, x_max, y_max, conf, class1, class2, ...]
    predictions = torch.tensor([20, 30, 50, 70, 0.9, 0.1, 0.9])

    targets = torch.tensor([25, 40, 50, 70, 1.0, 0, 1])

    print(intersection_over_union(predictions, targets, box_format="midpoint"))