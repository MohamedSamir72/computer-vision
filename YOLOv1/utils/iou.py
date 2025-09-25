import torch

def convert_to_corners(box: torch.Tensor, box_format: str = "corners") -> torch.Tensor:
    """
    Converts a bounding box to corner format [x1, y1, x2, y2].

    Args:
        box (torch.Tensor): The bounding box.
        box_format (str): Format of the bounding box ("midpoint" or "corners").

    Returns:
        torch.Tensor: Bounding box in corner format.
    """
    if box_format == "corners":
        return box[:4]  # Assume only the first 4 values are box coordinates
    elif box_format == "midpoint":
        x_center, y_center, width, height = box[:4]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return torch.tensor([x1, y1, x2, y2])
    else:
        raise ValueError("Invalid box_format: choose 'corners' or 'midpoint'.")


def intersection_over_union(pred_box: torch.Tensor, 
                            target_box: torch.Tensor, 
                            threshold: float = 0.5, 
                            box_format: str = "midpoint") -> float:
    """
    Calculates the IoU for a predicted and ground truth bounding box.

    Args:
        pred_box (torch.Tensor): Predicted bounding box.
        target_box (torch.Tensor): Ground truth bounding box.
        threshold (float): IoU threshold for match.
        box_format (str): Format of boxes ("midpoint" or "corners").

    Returns:
        float: IoU if above threshold, else 0.0
    """
    pred_box_corners = convert_to_corners(pred_box, box_format)
    target_box_corners = convert_to_corners(target_box, box_format)

    x1 = torch.max(pred_box_corners[0], target_box_corners[0])
    y1 = torch.max(pred_box_corners[1], target_box_corners[1])
    x2 = torch.min(pred_box_corners[2], target_box_corners[2])
    y2 = torch.min(pred_box_corners[3], target_box_corners[3])

    # Calculate intersection area
    intersection_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Areas of the boxes
    box1_area = (pred_box_corners[2] - pred_box_corners[0]) * (pred_box_corners[3] - pred_box_corners[1])
    box2_area = (target_box_corners[2] - target_box_corners[0]) * (target_box_corners[3] - target_box_corners[1])

    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    
    return iou.item() if iou >= threshold else 0.0



if __name__ == "__main__":
    predictions = torch.tensor([20, 30, 50, 70, 0.9, 0.1, 0.9])  # Example: midpoint or corner + confidence + class
    targets = torch.tensor([25, 40, 50, 70, 1.0, 0, 1])

    # You can choose "midpoint" or "corners" depending on input format
    iou_value = intersection_over_union(predictions, targets, box_format="midpint")
    print(f"IoU Value: {iou_value}")
