import torch
from utils.iou import IoUCalculator

def non_max_suppression(predictions: list[torch.Tensor],
                        iou_threshold: float = 0.5,
                        prob_threshold: float = 0.5,
                        box_format: str ="corners",
                        ):
    """
    Performs Non-Maximum Suppression (NMS) on the bounding boxes.
    Args:
        predictions (torch.Tensor): Predictions from the model of shape (num_boxes, 6)
                                     where each box is represented as (x1, y1, x2, y2, conf, cls).
        iou_threshold (float): IoU threshold for NMS.
        prob_threshold (float): Probability threshold to filter boxes before NMS.
        box_format (str): Format of the bounding boxes. Either "corners" or "midpoint".
                          "corners" format is (x1, y1, x2, y2).
                          "midpoint" format is (x_center, y_center, width, height).
    """

    assert type(predictions) == torch.Tensor, "Predictions should be a torch.Tensor"
    assert box_format in ["corners", "midpoint"], "box_format should be either 'corners' or 'midpoint'"
    
    # Filter out boxes with low confidence scores
    bboxes = predictions[predictions[:, 4] > prob_threshold]
    # Sort the boxes by confidence score in descending order
    bboxes = bboxes[bboxes[:, 4].argsort(descending=True)]

    selected_bboxes = []
    print(bboxes.shape)
    print(bboxes.size(0))
    
    while bboxes.size(0):
        # Select the box with highest confidence
        chosen_box = bboxes[0]
        selected_bboxes.append(chosen_box)
        if bboxes.size(0) == 1:
            break

        # Compute IoU between the chosen box and the rest
        ious = torch.zeros(bboxes.size(0) - 1)
        for i in range(1, bboxes.size(0)):
            iou = IoUCalculator.intersection_over_union(
                chosen_box[:4], bboxes[i, :4], box_format=box_format
            )
            ious[i-1] = iou

        # Keep boxes with IoU less than threshold or different class
        keep = (ious < iou_threshold) | (bboxes[1:, 5] != chosen_box[5])
        bboxes = bboxes[1:][keep]

    if selected_bboxes:
        return torch.stack(selected_bboxes)
    else:
        return torch.empty((0, 6))

if __name__ == "__main__":
    boxes = torch.Tensor([[1, 1, 2, 2, 0.9, 0],
                          [1, 1, 2, 2, 0.8, 0],
                          [2, 2, 3, 3, 0.7, 1],
                          [2, 2, 3, 3, 0.6, 1],
                          [10, 10, 11, 11, 0.5, 0]])
    
    result = non_max_suppression(boxes, iou_threshold=0.5, box_format="corners")
    print("NMS result:\n", result)