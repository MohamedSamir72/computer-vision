import torch

class BoundingBox:
    """
    A class to represent a bounding box and perform IoU calculations.
    """

    def __init__(self, coordinates: torch.Tensor, box_format: str = "midpoint"):
        """
        Initializes a BoundingBox object.

        Args:
            coordinates (torch.Tensor): Bounding box coordinates of shape (4,) or (6,).
            box_format (str): The format of the bounding box, either "midpoint" or "corner".
        """
        self.box_format = box_format
        self.coordinates = coordinates

        if self.box_format == "midpoint":
            # Convert from midpoint to corner format if necessary
            self.to_corner_format()

    def to_corner_format(self):
        """Converts from midpoint format to corner format (x1, y1, x2, y2)."""
        self.x1 = self.coordinates[0] - self.coordinates[2] / 2
        self.y1 = self.coordinates[1] - self.coordinates[3] / 2
        self.x2 = self.coordinates[0] + self.coordinates[2] / 2
        self.y2 = self.coordinates[1] + self.coordinates[3] / 2
        self.corner_coordinates = torch.tensor([self.x1, self.y1, self.x2, self.y2])

    def calculate_iou(self, other_box: "BoundingBox") -> float:
        """
        Calculates the Intersection over Union (IoU) between the current box and another bounding box.

        Args:
            other_box (BoundingBox): The second bounding box to compare with.

        Returns:
            float: IoU value between 0 and 1.
        """
        x1 = torch.max(self.corner_coordinates[0], other_box.corner_coordinates[0])
        y1 = torch.max(self.corner_coordinates[1], other_box.corner_coordinates[1])
        x2 = torch.min(self.corner_coordinates[2], other_box.corner_coordinates[2])
        y2 = torch.min(self.corner_coordinates[3], other_box.corner_coordinates[3])

        # Calculate intersection area (clamping to ensure positive area)
        intersection_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        # Calculate areas of both boxes
        pred_area = (self.corner_coordinates[2] - self.corner_coordinates[0]) * (self.corner_coordinates[3] - self.corner_coordinates[1])
        target_area = (other_box.corner_coordinates[2] - other_box.corner_coordinates[0]) * (other_box.corner_coordinates[3] - other_box.corner_coordinates[1])

        # Calculate union area
        union_area = pred_area + target_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area
        return iou.item()

class IoUCalculator:
    """
    A class to perform the IoU calculation between a list of predicted boxes and target boxes.
    """

    @staticmethod
    def intersection_over_union(pred_box: torch.Tensor, 
                                target_box: torch.Tensor, 
                                threshold: float = 0.5, 
                                box_format: str = "midpoint") -> float:
        """
        Calculates the IoU for a single pair of predicted and target boxes.

        Args:
            pred_box (torch.Tensor): The predicted bounding box.
            target_box (torch.Tensor): The ground truth bounding box.
            threshold (float): IoU threshold to consider a match (default is 0.5).
            box_format (str): Format of the bounding boxes ("midpoint" or "corner").

        Returns:
            float: IoU value between 0 and 1.
        """
        pred_bbox = BoundingBox(pred_box, box_format=box_format)
        target_bbox = BoundingBox(target_box, box_format=box_format)

        iou = pred_bbox.calculate_iou(target_bbox)

        if iou >= threshold:
            return iou
        else:
            return 0.0  # Return 0 if IoU is below the threshold

# Example usage
if __name__ == "__main__":
    predictions = torch.tensor([20, 30, 50, 70, 0.9, 0.1, 0.9])  # [x1, y1, x2, y2, conf, class1, class2, ...]
    targets = torch.tensor([25, 40, 50, 70, 1.0, 0, 1])  # [x1, y1, x2, y2, conf, class1, class2, ...]

    # IoU calculation
    iou_value = IoUCalculator.intersection_over_union(predictions, targets, box_format="midpoint")
    print(f"IoU Value: {iou_value}")
