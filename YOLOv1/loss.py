import torch
from torch import nn
from utils.config import Config
from utils.iou import intersection_over_union

class YOLO_Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

        self.mse = nn.MSELoss(reduction="sum")



    def forward(self, predictions, target):
        pass

if __name__ == "__main__":
    predictions = torch.randn((2, 7, 7, 30))
    target = torch.randn((2, 7, 7, 30))

    criterion = YOLO_Loss()
    loss = criterion(predictions, target)
    print(loss)