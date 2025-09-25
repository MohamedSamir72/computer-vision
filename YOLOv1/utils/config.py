from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class Config:
    """
    Configuration settings
    """
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 4
    NUM_EPOCHS: int = 100
    ARCHITECTURE = [
        (7, 64, 2, 3),
        "MaxPool",
        (3, 192, 1, 1),
        "MaxPool",
        (1, 128, 1, 0),
        (3, 256, 1, 1),
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        "MaxPool",
        [(1, 256, 1, 0), (3, 512, 1, 1), 4],
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        "MaxPool",
        [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
        (3, 1024, 1, 1),
        (3, 1024, 2, 1),
        (3, 1024, 1, 1),
        (3, 1024, 1, 1)
    ]
    
