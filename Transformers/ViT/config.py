from dataclasses import dataclass
import torch

@dataclass
class Config:
    LEARNING_RATE: float = 0.001
    DROP_OUT: float = 0.1
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 10
    IMG_SIZE: int = 32
    CHANNELS: int = 3
    PATCH_SIZE: int = 4
    EMBEDING_SIZE: int = 512
    NUM_HEADS: int = 8
    MLP_DIM = 512
    DEPTH: int = 6
    NUM_CLASSES: int = 10
    SAVE_PATH: str = "./models"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
