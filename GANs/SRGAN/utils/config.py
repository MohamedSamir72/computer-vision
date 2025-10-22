from dataclasses import dataclass
import torch
import os

@dataclass
class Config:
    """" Configuration Settings """
    OUTPUT_DIR: str = "output"
    DATASET_NAME: str = "adityachandrasekhar/image-super-resolution"
    KAGGLE_JSON_PATH: str = os.path.join(os.path.dirname(__file__), "kaggle.json")
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 10
    IMG_SIZE: int = 64
    CONTENT_LOSS_WEIGHT: float = 1.0
    PRINT_INTERVAL = 100
    SAVE_INTERVAL = 500
    SAVE_MODEL_INTERVAL = 1

    
if __name__ == "__main__":
    config = Config()
    
