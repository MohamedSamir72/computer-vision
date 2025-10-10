from dataclasses import dataclass

@dataclass
class Config:
    """" Configuration Settings """
    OUTPUT_DIR: str = "output"
    DATASET_NAME: str = "adityachandrasekhar/image-super-resolution"
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 10

    
if __name__ == "__main__":
    config = Config()
    