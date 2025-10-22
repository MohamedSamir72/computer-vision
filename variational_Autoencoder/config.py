from dataclasses import dataclass

@dataclass
class Config:
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 64
    EPOCHS: int = 50
    
