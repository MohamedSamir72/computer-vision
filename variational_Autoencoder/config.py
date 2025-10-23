from dataclasses import dataclass

@dataclass
class Config:
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 64
    EPOCHS: int = 10
    DEVICE: str = 'cuda'
    RECONSTRUCTION_LOSS_WEIGHT: float = 1.0
    KLD_LOSS_WEIGHT: float = 1.0
    
