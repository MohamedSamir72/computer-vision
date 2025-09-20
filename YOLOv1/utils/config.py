from dataclasses import dataclass
import torch

@dataclass
class Config:
    """
    Configuration settings
    """
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 4
    epochs: int = 100
    learning_rate: float = 0.001
    
