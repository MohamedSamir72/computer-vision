from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch

class Map_Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        super().__init__()
        self.data = data
        self.labels = labels
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  
                transforms.Resize((32, 32)),  
                transforms.Normalize((0.5,), (0.5,))  
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the sample
        sample = self.data[idx]
        label = self.labels[idx]
        
        # Ensure sample is on CPU and convert to numpy if it's a tensor
        if isinstance(sample, torch.Tensor):
            sample = sample.cpu().numpy()
            
        # Convert to PIL Image (mode 'L' for grayscale)
        sample_pil = Image.fromarray(sample.astype('uint8'), mode='L')

        sample = self.transform(sample_pil)

        return sample, label

