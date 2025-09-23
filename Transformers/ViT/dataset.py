from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from config import Config
import os

def get_dataloaders():
    ### Data transformations
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    ### Datasets and DataLoaders
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    return train_dataloader, val_dataloader

if __name__ == "__main__":
    if not os.path.exists('./models'):
        os.makedirs('./models')