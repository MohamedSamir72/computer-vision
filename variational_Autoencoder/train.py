import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from config import Config
from model import Conv_VAE
from dataset import Map_Dataset

def VAE_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    MSE = nn.MSELoss(reduction='sum')(recon_x, x)

    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def train():
    # Load Configurations
    config = Config()
    Config.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Dataset
    train_data = MNIST(root='./data', train=True, download=True)
    train_dataset = Map_Dataset(data=train_data.data, labels=train_data.targets)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Initialize Model, Optimizer
    model = Conv_VAE()
    model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    model.train()

    for epoch in range(config.EPOCHS):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(config.DEVICE)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)

            loss = VAE_loss(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset)}')


if __name__ == "__main__":
    train()