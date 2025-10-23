import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from config import Config
from model import Conv_VAE
from dataset import Map_Dataset
from tqdm import tqdm

def VAE_loss(recon_x, x, mu, logvar, reconstruction_loss_weight=1.0, kld_loss_weight=1.0):
    # Reconstruction loss
    mse = (recon_x - x) ** 2
    mse = mse.flatten(1).sum(dim=1)
    mse = mse.mean()
    
    # KL Divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1)
    kl_divergence = kl_divergence.mean()

    return (reconstruction_loss_weight * mse) + (kld_loss_weight * kl_divergence)

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

    print("Starting Training...")
    for epoch in range(config.EPOCHS):
        total_loss = 0
        for batch_idx, (data, _) in tqdm(enumerate(train_loader)):
            data = data.to(config.DEVICE)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)

            loss = VAE_loss(recon_batch, data, mu, logvar, 
                            config.RECONSTRUCTION_LOSS_WEIGHT, 
                            config.KLD_LOSS_WEIGHT)
            loss.backward()
            total_loss += loss
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset)}')


if __name__ == "__main__":
    train()


    # # handle random seed for reproducibility
    # torch.manual_seed(0)

    # x = torch.randn((4, 1, 28, 28))
    # x_hat = torch.randn((4, 1, 28, 28))
    # mu = torch.randn((4, 2))
    # logvar = torch.randn((4, 2))

    # loss = VAE_loss(x_hat, x, mu, logvar,
    #                 Config.RECONSTRUCTION_LOSS_WEIGHT,
    #                 Config.KLD_LOSS_WEIGHT)
    # # print("Loss:", loss)