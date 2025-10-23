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


