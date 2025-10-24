import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from config import Config
from model import Conv_VAE
from dataset import Map_Dataset
from tqdm import tqdm
import os

def VAE_loss(recon_x, x, mu, logvar, reconstruction_loss_weight=1.0, kld_loss_weight=1.0):
    # Reconstruction loss
    mse = (recon_x - x) ** 2
    mse = mse.flatten(1).sum(dim=1)
    mse = mse.mean()
    
    # KL Divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1)
    kl_divergence = kl_divergence.mean()

    return (reconstruction_loss_weight * mse) + (kld_loss_weight * kl_divergence)

def train(
        model, 
        device, 
        train_dataset,
        test_dataset, 
        batch_size, 
        learning_rate, 
        epochs,
        reconstruction_loss_weight,
        kld_loss_weight,
        save_path="best_model.pth"):

    model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss tracking
    train_losses, eval_losses = [], []
    best_eval_loss = float('inf')  # Initialize with infinity    

    print(f"Starting training for {epochs} epochs on {device}...")

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        # === TRAINING ===
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}] - Training", leave=False)
        for images, _ in progress_bar:
            images = images.to(device)

            x_recon, mu, logvar = model(images)

            loss = VAE_loss(x_recon, images, mu, logvar,
                            reconstruction_loss_weight, kld_loss_weight)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate batch loss
            running_train_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({'batch_loss': loss.item()})
        
        # Average training loss
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # === EVALUATION ===
        model.eval()
        running_eval_loss = 0.0

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                x_recon, mu, logvar = model(images)

                loss = VAE_loss(x_recon, images, mu, logvar,
                                reconstruction_loss_weight, kld_loss_weight)
                running_eval_loss += loss.item() * images.size(0)

        # Average eval loss
        epoch_eval_loss = running_eval_loss / len(test_loader.dataset)
        eval_losses.append(epoch_eval_loss)

        # === LOGGING ===
        print(f"Epoch [{epoch}/{epochs}] | "
              f"Train Loss: {epoch_train_loss:.6f} | "
              f"Eval Loss: {epoch_eval_loss:.6f}")
        
        
        # === SAVE BEST MODEL ===
        if epoch_eval_loss < best_eval_loss:
            best_eval_loss = epoch_eval_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model at epoch {epoch} (Eval Loss: {best_eval_loss:.6f})")

    print("Training complete.")
    print(f"Best model saved to: {os.path.abspath(save_path)}")
    return train_losses, eval_losses



if __name__ == "__main__":
    train_data = MNIST(root='./data', train=True, download=True)
    test_data = MNIST(root='./data', train=False, download=True)
    train_dataset = Map_Dataset(data=train_data.data, labels=train_data.targets)
    test_dataset = Map_Dataset(data=test_data.data, labels=test_data.targets)

    model = Conv_VAE()
    train(
        model=model,
        device=Config.DEVICE,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        reconstruction_loss_weight=Config.RECONSTRUCTION_LOSS_WEIGHT,
        kld_loss_weight=Config.KLD_LOSS_WEIGHT
    )
