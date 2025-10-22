import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image
from torchvision.models import vgg19, VGG19_Weights
from model import Generator, Discriminator
from utils.config import Config
from dataset import MapDataset
import os

# Use VGG19 pretrained on ImageNet for perceptual loss
class VGGPretrained(nn.Module):
    def __init__(self):
        super(VGGPretrained, self).__init__()
        weights = VGG19_Weights.DEFAULT
        vgg = vgg19(weights=weights).features
        self.vgg = nn.Sequential(*list(vgg.children())[:35])
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.vgg(x)

# Loss Functions
adversarial_loss = nn.BCEWithLogitsLoss()  # GAN Loss (Binary Cross-Entropy)
mse_loss = nn.MSELoss()  # Mean Squared Error for Content Loss

def train():
    # Ensure output directories exist
    os.makedirs("images", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Hyperparameters
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config.BATCH_SIZE
    num_epochs = config.NUM_EPOCHS
    learning_rate = config.LEARNING_RATE
    base_dir = os.getcwd()
    
    dataset = MapDataset(high_res_dir=os.path.join(base_dir, "datasets", "dataset", "train", "high_res"),
                         low_res_dir=os.path.join(base_dir, "datasets", "dataset", "train", "low_res"),
                         img_size=config.IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Optimizers
    optimizer_G = Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # VGG Model for Perceptual Loss (optional)
    vgg_model = VGGPretrained().to(device)  # Assuming you have a VGG model defined for perceptual loss
    
    for epoch in range(num_epochs):
        for i, (lr_images, hr_images) in enumerate(dataloader):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            
            # ============================
            # Train Discriminator
            # ============================
            optimizer_D.zero_grad()

            # Real images
            real_preds = discriminator(hr_images)
            d_loss_real = adversarial_loss(real_preds, torch.ones_like(real_preds))

            # Fake images
            fake_images = generator(lr_images)
            fake_preds = discriminator(fake_images.detach())  # Detach to avoid training generator
            d_loss_fake = adversarial_loss(fake_preds, torch.zeros_like(fake_preds))

            # Total Discriminator Loss
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # ============================
            # Train Generator
            # ============================
            optimizer_G.zero_grad()

            # Adversarial loss (trying to fool the discriminator)
            fake_preds = discriminator(fake_images)
            g_loss_adv = adversarial_loss(fake_preds, torch.ones_like(fake_preds))

            # Content loss (Perceptual Loss from VGG)
            # Resize tensors before VGG
            fake_resized = F.interpolate(fake_images, size=(224, 224), mode='bilinear', align_corners=False)
            real_resized = F.interpolate(hr_images, size=(224, 224), mode='bilinear', align_corners=False)

            # Pass to VGG
            fake_features = vgg_model(fake_resized)
            real_features = vgg_model(real_resized)
            g_loss_content = mse_loss(fake_features, real_features)

            # Total Generator Loss
            g_loss = g_loss_adv + config.CONTENT_LOSS_WEIGHT * g_loss_content
            g_loss.backward()
            optimizer_G.step()

            # Print stats every few steps
            if i % config.PRINT_INTERVAL == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

            # Save generated images every few steps
            if i % config.SAVE_INTERVAL == 0:
                save_image(fake_images.data[:25], f"images/{epoch}_{i}.png", nrow=5, normalize=True)

        # Save models after each epoch
        if (epoch + 1) % config.SAVE_MODEL_INTERVAL == 0:
            torch.save(generator.state_dict(), f"models/generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"models/discriminator_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
