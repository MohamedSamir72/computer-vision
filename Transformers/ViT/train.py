import torch
from torch import nn
import torch.optim as optim
from model import ViT
from config import Config
from dataset import get_dataloaders
from tqdm import tqdm

print("starting training...")
def train():
    print("getting data loaders...")
    # Get data loaders
    train_dataloader, val_dataloader = get_dataloaders()

    print("training...")
    # Initialize model, loss function, optimizer
    model = ViT(
        in_ch=Config.CHANNELS,
        patch_size=Config.PATCH_SIZE,
        emb_size=Config.EMBEDING_SIZE,
        img_size=Config.IMG_SIZE,
        num_heads=Config.NUM_HEADS,
        mlp_dim=Config.MLP_DIM,
        depth=Config.DEPTH,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROP_OUT
    ).to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    print("starting training loop...")
    # Training loop
    for epoch in tqdm(range(Config.NUM_EPOCHS)):
        model.train()
        running_loss = 0.0
        for images, labels in train_dataloader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}], Loss: {running_loss/len(train_dataloader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), f"{Config.SAVE_PATH}/vit_model.pth")
    print("Model saved!")


print("ending training...")

if __name__ == "__main__":
    train()