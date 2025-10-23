import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) if not transpose else nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        print(x.shape)
        return x
    
class VAE(nn.Module):
    def __init__(self, in_ch, latent_channels=2):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels=in_ch, out_channels=32, kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
        )
        self.mu = nn.Conv2d(128, latent_channels, 3, 1)
        self.logvar = nn.Conv2d(128, latent_channels, 3, 1)

        self.decoder = nn.Sequential(
            ConvBlock(in_channels=latent_channels, out_channels=128, kernel_size=3, stride=1, padding=1, transpose=True),
            ConvBlock(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, transpose=True),
            ConvBlock(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, transpose=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=in_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

if __name__ == "__main__":
    model = VAE(in_ch=1)
    x = torch.randn((2, 1, 32, 32))
    model.encoder(x)