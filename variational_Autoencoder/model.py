import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False) if not transpose 
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        return x
    
class Conv_VAE(nn.Module):
    def __init__(self, in_ch=1, latent_channels=4):
        super().__init__()
        # Encoder Block
        self.encoder = nn.Sequential(
            ConvBlock(in_channels=in_ch, out_channels=8, kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
        )

        """ 
        Latent space with 2 conv layers for mu and logvar 
        """
        # self.mu = nn.Conv2d(in_channels=32, out_channels=latent_channels, kernel_size=1, stride=1)
        # self.logvar = nn.Conv2d(in_channels=32, out_channels=latent_channels, kernel_size=1, stride=1)

        """ 
        Latent space with only one conv layer for mu and logvar 
        """
        self.conv_latent = nn.Conv2d(in_channels=32, out_channels=2*latent_channels, kernel_size=1, stride=1)

        # Decoder Block
        self.decoder = nn.Sequential(
            ConvBlock(in_channels=latent_channels, out_channels=16, kernel_size=3, stride=2, padding=1, transpose=True),
            ConvBlock(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, transpose=True),
            nn.ConvTranspose2d(in_channels=8, out_channels=in_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward_encoder(self, x):
        x = self.encoder(x)
        """ 
        Old latent space with 2 conv layers for mu and logvar 
        """
        # mu = self.mu(x)
        # logvar = self.logvar(x)

        """ 
        New latent space with only one conv layer for mu and logvar 
        """
        mu, logvar = torch.chunk(self.conv_latent(x), 2, dim=1)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + (eps * std)

        return z, mu, logvar

    def forward_decoder(self, x):
        return self.decoder(x)

    def forward(self, x):
        z, mu, logvar = self.forward_encoder(x)
        # print("Latent z shape:", z.shape)

        x_recon = self.forward_decoder(z)
        # print("Reconstructed x shape:", x_recon.shape)
        return x_recon, mu, logvar


if __name__ == "__main__":
    model = Conv_VAE()
    x = torch.randn((4, 1, 32, 32))
    model(x)