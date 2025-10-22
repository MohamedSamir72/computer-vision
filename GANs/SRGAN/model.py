import torch
from torch import nn

##### Conv block #####
class ConvBlock(nn.Module):
    def __init__(self, 
                 in_ch, 
                 out_ch, 
                 kernel_size, 
                 stride, 
                 padding,
                 act_type='leaky_relu',
                 pixel_shuffle=False):       
        super().__init__()

        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)]

        if pixel_shuffle:
            layers.append(nn.PixelShuffle(2))
            layers.append(nn.PReLU() if act_type == 'prelu' else nn.LeakyReLU(0.2))
        else:
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.PReLU() if act_type == 'prelu' else nn.LeakyReLU(0.2))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

##### Residual block #####
class ResBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 stride,
                 padding):
        super().__init__()

        self.res = nn.Sequential(
            ConvBlock(in_ch, out_ch, kernel_size, stride, padding, act_type='prelu'),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return x + self.res(x)  # Skip connection

##### Discriminator block #####
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        out_ch = 64

        self.initial = nn.Sequential(
            nn.Conv2d(3, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        self.conv_block = ConvBlock(out_ch, out_ch, kernel_size=3, stride=2, padding=1, act_type='leaky_relu')

        self.conv_blocks = []
        for _ in range(3):
            self.conv_blocks.append(ConvBlock(out_ch, out_ch*2, kernel_size=3, stride=1, padding=1, act_type='leaky_relu'))
            self.conv_blocks.append(ConvBlock(out_ch*2, out_ch*2, kernel_size=3, stride=2, padding=1, act_type='leaky_relu'))

            out_ch *= 2

        self.conv_blocks = nn.Sequential(*self.conv_blocks)

        # Dimensionality reduction to be suitable with different input sizes (images)
        self.gap = nn.AdaptiveAvgPool2d((3, 3)) 

        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_ch*3*3, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.conv_block(x)
        x = self.conv_blocks(x)
        x = self.gap(x)
        
        return torch.sigmoid(self.final(x))
    
##### Generator block #####
class Generator(nn.Module):
    def __init__(self,
                 in_ch=3,
                 out_ch=3):
        super().__init__()

        self.initial = nn.Conv2d(in_ch, 64, kernel_size=9, stride=1, padding=4)
        self.act1 = nn.PReLU()
        self.res_blocks = nn.Sequential(
            *[ResBlock(64, 64, kernel_size=3, stride=1, padding=1) for _ in range(16)]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.upsample1 = ConvBlock(64, 256, kernel_size=3, stride=1, padding=1, act_type='prelu', pixel_shuffle=True)
        self.upsample2 = ConvBlock(64, 256, kernel_size=3, stride=1, padding=1, act_type='prelu', pixel_shuffle=True)
        self.final = nn.Conv2d(64, out_ch, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.act1(self.initial(x))
        x = self.res_blocks(initial)
        x = self.conv2(x)
        x = x + initial
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.final(x)

        return torch.tanh(x)
    

if __name__ == "__main__":
    """
    Test the Generator and Discriminator models
    """    
    
    x = torch.randn((1, 3, 24, 24))

    model = Generator()
    out = model(x)
    print(f"Output of Generator: {out.shape}")

    model2 = Discriminator()
    out2 = model2(x)
    print(f"Output of Disciminator: {out2.shape}")

