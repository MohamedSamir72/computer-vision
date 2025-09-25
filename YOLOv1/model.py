import torch
from torch import nn
from utils.config import Config

class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.conv(x)   


class YOLOv1(nn.Module):
    def __init__(self, in_ch=3, **kwargs):
        super().__init__()
        self.layers = self.conv_layers(in_ch)
        # print(self.layers)
        self.fc = self.fc_layer(**kwargs)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, start_dim=1)

        return self.fc(x)

    def conv_layers(self, in_ch):
        layers = []
        in_ch = in_ch
        for i in range(len(Config.ARCHITECTURE)):
            layer = Config.ARCHITECTURE[i]
            
            if type(layer) == tuple:
                layers += [CNNBlock(in_ch=in_ch, 
                                    out_ch=layer[1], 
                                    kernel_size=layer[0], 
                                    stride=layer[2], 
                                    padding=layer[3])]
                in_ch = layer[1]
                
            elif type(layer) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(layer) == list:
                conv1 = layer[0]
                conv2 = layer[1]
                num_repeats = layer[2]

                for _ in range(num_repeats):
                    layers += [CNNBlock(in_ch=in_ch, 
                                    out_ch=conv1[1], 
                                    kernel_size=conv1[0], 
                                    stride=conv1[2], 
                                    padding=conv1[3])]
                    in_ch = conv1[1]
                    
                    layers += [CNNBlock(in_ch=in_ch, 
                                    out_ch=conv2[1], 
                                    kernel_size=conv2[0], 
                                    stride=conv2[2], 
                                    padding=conv2[3])]
                    in_ch = conv2[1]     

        return nn.Sequential(*layers)
    
    def fc_layer(self, split_size, num_boxes, num_classes):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(split_size*split_size*1024, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, split_size*split_size*(num_boxes*5 + num_classes))
        )
    
if __name__ == "__main__":
    model = YOLOv1(3, split_size=7, num_boxes=2, num_classes=20)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)