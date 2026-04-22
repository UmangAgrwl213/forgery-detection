import torch
import torch.nn as nn
import torchvision.models as models

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class ResNetUNet(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()

        self.base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1 = nn.Sequential(*self.base_layers[3:4]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1 = nn.Sequential(*self.base_layers[4]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = nn.Sequential(*self.base_layers[5]) # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = nn.Sequential(*self.base_layers[6]) # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = nn.Sequential(*self.base_layers[7]) # size=(N, 512, x.H/32, x.W/32)

        self.decode4 = DecoderBlock(512, 256, 256)
        self.decode3 = DecoderBlock(256, 128, 128)
        self.decode2 = DecoderBlock(128, 64, 64)
        self.decode1 = DecoderBlock(64, 64, 64)
        self.decode0 = DecoderBlock(64, 0, 32)

        self.final_conv = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, input):
        e0 = self.layer0(input)    # 112x112, 64
        e0_1 = self.layer0_1(e0)   # 56x56, 64
        e1 = self.layer1(e0_1)     # 56x56, 64
        e2 = self.layer2(e1)       # 28x28, 128
        e3 = self.layer3(e2)       # 14x14, 256
        e4 = self.layer4(e3)       # 7x7, 512

        d4 = self.decode4(e4, e3)  # 14x14, 256
        d3 = self.decode3(d4, e2)  # 28x28, 128
        d2 = self.decode2(d3, e1)  # 56x56, 64
        d1 = self.decode1(d2, e0)  # 112x112, 64
        d0 = self.decode0(d1)      # 224x224, 32

        return self.final_conv(d0)
