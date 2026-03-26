import torch
from models.unet import UNet

model = UNet().cuda()
x = torch.randn(1, 3, 256, 256).cuda()
y = model(x)

print("Output shape:", y.shape)
