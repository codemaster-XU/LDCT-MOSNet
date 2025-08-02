import torch
import torch.nn as nn
class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2, 0)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4= nn.Conv2d(64, 128, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x1 = x
        x = self.conv2(x)
        x = self.pool(x)
        x2 = x
        x = self.conv3(x)
        x = self.pool(x)
        x3 = x
        x = self.conv4(x)
        x = self.pool(x)
        x4 = x
        """
            x1: torch.Size([1, 16, 255, 255])
            x2: torch.Size([1, 32, 126, 126])
            x3: torch.Size([1, 64, 62, 62])
            x4: torch.Size([1, 128, 30, 30])
        """
        # print("x1:",x1.shape)
        # print("x2:", x2.shape)
        # print("x3:", x3.shape)
        # print("x4:", x4.shape)
        return x1, x2, x3,x4

class BaseDecoder(nn.Module):
    def __init__(self):
        super(BaseDecoder, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = nn.Conv2d(128, 64, 1)
        self.conv_up2 = nn.Conv2d(64, 32, 1)
        self.conv_up3 = nn.Conv2d(32, 16, 1)
        self.conv_up4 = nn.Conv2d(16, 16, 1)

    def forward(self, x1, x2, x3,x4):
        x4 = self.conv_up1(x4)
        x4 = self.upsample1(x4)
        x3 = x3 + x4
        x3 = self.conv_up2(x3)
        x3 = self.upsample2(x3)
        x2 = x2 + x3
        x2 = self.conv_up3(x2)
        x2 = self.upsample3(x2)
        x1 = x1 + x2
        x1 = self.upsample3(x1)
        return x1

class MiniNet(nn.Module):
    def __init__(self, num_class):
        super(MiniNet, self).__init__()
        self.encoder = BaseEncoder()
        self.decoder = BaseDecoder()
        self.final_conv = nn.Conv2d(16, num_class, 1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x = self.decoder(x1, x2, x3,x4)
        x = self.final_conv(x)
        return x

# model = EDFormer(10)
# input_tensor = torch.randn(1, 3, 512, 512)
# output = model(input_tensor)
# print(output.shape)
# count_parameters(model)