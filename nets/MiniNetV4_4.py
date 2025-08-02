import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn
import torch
from parameters_count import count_parameters
BatchNorm2d = nn.BatchNorm2d

class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.point_wise_up = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
            self.depth_wise = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
            self.point_wise_down = nn.Sequential(
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            residual = x
            x = self.point_wise_up(x)
            x = self.depth_wise(x)
            x = self.point_wise_down(x)
            return residual + x
        else:
            x = self.conv(x)
            return x

class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3,16,3,1,1)

        self.block1 = InvertedResidualBlock(16,32,1,1)
        self.block2 = InvertedResidualBlock(32, 32, 1, 2)
        self.block3 = InvertedResidualBlock(32, 32, 1, 3)
        self.block4 = InvertedResidualBlock(32, 32, 1, 4)
        self.block5 = InvertedResidualBlock(32, 32, 1, 5)
        self.block6 = InvertedResidualBlock(32, 32, 1, 6)
        self.block7 = InvertedResidualBlock(32, 32, 1, 7)


        self.block8 = InvertedResidualBlock(32,64,1,1)
        self.block9 = InvertedResidualBlock(64, 128, 1, 1)
        self.block10 = InvertedResidualBlock(128, 256, 1, 1)

        self.conv2 = nn.Conv2d(256,512,1,1)
        self.conv3 = nn.Conv2d(512,1024,1,1)
        self.pool = nn.MaxPool2d(2, 2, 0)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x1 = x # 32
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.pool(x)
        x2 = x # 32
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.pool(x)
        x3 = x #256
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x4 = x #512
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
        self.conv_up1 = nn.Conv2d(1024, 256, 1)
        self.conv_up2 = nn.Conv2d(256, 32, 1)
        self.conv_up3 = nn.Conv2d(32, 16, 1)

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

# model = MiniNet(10)
# input_tensor = torch.randn(1, 3, 512, 512)
# output = model(input_tensor)
# print(output.shape)
# count_parameters(model)