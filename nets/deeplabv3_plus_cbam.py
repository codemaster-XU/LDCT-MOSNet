import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
import torch.utils.model_zoo as model_zoo

class SELayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y.expand_as(x)

class ConvBNReLU(nn.Sequential):
	def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
		padding = (kernel_size - 1) // 2
		super(ConvBNReLU, self).__init__(
			nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
			nn.BatchNorm2d(out_planes),
			nn.ReLU6(inplace=True)
		)
		self.stride = stride
		self.kernel_size = kernel_size

class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
		super(InvertedResidual, self).__init__()
		self.stride = stride
		assert stride in [1, 2]

		hidden_dim = int(round(inp * expand_ratio))
		self.use_res_connect = self.stride == 1 and inp == oup

		# 添加SE模块
		self.se = SELayer(hidden_dim)

		layers = []
		if expand_ratio != 1:
			# pw
			layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
		
		layers.extend([
			# dw
			ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
			# SE attention
			self.se,
			# pw-linear
			nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
			nn.BatchNorm2d(oup),
		])
		self.conv = nn.Sequential(*layers)

	def forward(self, x):
		if self.use_res_connect:
			return x + self.conv(x)
		return self.conv(x)

class MobileNetV2(nn.Module):
	def __init__(self, downsample_factor=8, pretrained=True):
		super(MobileNetV2, self).__init__()
		from functools import partial
		
		width_mult = 1.
		block = InvertedResidual
		input_channel = 32
		last_channel = 320
		inverted_residual_setting = [
			# t, c, n, s
			[1, 16, 1, 1],   # 1/2
			[6, 24, 2, 2],   # 1/4
			[6, 32, 3, 2],   # 1/8
			[6, 64, 4, 2],   # 1/16
			[6, 96, 3, 1],   # 1/16
			[6, 160, 3, 2],  # 1/32
			[6, 320, 1, 1],  # 1/32
		]

		# building first layer
		input_channel = int(input_channel * width_mult)
		self.last_channel = int(last_channel * max(1.0, width_mult))
		
		features = [ConvBNReLU(3, input_channel, stride=2)]
		
		# building inverted residual blocks
		for t, c, n, s in inverted_residual_setting:
			output_channel = int(c * width_mult)
			for i in range(n):
				stride = s if i == 0 else 1
				features.append(block(input_channel, output_channel, stride, expand_ratio=t))
				input_channel = output_channel

		self.features = nn.Sequential(*features)
		self.total_idx = len(self.features)
		self.down_idx = [2, 4, 7, 14]

		# 修改下采样策略
		if downsample_factor == 8:
			for i in range(self.down_idx[-2], self.down_idx[-1]):
				self.features[i].apply(
					partial(self._nostride_dilate, dilate=2)
				)
			for i in range(self.down_idx[-1], self.total_idx):
				self.features[i].apply(
					partial(self._nostride_dilate, dilate=4)
				)
		elif downsample_factor == 16:
			for i in range(self.down_idx[-1], self.total_idx):
				self.features[i].apply(
					partial(self._nostride_dilate, dilate=2)
				)

		# 添加通道注意力
		self.channel_attention = nn.Sequential(
			SELayer(self.last_channel),
			CBAM(self.last_channel, ratio=8)
		)

		# 移除预训练权重加载
		# if pretrained:
		#     self._load_pretrained_model()

	def _nostride_dilate(self, m, dilate):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			# 检查是否是ConvBNReLU类
			if isinstance(m, ConvBNReLU):
				conv = m[0]  # 获取Conv2d层
				if conv.stride == (2, 2):
					conv.stride = (1, 1)
					if conv.kernel_size == (3, 3):
						conv.dilation = (dilate//2, dilate//2)
						conv.padding = (dilate//2, dilate//2)
				else:
					if conv.kernel_size == (3, 3):
						conv.dilation = (dilate, dilate)
						conv.padding = (dilate, dilate)
			# 处理普通Conv2d层
			elif hasattr(m, 'stride'):
				if m.stride == (2, 2):
					m.stride = (1, 1)
					if m.kernel_size == (3, 3):
						m.dilation = (dilate//2, dilate//2)
						m.padding = (dilate//2, dilate//2)
				else:
					if m.kernel_size == (3, 3):
						m.dilation = (dilate, dilate)
						m.padding = (dilate, dilate)

	def forward(self, x):
		low_level_features = self.features[:4](x)
		x = self.features[4:](low_level_features)
		x = self.channel_attention(x)
		return low_level_features, x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        
        # 1x1卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        
        # 使用标准3x3卷积替代深度可分离卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, padding=6*rate, dilation=6*rate, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, padding=12*rate, dilation=12*rate, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, padding=18*rate, dilation=18*rate, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        
        # 全局上下文分支
        self.branch5_pool = nn.AdaptiveAvgPool2d(1)
        self.branch5_conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )

        # 特征融合
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out*5, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            CBAM(dim_out, ratio=8)  # 保留CBAM注意力机制
        )

    def forward(self, x):
        size = x.size()
        
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        
        x5 = self.branch5_pool(x)
        x5 = self.branch5_conv(x5)
        x5 = F.interpolate(x5, size[2:], mode='bilinear', align_corners=True)
        
        out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.conv_cat(out)
        
        return out

class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone=="xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenet":
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
        # 简低层特征处理
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 简化特征融合
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            CBAM(256, ratio=8)  # 只在最后使用CBAM
        )
        
        self.cls_conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), 
                         mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x 