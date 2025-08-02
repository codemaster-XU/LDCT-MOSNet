import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
from nets.resnet import ResNet101,SE_ResNet101


class MobileNetV2(nn.Module):
	def __init__(self, downsample_factor=8, pretrained=True):
		super(MobileNetV2, self).__init__()
		from functools import partial

		model           = mobilenetv2(pretrained)
		self.features   = model.features[:-1]

		self.total_idx  = len(self.features)
		self.down_idx   = [2, 4, 7, 14]

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

	def _nostride_dilate(self, m, dilate):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
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
		return low_level_features, x


import torch
import torch.nn as nn
import torch.nn.functional as F



# class SEBlock(nn.Module):
# 	def __init__(self, channel, reduction=16):
# 		super(SEBlock, self).__init__()
# 		self.avg_pool = nn.AdaptiveAvgPool2d(1)
# 		self.fc = nn.Sequential(
# 			nn.Linear(channel, channel // reduction, bias=False),
# 			nn.ReLU(inplace=True),
# 			nn.Linear(channel // reduction, channel, bias=False),
# 			nn.Sigmoid()
# 		)
#
# 	def forward(self, x):
# 		b, c, _, _ = x.size()
# 		y = self.avg_pool(x).view(b, c)
# 		y = self.fc(y).view(b, c, 1, 1)
# 		return x * y.expand_as(x)


class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
			nn.BatchNorm2d(dim_out, momentum=bn_mom),
			nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
			nn.BatchNorm2d(dim_out, momentum=bn_mom),
			nn.ReLU(inplace=True),
		)
		self.branch3 = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
			nn.BatchNorm2d(dim_out, momentum=bn_mom),
			nn.ReLU(inplace=True),
		)
		self.branch4 = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
			nn.BatchNorm2d(dim_out, momentum=bn_mom),
			nn.ReLU(inplace=True),
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
			nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
			nn.BatchNorm2d(dim_out, momentum=bn_mom),
			nn.ReLU(inplace=True),
		)
		# self.se = SEBlock(dim_out)

	def forward(self, x):
		[b, c, row, col] = x.size()
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
		global_feature = torch.mean(x, 2, True)
		global_feature = torch.mean(global_feature, 3, True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		# result = self.se(result)
		return result


class DeepLab(nn.Module):
	def __init__(self, num_classes, backbone="se_resnet101", pretrained=True, downsample_factor=16):
		super(DeepLab, self).__init__()
		if backbone == "xception":
			self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
			in_channels = 2048
			low_level_channels = 256
		elif backbone == "mobilenet":
			print("mobilenet")
			self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
			in_channels = 320
			low_level_channels = 24
		elif backbone == "resnet101":
			print("resnet101")
			self.backbone = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=pretrained, output_stride=8)
			in_channels = 2048
			low_level_channels = 256
		elif backbone == "se_resnet101":
			self.backbone = SE_ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=pretrained, output_stride=8)
			in_channels = 2048
			low_level_channels = 256
		else:
			raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception'.format(backbone))

		self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)
		self.shortcut_conv = nn.Sequential(
			nn.Conv2d(low_level_channels, 48, 1),
			nn.BatchNorm2d(48),
			nn.ReLU(inplace=True)
		)

		self.cat_conv = nn.Sequential(
			nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),

			nn.Conv2d(256, 256, 3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),

			nn.Dropout(0.1),
		)
		self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

	def forward(self, x):
		"""
		输入形状 torch.Size([1, 3, 500, 500])
		输入形状宽高HW 500 500
		骨干网浅层形状: torch.Size([1, 24, 125, 125])
		骨干网深层形状: torch.Size([1, 320, 32, 32])
		"""
		# print("输入形状",x.shape)
		H, W = x.size(2), x.size(3)
		# print("输入形状宽高HW",H,W)
		low_level_features, x = self.backbone(x)
		# print("骨干网浅层形状:",low_level_features.shape)
		# print("骨干网深层形状:", x.shape)
		x = self.aspp(x)
		# print("aspp层输出形状:", x.shape)

		# 1x1卷积固定通道数 对low_level_features改变通道数为48，不改变WH
		low_level_features = self.shortcut_conv(low_level_features)
		# print("扩张骨干网浅层形状通道固定48:",low_level_features.shape)
		# 将ASPP层的输出x上采样到与low_level_features相同的尺寸。
		x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
						  align_corners=True)
		# print("骨干网深层形状上采样形状:",x.shape)

		# 上采样之后对 x 和 low_feature进行拼接
		x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
		# print("拼接形状",x.shape)
		# 分类卷积 输出维度 num_classes
		x = self.cls_conv(x)
		# print("分类结果形状",x.shape)
		x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
		# print("最终上采样到H,W的结果", x.shape)
		return x