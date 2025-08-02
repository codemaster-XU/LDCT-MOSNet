# 医学图像分割BIBM
# author: Zongqi Xu
# date: 2025/7/31
# SouthWest Minzu University

### 数据集

本研究基于中国河南省某三甲医院及其医联体（涵盖省内多家农村医院、县级医院及三甲医院）的多中心慢阻肺普筛CT数据集，共纳入2000例受试者。数据来源覆盖不少于12个区域的标准化CT扫描，并已通过伦理审查（伦理号：2024HL-040）。数据具有典型的多中心研究的共性挑战：样本不平衡、放射剂量偏差、配准困难等，研究团队对2000例患者的CT影像进行了精细标注，涵盖29类器官/组织标签（详见正文），其中1067例已记录完整的设备型号与扫描参数。
LDCT-MOSnet在独立外部数据集LIDC-IDRI（肺癌分割）与TCIA（肺结节分割）上进行泛化测试。

### 本项目实现和比较了以下几种模型：

改进
- DeepLabV3+ MobileNetV2
- DSDeepLab（改进） MobileNetV2
- DenseASSP-DeeplabV3
- MMulti-Scale Hierarchical Feature Reuse​​-DeeplabV3
- Composite Loss Function with Dynamic Edge Awareness and Multi-Scale Confidence Weighting​-DeeplabV3

比较
- AgileFormer
- mmcv
- nnUNet
- ParaTransCNN
- SACnet
- SwinTansformer
- Unet
- Unet++
- ARU-Net

## 环境要求 
python
python 3.8
cuda 12.1
torch 2.1.0
NVIDIA RTX3090 （24GB VRAM per GPU）*8

## 项目结构（根据具体情况切换参数并更换模型的局部模块）
├── logs/ # 日志（训练和预测的日志）
├── nets/ # 网络模型定义
├── utils/ # 工具函数
├── segment_anything # sam2库
├── model_data # 基础模型权重
├── VOCdevkit # 数据集
├── train.py # 训练脚本-要根据模型更改切换参数和模块
├── train.py # MiniNet训练脚本-要根据模型更改切换参数和模块
├── predict.py # 预测脚本
└── parameters_count.py # 参数统计脚本

- 相关论文正在投稿BIBM 2025
