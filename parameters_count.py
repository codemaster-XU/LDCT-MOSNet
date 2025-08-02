import torch
from nets.deeplabv3_plus_DenseAspp import DeepLab

def count_parameters(model):
    model_parameters = model.parameters()
    print("Model Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {torch.numel(param):.0f} parameters")
    num_params = sum([torch.numel(p) for p in model_parameters if p.requires_grad])
    num_params_millions = num_params / 1e6  # 转换为百万单位
    print(f"\nTotal number of trainable parameters: {num_params_millions:.2f}M")  # 输出保留两位小数

count_parameters(DeepLab(10,'mobilenet'))
"""
 DeepLabV3+-MobileNetV2     5.82M
 DeepLabV3+cbam+se-MobileNetV2     4.65M
 DeepLabV3+cbam-MobileNetV2-improved     11.8M
 DeepLabV3+-Xception    54.71M     
 Resnet101      59.34M
 SE_Resnet101   64.09M  
 SETR_Naive_S   88.00M
 SETR_Naive_L   305.13M
 SETR_Naive_H   635.16M
"""


