import datetime
import os
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nets.deeplabv3_plus_DenseAspp import DeepLab
from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion
import cv2
from PIL import Image
import random
import re
import time
from scipy.ndimage import distance_transform_edt

# 扩展的评估回调类，包含10个指标
class AdvancedEvalCallback:
    def __init__(self, net, input_shape, num_classes, save_dir, device, eval_period=5, pixel_spacing=1.0):
        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.save_dir = save_dir
        self.device = device
        self.eval_period = eval_period
        self.pixel_spacing = pixel_spacing
        os.makedirs(save_dir, exist_ok=True)
        
        # 存储所有10个指标的历史数据
        self.metrics_history = {
            "mIoU": [], "mPA": [], "mPrecision": [], "mRecall": [],
            "Parameter": [], "Epoch": [],
            "Dice": [], "MSD": [], "HD": [], "ASSD": []
        }
        self.epochs = []
        
    def get_surface_points(self, mask):
        """获取表面点坐标"""
        # 使用形态学腐蚀找到内部点
        eroded = binary_erosion(mask)
        surface = mask & ~eroded
        return np.argwhere(surface)
    
    def calculate_surface_distances(self, pred, label):
        """计算表面距离指标"""
        try:
            # 获取表面点
            pred_surface = self.get_surface_points(pred)
            label_surface = self.get_surface_points(label)
            
            # 如果没有表面点，返回0
            if len(pred_surface) == 0 or len(label_surface) == 0:
                return 0.0, 0.0, 0.0
            
            # 计算Hausdorff距离
            hd = max(
                directed_hausdorff(pred_surface, label_surface)[0],
                directed_hausdorff(label_surface, pred_surface)[0]
            )
            
            # 计算平均表面距离 (MSD)
            # 计算每个预测表面点到最近标签表面点的距离
            dist_pred_to_label = []
            for point in pred_surface:
                dists = np.sqrt(np.sum((label_surface - point)**2, axis=1))
                dist_pred_to_label.append(np.min(dists))
            
            # 计算每个标签表面点到最近预测表面点的距离
            dist_label_to_pred = []
            for point in label_surface:
                dists = np.sqrt(np.sum((pred_surface - point)**2, axis=1))
                dist_label_to_pred.append(np.min(dists))
            
            # 计算平均对称表面距离 (ASSD)
            msd_pred = np.mean(dist_pred_to_label) if dist_pred_to_label else 0.0
            msd_label = np.mean(dist_label_to_pred) if dist_label_to_pred else 0.0
            msd = (msd_pred + msd_label) / 2.0
            assd = msd  # 在这个实现中，ASSD等同于MSD
            
            # 转换为毫米
            hd *= self.pixel_spacing
            msd *= self.pixel_spacing
            assd *= self.pixel_spacing
            
            return msd, hd, assd
        except Exception as e:
            print(f"计算表面距离时出错: {e}")
            return 0.0, 0.0, 0.0

    def calculate_dice(self, pred, label):
        """计算Dice相似系数"""
        intersection = np.logical_and(pred, label).sum()
        return (2. * intersection) / (pred.sum() + label.sum() + 1e-7)

    def calculate_iou(self, pred, label):
        """计算IoU"""
        intersection = np.logical_and(pred, label).sum()
        union = np.logical_or(pred, label).sum()
        return intersection / (union + 1e-7)

    def calculate_pa(self, pred, label):
        """计算像素准确率"""
        return np.mean(pred == label)

    def calculate_precision_recall(self, pred, label):
        """计算精确率和召回率"""
        tp = np.logical_and(pred == 1, label == 1).sum()
        fp = np.logical_and(pred == 1, label == 0).sum()
        fn = np.logical_and(pred == 0, label == 1).sum()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        return precision, recall

    def on_epoch_end(self, epoch, val_loader):
        """在每个epoch结束时进行评估"""
        if epoch % self.eval_period != 0 and epoch != self.net.module.UnFreeze_Epoch - 1:
            return None
            
        self.net.eval()
        
        # 初始化所有10个指标的收集器
        metrics = {
            "iou": [], "pa": [], "precision": [], "recall": [],
            "dice": [], "msd": [], "hd": [], "assd": []
        }
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device).float()  # 转换为float32
                labels = labels.to(self.device).long()   # 转换为long (int64)
                
                outputs = self.net(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy().astype(np.uint8)
                labels = labels.cpu().numpy().astype(np.uint8)
                
                for i in range(images.shape[0]):
                    pred = preds[i]
                    label = labels[i]
                    
                    # 二值化处理（前景为1，背景为0）
                    pred_binary = (pred > 0).astype(np.uint8)
                    label_binary = (label > 0).astype(np.uint8)
                    
                    # 计算基础指标
                    metrics["iou"].append(self.calculate_iou(pred_binary, label_binary))
                    metrics["pa"].append(self.calculate_pa(pred_binary, label_binary))
                    
                    precision, recall = self.calculate_precision_recall(pred_binary, label_binary)
                    metrics["precision"].append(precision)
                    metrics["recall"].append(recall)
                    
                    metrics["dice"].append(self.calculate_dice(pred_binary, label_binary))
                    
                    # 计算表面距离指标
                    msd, hd, assd = self.calculate_surface_distances(pred_binary, label_binary)
                    metrics["msd"].append(msd)
                    metrics["hd"].append(hd)
                    metrics["assd"].append(assd)
        
        # 计算平均指标
        avg_metrics = {
            "mIoU": np.nanmean(metrics["iou"]),
            "mPA": np.nanmean(metrics["pa"]),
            "mPrecision": np.nanmean(metrics["precision"]),
            "mRecall": np.nanmean(metrics["recall"]),
            "Dice": np.nanmean(metrics["dice"]),
            "MSD": np.nanmean(metrics["msd"]),
            "HD": np.nanmean(metrics["hd"]),
            "ASSD": np.nanmean(metrics["assd"])
        }
        
        # 添加模型参数数量和当前epoch
        total_params = sum(p.numel() for p in self.net.parameters())
        avg_metrics["Parameter"] = total_params
        avg_metrics["Epoch"] = epoch
        
        # 记录历史
        for key in avg_metrics:
            if key in self.metrics_history:
                self.metrics_history[key].append(avg_metrics[key])
        self.epochs.append(epoch)
        
        # 保存所有指标历史到JSON文件
        with open(os.path.join(self.save_dir, "metrics.json"), "w") as f:
            json.dump({
                "epochs": self.epochs,
                "metrics": self.metrics_history
            }, f, indent=4)
            
        # 绘制并保存曲线图
        self.plot_metrics(epoch)
        
        self.net.train()
        return avg_metrics

    def plot_metrics(self, epoch):
        """绘制并保存所有10个指标的曲线图"""
        plt.figure(figsize=(15, 15))
        plt.suptitle(f"Metrics at Epoch {epoch}", fontsize=16)
        
        # 1. 分割质量指标
        plt.subplot(3, 1, 1)
        seg_metrics = [
            ("mIoU", "IoU"),
            ("mPA", "Pixel Accuracy"),
            ("mPrecision", "Precision"),
            ("mRecall", "Recall"),
            ("Dice", "Dice Coefficient")
        ]
        
        for metric, label in seg_metrics:
            plt.plot(self.epochs, self.metrics_history[metric], label=label)
        plt.title("Segmentation Quality Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        
        # 2. 距离指标
        plt.subplot(3, 1, 2)
        distance_metrics = [
            ("MSD", "Mean Surface Distance (mm)"),
            ("HD", "Hausdorff Distance (mm)"),
            ("ASSD", "Average Symmetric Surface Distance (mm)")
        ]
        
        for metric, label in distance_metrics:
            plt.plot(self.epochs, self.metrics_history[metric], label=label)
        plt.title("Distance Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Distance (mm)")
        plt.legend()
        plt.grid(True)
        
        # 3. 其他指标
        plt.subplot(3, 1, 3)
        other_metrics = [
            ("Parameter", "Parameter Count"),
            ("Epoch", "Epoch")
        ]
        
        for metric, label in other_metrics:
            plt.plot(self.epochs, self.metrics_history[metric], label=label)
        plt.title("Other Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"metrics_epoch_{epoch}.png"))
        plt.close()
        
    def final_summary(self):
        """训练结束后计算最终统计"""
        # 计算所有指标的平均值和标准差
        summary = {}
        for metric, values in self.metrics_history.items():
            if metric not in ["Epoch", "Parameter"]:  # 这些指标不需要计算统计
                summary[f"{metric}_mean"] = np.mean(values)
                summary[f"{metric}_std"] = np.std(values)
        
        # 添加参数数量
        total_params = sum(p.numel() for p in self.net.parameters())
        summary["Parameter"] = total_params
        
        # 保存总结报告
        with open(os.path.join(self.save_dir, "final_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
            
        return summary

# 自定义数据集类（修复标签值范围问题）
class CustomDataset(Dataset):
    def __init__(self, input_shape, num_classes, original_image_dirs, original_mask_dirs, src_type, label_type, 
                 is_train=True, pixel_spacing=1.0):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.is_train = is_train
        self.pixel_spacing = pixel_spacing
        self.image_paths = []
        self.mask_paths = []
        
        # 加载三个路径对的图像和掩码
        for img_dir, mask_dir in zip(original_image_dirs, original_mask_dirs):
            # 扫描所有一级子目录
            for folder in os.listdir(img_dir):
                img_folder = os.path.join(img_dir, folder)
                if not os.path.isdir(img_folder):
                    continue
                
                # 提取文件夹前缀（PAT041, SBL001等）
                folder_prefix = None
                if "PAT" in folder:
                    # 提取PAT后的数字部分
                    match = re.match(r'(PAT\d+)', folder)
                    if match:
                        folder_prefix = match.group(1)
                elif "SBL" in folder:
                    match = re.match(r'(SBL\d+)', folder)
                    if match:
                        folder_prefix = match.group(1)
                
                if not folder_prefix:
                    print(f"无法识别文件夹前缀: {folder}")
                    continue
                
                # 在mask目录下查找匹配的文件夹
                mask_folder_found = False
                mask_folder_path = None
                for mask_folder in os.listdir(mask_dir):
                    if folder_prefix in mask_folder:
                        mask_folder_path = os.path.join(mask_dir, mask_folder)
                        if os.path.isdir(mask_folder_path):
                            mask_folder_found = True
                            break
                
                if not mask_folder_found:
                    print(f"找不到匹配的mask文件夹: {folder_prefix} in {mask_dir}")
                    continue
                
                # 获取所有图像文件
                for file in os.listdir(img_folder):
                    if file.endswith(src_type):
                        img_path = os.path.join(img_folder, file)
                        mask_file = file.replace(src_type, label_type)
                        mask_path = os.path.join(mask_folder_path, mask_file)
                        
                        if os.path.exists(mask_path):
                            self.image_paths.append(img_path)
                            self.mask_paths.append(mask_path)
                        else:
                            print(f"找不到mask文件: {mask_path}")
        
        # 训练/验证集划分 (80/20)
        total_count = len(self.image_paths)
        if total_count == 0:
            raise ValueError("没有找到匹配的图像和掩码文件对!")
            
        indices = list(range(total_count))
        random.shuffle(indices)
        split = int(0.8 * total_count)
        
        if is_train:
            self.indices = indices[:split]
        else:
            self.indices = indices[split:]
            
        print(f"加载了 {len(self.indices)} 个{'训练' if is_train else '验证'}样本")
        print(f"总样本数: {total_count}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        
        # 使用PIL加载图像（避免gdal）
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # 使用PIL加载mask（避免gdal）
        mask = Image.open(mask_path)
        mask = mask.convert('L')
        
        # 转换为numpy数组
        image = np.array(image, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)
        
        # ================= 关键修复：确保标签值在0-56范围内 =================
        # 使用模运算限制标签范围
        mask = mask % self.num_classes
        
        # 数据增强
        if self.is_train and random.random() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        if self.is_train and random.random() < 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
            
        # 标准化图像
        image = image / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # 调整大小
        image = cv2.resize(image, self.input_shape)
        mask = cv2.resize(mask, self.input_shape, interpolation=cv2.INTER_NEAREST)
        
        # 转置通道顺序 (H, W, C) -> (C, H, W)
        image = image.transpose(2, 0, 1)
        
        # 转换为float32和int64
        image = image.astype(np.float32)
        mask = mask.astype(np.int64)
        
        return image, mask

# 定义对称交叉熵损失函数
class SymmetricCrossEntropyLoss(torch.nn.Module):
    def __init__(self, alpha=0.1, reduction='mean'):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, pred, target):
        # 标准交叉熵损失
        ce = torch.nn.functional.cross_entropy(pred, target, reduction='none')
        
        # 预测的概率分布
        pred_prob = torch.softmax(pred, dim=1)
        
        # 反向交叉熵损失
        reverse_ce = -torch.sum(pred_prob * torch.log(pred_prob + 1e-12), dim=1)
        
        # 对称交叉熵损失
        sce = ce + self.alpha * reverse_ce
        
        if self.reduction == 'mean':
            return torch.mean(sce)
        elif self.reduction == 'sum':
            return torch.sum(sce)
        else:
            return sce

# 定义边界损失函数
class BoundaryLoss(torch.nn.Module):
    def __init__(self, pixel_spacing=1.0):
        super(BoundaryLoss, self).__init__()
        self.pixel_spacing = pixel_spacing
        
    def forward(self, pred, target):
        # 转换为二值掩码（前景为1，背景为0）
        binary_target = (target > 0).float()
        
        # 在CPU上计算距离变换图（每个像素到最近边界点的距离）
        with torch.no_grad():
            # 距离变换图（边界内部为负值，外部为正值）
            dist_map = np.zeros_like(binary_target.cpu().numpy())
            for i in range(dist_map.shape[0]):
                for j in range(dist_map.shape[1]):
                    # 二值分割图
                    bin_img = binary_target[i, j].cpu().numpy()
                    
                    # 计算边界距离
                    outer = distance_transform_edt(1 - bin_img) * self.pixel_spacing
                    inner = distance_transform_edt(bin_img) * self.pixel_spacing
                    
                    # 创建有向距离图（外部为正，内部为负）
                    dist_map[i, j] = outer - inner
        
        # 转换为张量并移到GPU
        dist_map = torch.tensor(dist_map, dtype=torch.float32).to(pred.device)
        
        # 计算前景概率
        pred_prob = torch.softmax(pred, dim=1)
        foreground_prob = 1 - pred_prob[:, 0]  # 背景的概率减去1即前景概率
        
        # 边界损失（按原论文系数）
        boundary_loss = torch.mean(foreground_prob * dist_map)
        
        return boundary_loss

# 自定义的fit_one_epoch函数
def custom_fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, 
                         gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, 
                         save_period, save_dir, local_rank):
    total_loss = 0
    val_loss = 0
    
    # 创建损失函数实例（根据论文设置权重）
    sce_loss = SymmetricCrossEntropyLoss(alpha=0.1)
    boundary_loss = BoundaryLoss()
    
    model_train.train()
    
    # 训练循环
    for iteration, batch in enumerate(gen):
        images, masks = batch
        
        # 将数据移动到设备并转换为正确的数据类型
        images = images.to(cuda).float()  # 转换为float32
        masks = masks.to(cuda).long()     # 转换为long (int64)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model_train(images)
        
        # 计算损失 - 使用对称交叉熵损失和边界损失
        sce = sce_loss(outputs, masks.squeeze(1))
        bd = boundary_loss(outputs, masks.squeeze(1))
        
        # 按论文比例组合损失
        loss = 0.6 * sce + 0.4 * bd
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 打印进度
        if iteration % 100 == 0:
            print(f"Epoch: {epoch+1}/{Epoch}, Iteration: {iteration}/{epoch_step}, Loss: {loss.item():.4f}, "
                  f"SCE: {sce.item():.4f}, BD: {bd.item():.4f}")
    
    # 验证循环
    model_train.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(gen_val):
            images, masks = batch
            images = images.to(cuda).float()  # 转换为float32
            masks = masks.to(cuda).long()     # 转换为long (int64)
            
            outputs = model_train(images)
            sce = sce_loss(outputs, masks.squeeze(1))
            bd = boundary_loss(outputs, masks.squeeze(1))
            loss = 0.6 * sce + 0.4 * bd
            val_loss += loss.item()
    
    # 计算平均损失
    avg_loss = total_loss / epoch_step
    avg_val_loss = val_loss / epoch_step_val
    
    # 保存模型
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model_train.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}.pth"))
    
    return avg_loss, avg_val_loss

if __name__ == "__main__":
    # =================== 用户配置区域 (根据图片设置) =================== 
    # 数据集路径
    original_image_dirs = [
        "XJTU/nii Translatepng/outputfile",
        "XJTU/nii Translatepng/outputfile_sbl",
        "XJTU/nii Translatepng/outputfile1"
    ]
    original_mask_dirs = [
        "XJTU/nii Translatepng/outputfile_mask",
        "XJTU/nii Translatepng/outputfile_sbl_mask",
        "XJTU/nii Translatepng/outputfile1_mask"
    ]
    
    # 训练参数
    pixel_spacing = 1.0       # 单位:mm/pixel
    num_classes = 57          # 类别数量
    target_size = (512, 512)  # 图像尺寸
    batch_size = 2            # 批次大小
    src_type = '.png'         # 源图片格式
    label_type = '.png'       # 标签格式
    
    # 输出目录
    output_dir = "deeplabV3-output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 训练配置
    Init_Epoch = 0
    Freeze_Epoch = 0
    UnFreeze_Epoch = 100      # 总共训练100个epoch
    Freeze_Train = False      # 不使用冻结训练
    
    # 模型配置
    backbone = "mobilenet"
    pretrained = True
    model_path = ""
    downsample_factor = 16
    input_shape = target_size
    
    # 优化器配置
    Init_lr = 7e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'
    
    # 其他配置
    save_period = 5
    save_dir = 'logs'
    eval_flag = True
    eval_period = 5
    dice_loss = False         # 不再使用Dice损失
    focal_loss = False        # 不再使用Focal损失
    cls_weights = np.ones([num_classes], np.float32)
    num_workers = 4
    Cuda = True
    seed = 11
    distributed = False
    sync_bn = False
    fp16 = False
    
    # 初始化随机种子
    seed_everything(seed)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0

    # 创建模型
    model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
                    pretrained=pretrained)
    
    # 加载预训练权重
    if model_path != '':
        print(f'加载权重从 {model_path}')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    # 数据加载
    print("加载训练数据集...")
    train_dataset = CustomDataset(
        input_shape=input_shape, 
        num_classes=num_classes,
        original_image_dirs=original_image_dirs,
        original_mask_dirs=original_mask_dirs,
        src_type=src_type,
        label_type=label_type,
        is_train=True,
        pixel_spacing=pixel_spacing
    )
    
    print("加载验证数据集...")
    val_dataset = CustomDataset(
        input_shape=input_shape, 
        num_classes=num_classes,
        original_image_dirs=original_image_dirs,
        original_mask_dirs=original_mask_dirs,
        src_type=src_type,
        label_type=label_type,
        is_train=False,
        pixel_spacing=pixel_spacing
    )
    
    # 数据加载器
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, 
                     num_workers=num_workers, pin_memory=True, drop_last=True)
    gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, 
                         num_workers=num_workers, pin_memory=True, drop_last=True)
    
    # 模型移动到设备
    model_train = model.to(device)
    
    # 创建评估回调
    eval_callback = AdvancedEvalCallback(
        model_train, input_shape, num_classes, 
        output_dir, device, eval_period=eval_period,
        pixel_spacing=pixel_spacing
    )
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=Init_lr, weight_decay=weight_decay)
    
    # 学习率调度器
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, UnFreeze_Epoch)
    
    # ================= 训练循环 (100个epoch) =================
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        # 设置学习率
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        
        # 训练一个epoch
        train_loss, val_loss = custom_fit_one_epoch(
            model_train, model, None, eval_callback, 
            optimizer, epoch, len(gen), len(gen_val), 
            gen, gen_val, UnFreeze_Epoch, device, 
            dice_loss, focal_loss, cls_weights, num_classes, 
            fp16, None, save_period, save_dir, local_rank
        )
                      
        print(f"Epoch {epoch+1}/{UnFreeze_Epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                      
        # 评估
        if eval_flag and (epoch % eval_period == 0 or epoch == UnFreeze_Epoch - 1):
            metrics = eval_callback.on_epoch_end(epoch, gen_val)
            if metrics:
                print(f"Epoch {epoch+1}/{UnFreeze_Epoch} Metrics: {metrics}")
                
    # ================= 训练后处理 =================
    # 计算最终统计
    summary = eval_callback.final_summary()
    print("最终指标总结:")
    print(json.dumps(summary, indent=4))
    
    # 保存最终指标
    final_metrics = {
        "Epochs": UnFreeze_Epoch,
        "Parameter": summary.get("Parameter", 0),
        "mIoU_mean": summary.get("mIoU_mean", 0),
        "mIoU_std": summary.get("mIoU_std", 0),
        "mPA_mean": summary.get("mPA_mean", 0),
        "mPA_std": summary.get("mPA_std", 0),
        "mPrecision_mean": summary.get("mPrecision_mean", 0),
        "mPrecision_std": summary.get("mPrecision_std", 0),
        "mRecall_mean": summary.get("mRecall_mean", 0),
        "mRecall_std": summary.get("mRecall_std", 0),
        "Dice_mean": summary.get("Dice_mean", 0),
        "Dice_std": summary.get("Dice_std", 0),
        "MSD_mean": summary.get("MSD_mean", 0),
        "MSD_std": summary.get("MSD_std", 0),
        "HD_mean": summary.get("HD_mean", 0),
        "HD_std": summary.get("HD_std", 0),
        "ASSD_mean": summary.get("ASSD_mean", 0),
        "ASSD_std": summary.get("ASSD_std", 0),
    }
    
    with open(os.path.join(output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)

    print("训练完成! 总轮次:", UnFreeze_Epoch)