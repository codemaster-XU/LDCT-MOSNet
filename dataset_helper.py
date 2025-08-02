from PIL import Image
import numpy as np
import os
import cv2

"""
    预处理数据集的工具函数
    author: wangchongyang
    date: 2024/12/3
"""

"""
    给数据集文件排序
"""
def file_seq(folder_path):
    import os
    # 获取文件夹内所有文件
    files = os.listdir(folder_path)

    # 过滤出图片文件，这里假设图片文件的后缀为.jpg, .png等
    # 你可以根据需要添加或修改后缀
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif','.tif','.tiff']
    images = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]

    # 根据文件名进行排序
    images.sort()

    # 重命名文件，从0开始，只保留序号和后缀
    for index, image in enumerate(images):
        # 获取文件后缀
        _, ext = os.path.splitext(image)
        # 构建新的文件名（从0开始，只包含序号和后缀）
        new_file_name = f"{index}{ext}"
        # 构建完整的旧文件路径和新文件路径
        old_file_path = os.path.join(folder_path, image)
        new_file_path = os.path.join(folder_path, new_file_name)
        print('a')
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {old_file_path} to {new_file_path}")

"""
    清空arcgis pro 分割结果中的多余文件，只保留.tif文件
"""
def clear_tif_folder(folder_path):
    for filename in os.listdir(folder_path):
        if not filename.endswith('.tif'):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)
            # 删除文件
            os.remove(file_path)
            print(f"Deleted {file_path}")

"""
    单个mask可视化，用于png jpg
    这个方法会在训练中引入
"""
def make_mask_visible(path):
    img_file_path= path
    img_file_name = extract_filename(path)
    mask_file_name = img_file_name + '.tif'
    mask_file_name = os.path.join("VOCdevkit/tif_dataset/label",mask_file_name)
    label = mask_file_name
    label = cv2.imread(label)
    label = cv2.resize(label,(500,500))
    # 将标签图像转换为float类型
    label_float = label.astype(np.float32)
    # 归一化到0-1
    label_normalized = label_float / 3.0
    # 缩放到0-255
    label_scaled = cv2.normalize(label_normalized, None, 0, 255, cv2.NORM_MINMAX)
    # 转换为uint8类型
    label_255 = label_scaled.astype(np.uint8)
    img = cv2.imread(img_file_path)
    img = cv2.resize(img,(500,500))
    cv2.imshow('org_img',img)
    cv2.imshow('org_mask',label_255)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

"""
    单个mask可视化，用于tif
"""
def make_mask_visible_tif(path):
    img_file_path= path
    img_file_name = extract_filename(path)
    mask_file_name = img_file_name + '.tif'
    mask_file_name = os.path.join("VOCdevkit/tif_dataset/label",mask_file_name)
    label = mask_file_name
    label = cv2.imread(label,-1)
    label = cv2.resize(label,(500,500))
    label = label.astype('uint8')  # 转换为 uint8 类型
    print(label[label > 0])
    label[label > 0] = 255
    print(label[label > 0])
    img = cv2.imread(img_file_path)
    img = cv2.resize(img,(500,500))
    cv2.imshow('org_img',img)
    cv2.imshow('org_mask',label)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

"""
    保存训练中的超参数
    这个方法会在训练中引入
"""
def save_hyperparameters(hyperparameters, save_dir):
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 定义保存文件的路径
    file_path = os.path.join(save_dir, "hyperparameters.txt")

    # 打开文件并写入超参数
    with open(file_path, 'w') as file:
        for key, value in hyperparameters.items():
            file.write(f"{key}: {value}\n")
def extract_filename(path):
    # 使用os.path.basename获取路径中的文件名（包括扩展名）
    filename_with_extension = os.path.basename(path)
    # 使用os.path.splitext去掉文件扩展名
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    return filename_without_extension
def make_mask_visible_tif2(path):
    label = cv2.imread(path,-1)
    label = label.astype('uint8')  # 转换为 uint8 类型
    print(label)
    label[(label > 0) & (label <= 1)] = 80
    label[(label > 0) & (label < 80)] = 180
    label = cv2.resize(label,(1000,1000))
    cv2.imshow('org_mask',label)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
"""
    将整个标签文件夹中的Label可视化
"""
def make_mask_dir_visible_tif2(input_folder, output_folder):
    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 构建完整的文件路径
        file_path = os.path.join(input_folder, filename)

        # 检查是否是文件
        if os.path.isfile(file_path):
            # 读取文件
            label = cv2.imread(file_path, -1)
            label = label.astype('uint8')  # 转换为 uint8 类型

            # 修改标签值
            label[(label > 0) & (label <= 1)] = 80
            label[(label > 0) & (label < 80)] = 180

            # 调整图像大小
            label = cv2.resize(label, (512, 512))

            # 构建输出文件路径
            output_path = os.path.join(output_folder, filename)

            # 保存修改后的图像
            cv2.imwrite(output_path, label)
"""
    计算最多可能含有多少个类别
"""
def count_max_classes_including_background(label_folder_path):
    max_classes = 0  # 初始化最大类别数为0

    # 遍历文件夹中的所有文件
    for filename in os.listdir(label_folder_path):
        # 检查文件是否为图片
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            label_image_path = os.path.join(label_folder_path, filename)

            # 读取Label图片
            label_image = Image.open(label_image_path)
            label_array = np.array(label_image)

            # 找到唯一值，并且确保包括0（背景）
            unique_values = np.unique(label_array)
            if 0 not in unique_values:
                unique_values = np.append(unique_values, 0)
            unique_values = np.sort(unique_values)  # 确保类是排序的

            # 计算类别数量
            num_classes = len(unique_values)

            # 更新最大类别数
            if num_classes > max_classes:
                max_classes = num_classes

    return max_classes
def make_mask_dir_visible(input_directory, output_directory):
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 遍历指定目录中的所有文件
    for filename in os.listdir(input_directory):
        if filename.endswith('.jpg'):
            img_file_path = os.path.join(input_directory, filename)
            img_file_name = extract_filename(img_file_path)
            mask_file_name = img_file_name + '.jpg'
            mask_file_path = os.path.join(input_directory, mask_file_name)
            # 读取标签图像
            label = cv2.imread(mask_file_path)
            if label is None:
                print(f"无法读取标签图像: {mask_file_path}")
                continue
            # 调整标签图像大小
            # label = cv2.resize(label, (500, 500))
            # 将标签图像转换为float类型
            label_float = label.astype(np.float32)
            # 归一化到0-1
            label_normalized = label_float / 255.0
            # 缩放到0-255
            label_scaled = cv2.normalize(label_normalized, None, 0, 255, cv2.NORM_MINMAX)
            # 转换为uint8类型
            label_255 = label_scaled.astype(np.uint8)
            # 读取原始图像
            img = cv2.imread(img_file_path)
            if img is None:
                print(f"无法读取图像: {img_file_path}")
                continue
            # 构建输出文件路径
            output_mask_file_path = os.path.join(output_directory, mask_file_name)
            # 保存处理后的标签图像
            cv2.imwrite(output_mask_file_path, label_255)

def make_flood_label_visible(input_folder, output_folder):
    """
    将洪水数据集的标签可视化，按照指定的颜色方案
    Args:
        input_folder: 输入标签文件夹路径
        output_folder: 输出可视化结果文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 定义颜色映射
    colors = [
        (0, 0, 0),        # 背景
        (0, 0, 139),      # 建筑物被洪水淹没
        (0, 192, 203),    # 建筑物未被洪水淹没
        (0, 100, 0),      # 道路被洪水淹没
        (255, 165, 0),    # 道路未被洪水淹没
        (0, 255, 255),    # 水
        (0, 128, 0),      # 树
        (0, 0, 255),      # 车辆
        (128, 128, 128),  # 游泳池
        (128, 128, 0)     # 草地
    ]
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            # 读取标签图像
            label_path = os.path.join(input_folder, filename)
            label = cv2.imread(label_path, -1)  # -1表示按原样读取
            
            if label is None:
                print(f"无法读取文件: {label_path}")
                continue
                
            # 创建彩色图像
            height, width = label.shape if len(label.shape) == 2 else label.shape[:2]
            colored_label = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 为每个类别赋予对应的颜色
            for class_id, color in enumerate(colors):
                mask = label == class_id
                colored_label[mask] = color
            
            # 构建输出文件路径
            output_path = os.path.join(output_folder, f"visual_{filename}")
            
            # 保存可视化结果
            cv2.imwrite(output_path, colored_label)
            print(f"已处理: {filename}")

# 使用示例
# input_folder = "test"
# output_folder = "test_out"
# make_flood_label_visible(input_folder, output_folder)

def check_dataset_files(dataset_path, image_ids):
    """
    检查数据集文件是否存在及其扩展名
    Args:
        dataset_path: 数据集根目录
        image_ids: 图像ID列表
    """
    image_path = os.path.join(dataset_path, "VOC2007/JPEGImages")
    label_path = os.path.join(dataset_path, "VOC2007/SegmentationClass")
    
    print("检查数据集文件...")
    for image_id in image_ids:
        # 检查图像文件
        image_files = [f for f in os.listdir(image_path) if f.startswith(image_id + ".")]
        if not image_files:
            print(f"警告: 未找到图像文件 {image_id}")
        else:
            print(f"找到图像文件: {image_files[0]}")
            
        # 检查标签文件
        label_files = [f for f in os.listdir(label_path) if f.startswith(image_id + ".")]
        if not label_files:
            print(f"警告: 未找到标签文件 {image_id}")
        else:
            print(f"找到标签文件: {label_files[0]}")
