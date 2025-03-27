import cv2
import numpy as np
import os

# 数据集文件夹路径
#dataset_folder = "C:/Users/Luy/Desktop/NUAA_hazy"  # 替换成您的数据集文件夹路径
#dataset_folder = "C:/Users/Luy/Desktop/1k_hazy"  # 替换成您的数据集文件夹路径
#dataset_folder ='C:/Users/Luy/Desktop/dataset1/NUDT-SIRST/images'
dataset_folder = "C:/Users/Luy/Desktop/dataset-xisu/0.1/SIRST-UAV_0.3/images_train"

# 初始化通道的像素值总和
total_intensity_R = 0
total_intensity_G = 0
total_intensity_B = 0
total_pixels = 0

# 遍历数据集文件夹中的每个图像S
for filename in os.listdir(dataset_folder):
    if filename.endswith(".png"):  # 假设您的数据集中的图像格式为JPEG
        image_path = os.path.join(dataset_folder, filename)
        image = cv2.imread(image_path)
        if image is not None:
            # 将像素值添加到各通道的总和中
            total_intensity_R += np.sum(image[:, :, 0])  # 红色通道
            total_intensity_G += np.sum(image[:, :, 1])  # 绿色通道
            total_intensity_B += np.sum(image[:, :, 2])  # 蓝色通道
            total_pixels += image.shape[0] * image.shape[1]

# 计算平均像素强度
average_intensity_R = total_intensity_R / total_pixels
average_intensity_G = total_intensity_G / total_pixels
average_intensity_B = total_intensity_B / total_pixels
print(image.shape)
print(total_intensity_R,total_intensity_G,total_intensity_B)
print(total_pixels)
print(f"平均像素强度 (R): {average_intensity_R}")
print(f"平均像素强度 (G): {average_intensity_G}")
print(f"平均像素强度 (B): {average_intensity_B}")
