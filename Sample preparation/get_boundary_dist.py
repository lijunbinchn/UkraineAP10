## Example: A simple example to obtain boundary map
import glob

import numpy as np
import os
import cv2
from osgeo import gdal
from tqdm import tqdm
import scipy.ndimage as sn

def read_img(filename):
    dataset=gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width,im_height)

    del dataset
    return im_proj, im_geotrans, im_width, im_height, im_data

def write_img(filename, im_proj, im_geotrans, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset

def generate_boundary(mask_file_path, upsample_flag):

    im_proj, im_geotrans, im_width, im_height, im_data = read_img(mask_file_path)
    im_data = im_data.astype(np.uint8)
    # 判断掩膜是否为1
    # 获取唯一值
    unique_values = np.unique(im_data)
    # 判断掩膜是否包含 1 而不是 255
    if 1 in unique_values and 255 not in unique_values:
        # 如果掩膜是以 1 表示的，将其转换为 255
        im_data = (im_data * 255).astype(np.uint8)

    # 如果影像的分辨率过粗，地块粘连现象较严重，需要上采样，提高影像分辨率，然后进行边缘检测得到边界，然后在进行下采样。
    # 需要尝试看看效果需不需要上采样。
    if upsample_flag:
        # 定义缩放比例，例如将图像尺寸扩大 2 倍
        scale_factor = 2
        # 获取原始图像的尺寸
        height, width = im_data.shape[:2]
        # 计算新的尺寸
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # 进行上采样（调整图像尺寸）
        im_data_upsampled = cv2.resize(
            im_data,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC  # 使用双三次插值，适合于上采样
        )

        # 检测边缘
        boundary = cv2.Canny(im_data_upsampled, 50, 200)

        # 将图像缩小回原始尺寸
        boundary_resized = cv2.resize(boundary, (width, height), interpolation=cv2.INTER_AREA)

        # 将图像进行二值化，非0像元像素值都转为255
        _, boundary = cv2.threshold(boundary_resized, 0, 255, cv2.THRESH_BINARY)

    else:
        # 检测边缘
        boundary = cv2.Canny(im_data, 50, 200)
        # 可以尝试 dilation, 看看修改
        # kernel = np.ones((3, 3), np.uint8)
        # boundary = cv2.dilate(boundary, kernel, iterations=1)

    return boundary

def get_distance(mask_file_path):
    im_proj, im_geotrans, im_width, im_height, im_data = read_img(mask_file_path)
    im_data = im_data.astype(np.uint8)
    distance_result = cv2.distanceTransform(src=im_data, distanceType=cv2.DIST_L2, maskSize=3)
    min_value = np.min(distance_result)
    max_value = np.max(distance_result)
    if max_value != min_value:
        distance = ((distance_result - min_value) / (max_value - min_value)) * 255
    else:
        # 如果max_value == min_value，可以将结果设为零或进行其他处理
        distance = np.zeros_like(distance_result)

    # distance = ((distance_result - min_value) / (max_value - min_value)) * 255

    distance = distance.astype(np.uint8)
    return distance


# 要求输入的mask中的地块像元位255。
if __name__ == '__main__':

    mask_dir = r"E:\Codes\MyProjectCodes\Agricultural_Parcel\data\UKR\UKR_2023\data4trian\train\train_mask"
    # 将mask_dir 的 mask换为 boundary, distance
    boundary_dir, distance_dir = mask_dir.replace("mask", "boundary"), mask_dir.replace("mask", "distance")

    if not os.path.exists(boundary_dir):
        os.makedirs(boundary_dir, exist_ok=True)
    if not os.path.exists(distance_dir):
        os.makedirs(distance_dir, exist_ok=True)

    # 获取文件夹内所有掩膜文件的路径
    mask_paths = glob.glob(os.path.join(mask_dir, '*.tif'))
    i = 0
    for mask_path in tqdm(mask_paths):
        im_proj, im_geotrans, im_width, im_height, im_data = read_img(mask_path)
        mask_name = os.path.basename(mask_path)
        # 将mask_name中的mask换为 boundary
        boundary_name = mask_name.replace("mask", "boundary")
        # 将mask_name中的mask换为 distance
        distance_name = mask_name.replace("mask", "distance")

        # 生成边界栅格
        boundary_path = os.path.join(boundary_dir, boundary_name)
        # 可以先实验看看是否需要上采样
        boundary_result = generate_boundary(mask_path, upsample_flag=False)
        write_img(boundary_path, im_proj, im_geotrans, boundary_result)

        # 生成距离栅格
        distance_path = os.path.join(distance_dir, distance_name)
        distance_result = get_distance(mask_path)
        write_img(distance_path, im_proj, im_geotrans, distance_result)

        i += 1

        # if i == 100:
        #     break

    print("boundary and distance done")







    #     input_path = os.path.join(maskRoot, imgPath)
    #     boundaryOutPath = os.path.join(boundaryRoot, imgPath)
    #     im_proj, im_geotrans, im_width, im_height, im_data = read_img(input_path)
    #     # If im_data values are between 0 and 1
    #
    #     # 因为原始的掩膜数据是1，所以乘以255
    #     im_data = (im_data * 255).astype(np.uint8)
    #
    #     # 定义缩放比例，例如将图像尺寸扩大 2 倍
    #     scale_factor = 2
    #     # 获取原始图像的尺寸
    #     height, width = im_data.shape[:2]
    #     # 计算新的尺寸
    #     new_width = int(width * scale_factor)
    #     new_height = int(height * scale_factor)
    #
    #     # 进行上采样（调整图像尺寸）
    #     im_data_upsampled = cv2.resize(
    #         im_data,
    #         (new_width, new_height),
    #         interpolation=cv2.INTER_CUBIC  # 使用双三次插值，适合于上采样
    #     )
    #
    #     # 检测边缘
    #     boundary = cv2.Canny(im_data_upsampled, 50, 200)
    #
    #     # 将图像缩小回原始尺寸
    #     resized_boundary = cv2.resize(boundary, (width, height), interpolation=cv2.INTER_AREA)
    #
    #     # 将图像进行二值化，非0像元像素值都转为255
    #     ret, resized_boundary = cv2.threshold(resized_boundary, 0, 255, cv2.THRESH_BINARY)
    #
    #     write_img(boundaryOutPath, im_proj, im_geotrans, resized_boundary)
    #     # break
    #
    #     # input_path = os.path.join(maskRoot, imgPath)
    #     # boundaryOutPath = os.path.join(boundaryRoot, imgPath)
    #     # im_proj, im_geotrans, im_width, im_height, im_data = read_img(input_path)
    #     # im_data = (im_data * 255).astype(np.uint8)
    #     # boundary = cv2.Canny(im_data, 100, 200)
    #     # ## dilation
    #     # kernel = np.ones((3, 3), np.uint8)
    #     # boundary = cv2.dilate(boundary, kernel, iterations=1)
    #     # write_img(boundaryOutPath, im_proj, im_geotrans, boundary)
    #     # break

    # mask_file_path = r"E:\Codes\MyProjectCodes\Agricultural_Parcel\data\chang_guang_test\original_data\train\mask\CGDZ_1_offset_mask_24_vec.tif"
    # # mask_file_path = r"E:\Codes\MyProjectCodes\Agricultural_Parcel\hetao_parcel\data\train\label\hetao_009471_025343_label.tif"
    # im_proj, im_geotrans, im_width, im_height, im_data = read_img(mask_file_path)
    # # 假设 im_data 是掩膜数据
    # # 获取唯一值
    # unique_values = np.unique(im_data)
    # print("Unique values in mask:", unique_values)
    # # 判断掩膜是否包含 1 而不是 255
    # if 1 in unique_values and 255 not in unique_values:
    #     print("Mask is 1, converting to 255")
    #     # 如果掩膜是以 1 表示的，将其转换为 255
    #     im_data = (im_data * 255).astype(np.uint8)
    #
    # print("Modified mask data:", im_data)

