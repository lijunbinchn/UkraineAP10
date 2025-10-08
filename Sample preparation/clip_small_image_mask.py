import glob
import os
from osgeo import gdal
from tqdm import tqdm


#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset


#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
          im_bands, (im_height, im_width) = 1, im_data.shape
        # im_data = np.array([im_data])
        # im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    if im_bands == 1:
      dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
           dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def get_file_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

# 计算新的仿射变换参数函数
def calc_new_geotrans(geotrans, x_offset, y_offset):
    """
    计算裁剪图块的新仿射变换参数。
    geotrans: 原始仿射变换参数
    x_offset: x方向像素偏移量（列偏移量）
    y_offset: y方向像素偏移量（行偏移量）
    """
    new_geotrans = list(geotrans)
    new_geotrans[0] = geotrans[0] + x_offset * geotrans[1]
    new_geotrans[3] = geotrans[3] + y_offset * geotrans[5]
    return tuple(new_geotrans)

def TifCrop_with_geoinfo(TifPath, SaveDir, CropSize, RepetitionRate):
        '''
        滑动窗口裁剪函数
        TifPath: 影像路径
        SaveDir: 裁剪后保存目录
        CropSize: 裁剪尺寸
        RepetitionRate: 重复率
        '''
        dataset_img = readTif(TifPath)
        width = dataset_img.RasterXSize
        height = dataset_img.RasterYSize
        proj = dataset_img.GetProjection()
        geotrans = dataset_img.GetGeoTransform()
        img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据

        tif_name = get_file_name(TifPath)
        new_name = 0  # 文件计数器

        stride = int(CropSize * (1 - RepetitionRate))  # 步长

        # 主循环，裁剪大部分图块
        for i in range(0, height - CropSize + 1, stride):
            for j in range(0, width - CropSize + 1, stride):
                y_start = i
                x_start = j
                if len(img.shape) == 2:
                    cropped = img[y_start: y_start + CropSize,
                              x_start: x_start + CropSize]
                else:
                    cropped = img[:,
                              y_start: y_start + CropSize,
                              x_start: x_start + CropSize]
                # 计算新的仿射变换参数
                new_geotrans = calc_new_geotrans(geotrans, x_start, y_start)
                # 保存裁剪后的图像
                save_path = os.path.join(SaveDir, f"{tif_name}_{new_name}.tif")
                writeTiff(cropped, new_geotrans, proj, save_path)
                new_name += 1

        # 处理最后一列
        for i in range(0, height - CropSize + 1, stride):
            y_start = i
            x_start = width - CropSize
            if len(img.shape) == 2:
                cropped = img[y_start: y_start + CropSize,
                          x_start: width]
            else:
                cropped = img[:,
                          y_start: y_start + CropSize,
                          x_start: width]
            new_geotrans = calc_new_geotrans(geotrans, x_start, y_start)
            save_path = os.path.join(SaveDir, f"{tif_name}_{new_name}.tif")
            writeTiff(cropped, new_geotrans, proj, save_path)
            new_name += 1

        # 处理最后一行
        for j in range(0, width - CropSize + 1, stride):
            y_start = height - CropSize
            x_start = j
            if len(img.shape) == 2:
                cropped = img[y_start: height,
                          x_start: x_start + CropSize]
            else:
                cropped = img[:,
                          y_start: height,
                          x_start: x_start + CropSize]
            new_geotrans = calc_new_geotrans(geotrans, x_start, y_start)
            save_path = os.path.join(SaveDir, f"{tif_name}_{new_name}.tif")
            writeTiff(cropped, new_geotrans, proj, save_path)
            new_name += 1

        # 处理右下角的图块
        y_start = height - CropSize
        x_start = width - CropSize
        if len(img.shape) == 2:
            cropped = img[y_start: height,
                      x_start: width]
        else:
            cropped = img[:,
                      y_start: height,
                      x_start: width]
        new_geotrans = calc_new_geotrans(geotrans, x_start, y_start)
        save_path = os.path.join(SaveDir, f"{tif_name}_{new_name}.tif")
        writeTiff(cropped, new_geotrans, proj, save_path)
        new_name += 1

if __name__ == '__main__':

    # # 输入待裁剪影像的文件夹，对文件夹内的所有影像进行裁剪操作
    image_dir = r"E:\Codes\MyProjectCodes\Agricultural_Parcel\data\UKR\UKR_2023\data4trian\train\big_image"
    image_save_dir = r"E:\Codes\MyProjectCodes\Agricultural_Parcel\data\UKR\UKR_2023\data4trian\train\train_image"

    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
        print(f"创建文件夹：{image_save_dir}")

    tif_paths = glob.glob(os.path.join(image_dir, '*.tif'))
    print(f"待裁剪影像：{len(tif_paths)} 个")

    # 裁剪遥感影像数据
    for tif_path in tqdm(tif_paths):
        TifCrop_with_geoinfo(tif_path, image_save_dir, CropSize=256, RepetitionRate=0.5)


    # 输入待裁剪的掩膜栅格数据
    mask_dir = r"E:\Codes\MyProjectCodes\Agricultural_Parcel\data\UKR\UKR_2023\data4trian\train\big_mask"
    mask_save_dir = r"E:\Codes\MyProjectCodes\Agricultural_Parcel\data\UKR\UKR_2023\data4trian\train\train_mask"

    if not os.path.exists(mask_save_dir):
        os.makedirs(mask_save_dir)
        print(f"创建文件夹：{mask_save_dir}")

    mask_paths = glob.glob(os.path.join(mask_dir, '*.tif'))
    print(f"待裁剪影像：{len(mask_paths)} 个")

    # 裁剪掩膜栅格数据
    for mask_path in tqdm(mask_paths):
        TifCrop_with_geoinfo(TifPath=mask_path, SaveDir=mask_save_dir, CropSize=256, RepetitionRate=0.5)

