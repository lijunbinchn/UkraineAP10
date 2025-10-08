import os
from skimage.morphology import opening, square
from osgeo import gdal
import time
from tqdm import tqdm

def remove_small_patches_with_opening(data, square_size):
    # 使用开运算去除小斑块。可以调整结构元素大小来影响去除的小斑块的大小
    # square(3) 表示 3x3 的方形结构元素
    cleaned_data = opening(data.astype(bool), footprint=square(square_size))  # 结构元素为3x3方形
    return cleaned_data.astype(int)


# 设置GDAL环境变量以使用所有CPU
gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')

input_mask_path = r"/home/lijb/data/remove_small_parcels/parcel_result/ukr_parcel_result_HBGNet_2023.tif"  # 输入影像路径
output_folder = r"/home/lijb/data/remove_small_parcels/result"  # 输出影像路径
# 获取输入影像数据文件名 basename
input_basename = os.path.basename(input_mask_path)
# 去掉文件扩展名
input_filename = os.path.splitext(input_basename)[0]

square_size = 7
result_mask_path = os.path.join(output_folder, f"{input_filename}_opening_{square_size}x{square_size}.tif")

# 打开数据集
mask_dataset = gdal.Open(input_mask_path, gdal.GA_ReadOnly)
if not mask_dataset:
    print("无法打开数据集")

# 保存处理后的栅格数据
# 创建输出栅格数据集
driver = gdal.GetDriverByName('GTiff')
output_ds = driver.Create(result_mask_path,  # 输出栅格数据集路径
                          mask_dataset.RasterXSize,
                          mask_dataset.RasterYSize,  # 输出栅格数据集的高度
                          1,  # 波段数
                          mask_dataset.GetRasterBand(1).DataType,  # 数据类型
                          options=['COMPRESS=LZW', 'PREDICTOR=2',  # 创建选项
                                   'BIGTIFF=YES', "TILED=YES",
                                   'NUM_THREADS=ALL_CPUS']
                          )
# 设置地理变换参数和投影信息
output_ds.SetGeoTransform(mask_dataset.GetGeoTransform())
output_ds.SetProjection(mask_dataset.GetProjection())

# 设置无数据值
output_ds.GetRasterBand(1).SetNoDataValue(0)  # 设置0为无数据值

# 设置合适的块大小
block_xsize = 10240  # 可以根据实际情况调整
block_ysize = 10240  # 可以根据实际情况调整

# 获取栅格的大小
xsize = mask_dataset.RasterXSize
ysize = mask_dataset.RasterYSize

# 计算总的块数以设置进度条的总步数
total_blocks = ((xsize + block_xsize - 1) // block_xsize) * ((ysize + block_ysize - 1) // block_ysize)


# 逐块读取、处理和写入数据
with tqdm(total=total_blocks, desc="Processing Blocks") as pbar:
    for y in range(0, ysize, block_ysize):
        if y + block_ysize > ysize:
            read_ysize = ysize - y
        else:
            read_ysize = block_ysize

        for x in range(0, xsize, block_xsize):
            if x + block_xsize > xsize:
                read_xsize = xsize - x
            else:
                read_xsize = block_xsize

            # 读取当前块的数据
            mask_data = mask_dataset.GetRasterBand(1).ReadAsArray(x, y, read_xsize, read_ysize)

            cleaned_data = remove_small_patches_with_opening(mask_data, square_size=square_size)

            # 写入处理后的数据块到输出文件
            output_ds.GetRasterBand(1).WriteArray(cleaned_data, x, y)

            pbar.update(1)

            # 释放内存
            del mask_data

# 刷新缓存
output_ds.FlushCache()

# 关闭输入和输出数据集
raster_to_be_masked = None
mask_raster = None
output_ds = None

end = time.time()
print("所有文件处理完成！")

