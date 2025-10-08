#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import time

import geopandas as gpd
from osgeo import gdal
from osgeo import ogr
from tqdm import tqdm


def buffer_shp(input_file, output_file, buffer_distance=0):
    """
    对shp文件进行缓冲区分析
    :param input_file:输入文件路径
    :param output_file:输出文件路径
    :param buffer_distance:buffer范围
    :return:
    """
    # 读取shp文件
    gdf = gpd.read_file(input_file)

    csr = gdf.crs

    gdf = gdf.to_crs(epsg=3857)

    # 定义缓冲距离,这里设为500米
    # buffer_distance = 500

    # 建立缓冲区GeoSeries
    buffers = gdf.geometry.buffer(buffer_distance)

    # 将缓冲区GeoSeries转为GeoDataFrame,并与原始数据合并
    gdf_buffered = gpd.GeoDataFrame(geometry=buffers)

    # Join原始属性字段
    gdf_buffered = gdf_buffered.join(gdf.drop(columns='geometry'))

    gdf_buffered = gdf_buffered.to_crs(csr)

    # 输出结果
    # gdf_buffered.to_file(output_file, driver="GPKG")
    gdf_buffered.to_file(output_file, driver="ESRI Shapefile")

    # print('对文件' + os.path.basename(input_file) + '缓冲区分析成功！')

    return True

def buffer_shp_2(input_file, output_file, buffer_distance=0):

    gdf = gpd.read_file(input_file)
    csr = gdf.crs
    gdf = gdf.to_crs(epsg=3857)

    # 执行缓冲区操作并修复几何
    buffers = gdf.geometry.buffer(buffer_distance).make_valid()
    # 处理退化几何：将线/点转面或恢复原始几何
    repaired_geoms = []
    for idx, geom in enumerate(buffers):
        # 检查几何是否为 None
        if geom is None:
            # 处理策略：恢复原始几何，或标记为无效
            repaired = gdf.geometry.iloc[idx]
            repaired_geoms.append(repaired)
            print(f"警告：几何 {idx} 为 None，已跳过。")
            continue  # 跳过后续检查

        # 处理空几何
        if geom.is_empty:
            repaired = gdf.geometry.iloc[idx]  # 恢复原始几何
        # 处理线型几何
        elif geom.geom_type in ['LineString', 'MultiLineString']:
            buffered_line = geom.buffer(0.01)  # 给线添加微小缓冲区生成面
            if not buffered_line.is_empty:
                repaired = buffered_line
            else:
                repaired = gdf.geometry.iloc[idx]
        # 保留有效面
        elif geom.geom_type in ['Polygon', 'MultiPolygon']:
            repaired = geom
        else:
            repaired = gdf.geometry.iloc[idx]
        repaired_geoms.append(repaired)

    # 创建新的GeoDataFrame
    gdf_buffered = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(repaired_geoms, crs=3857),
        data=gdf.drop(columns='geometry')
    ).to_crs(csr)

    # 强制过滤非面几何（最终保障）
    gdf_buffered = gdf_buffered[
        gdf_buffered.geometry.type.isin(['Polygon', 'MultiPolygon'])
    ]

    # 保存结果
    if not gdf_buffered.empty:
        gdf_buffered.to_file(output_file, driver="ESRI Shapefile")
        return True
    else:
        print("警告：所有几何在缓冲后均无效！")
        return False

def get_tif_meta(tif_path):
    """
    获取tif文件的基本信息
    :param tif_path:
    :return:
    """
    dataset = gdal.Open(tif_path)
    # 栅格矩阵的列数
    width = dataset.RasterXSize
    # 栅格矩阵的行数
    height = dataset.RasterYSize
    # 获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    # 获取投影信息
    proj = dataset.GetProjection()
    return width, height, geotrans, proj

def shp2tif(shp_path, refer_tif_path, target_tif_path, attribute_field="class", nodata_value=0):
    """
    将shp文件矢量化，并根据参考文件赋值影像基本参数
    :param shp_path:
    :param refer_tif_path:
    :param target_tif_path:
    :param attribute_field:
    :param nodata_value:
    :return:
    """
    width, height, geotrans, proj = get_tif_meta(refer_tif_path)
    # 读取shp文件
    shp_file = ogr.Open(shp_path)
    # 获取图层文件对象
    shp_layer = shp_file.GetLayer()
    # 创建栅格
    target_ds = gdal.GetDriverByName('GTiff').Create(
        utf8_path=target_tif_path,  # 栅格地址
        xsize=width,  # 栅格宽
        ysize=height,  # 栅格高
        bands=1,  # 栅格波段数
        eType=gdal.GDT_Byte,  # 栅格数据类型
        options=['COMPRESS=LZW', 'PREDICTOR=2',
                 'BIGTIFF=YES', "TILED=YES",
                 'NUM_THREADS=ALL_CPUS']
    )
    # 将参考栅格的仿射变换信息设置为结果栅格仿射变换信息
    target_ds.SetGeoTransform(geotrans)
    # 设置投影坐标信息
    target_ds.SetProjection(proj)
    band = target_ds.GetRasterBand(1)
    # 设置背景nodata数值
    band.SetNoDataValue(nodata_value)
    band.FlushCache()

    # 栅格化函数
    gdal.RasterizeLayer(
        dataset=target_ds,  # 输出的栅格数据集
        bands=[1],  # 输出波段
        layer=shp_layer,  # 输入待转换的矢量图层
        # options=[f"ATTRIBUTE={attribute_field}"]  # 指定字段值为栅格值
        burn_values=[255]
        # burn_values=[1]
    )

    del target_ds


if __name__ == '__main__':

    temp_dir_path = r'E:\Data\MyProjectData\Agricultural_Parcel\Ukraine\data\ukr_label\temp'
    buffer_distence = -20
    start = time.time()

    roi_list = ['roi_1', 'roi_2', 'roi_3', 'roi_4', 'roi_5']
    data_root_path = r"E:\Data\MyProjectData\Agricultural_Parcel\Ukraine\data\ukr_label"

    for roi in roi_list:
        refer_image_path = os.path.join(data_root_path, f"ukr_{roi}", 's2_image', f"s2_image_{roi}.tif")
        label_shp_path = os.path.join(data_root_path, f"ukr_{roi}", "label_shp", f"label_{roi}.shp")
        mask_path = os.path.join(data_root_path, f"ukr_{roi}", "label_shp", f'{roi}_mask_{-1 * buffer_distence}.tif')

        label_name = os.path.splitext(os.path.basename(label_shp_path))[0]
        print(f'开始处理{label_name}')

        # 进行缓冲区分析
        print(f'开始对{os.path.basename(label_name)}进行 {buffer_distence} 缓冲区分析')
        shp_buffer_path = os.path.join(temp_dir_path, f'{label_name}_buffer.shp')
        buffer_shp_2(label_shp_path, shp_buffer_path, buffer_distence)

        print(f'开始对{os.path.basename(shp_buffer_path)}进行矢量标注转换为tif')
        shp2tif(shp_buffer_path, refer_image_path, mask_path)

        end = time.time()
        print(f'{os.path.basename(mask_path)}处理完成, 耗时{end - start:.2f}s')

