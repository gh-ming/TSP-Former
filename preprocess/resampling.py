import numpy as np
from osgeo import gdal, osr
import os
from datetime import datetime
from tqdm import tqdm

def check_raster_alignment(source_ds, target_ds):
    """
    检查两个数据集是否具有相同的投影、范围和分辨率
    :param source_ds: 源数据集（GDAL Dataset）
    :param target_ds: 目标数据集（GDAL Dataset）
    :return: 如果一致返回True，否则返回False
    """
    # 检查投影
    source_proj = source_ds.GetProjection()
    target_proj = target_ds.GetProjection()
    if source_proj != target_proj:
        return False

    # 检查地理变换参数（分辨率、旋转和左上角坐标）
    source_geotransform = source_ds.GetGeoTransform()
    target_geotransform = target_ds.GetGeoTransform()
    if source_geotransform != target_geotransform:
        return False

    # 检查范围
    source_cols = source_ds.RasterXSize
    source_rows = source_ds.RasterYSize
    target_cols = target_ds.RasterXSize
    target_rows = target_ds.RasterYSize
    if source_cols != target_cols or source_rows != target_rows:
        return False

    return True

def resample_to_target(source_ds, target_proj, target_geotransform, target_cols, target_rows, resample_alg):
    """
    将源数据集重采样到目标空间参考和范围
    :param source_ds: 源数据集（GDAL Dataset）
    :param target_proj: 目标投影
    :param target_geotransform: 目标地理变换参数
    :param target_cols: 目标列数
    :param target_rows: 目标行数
    :param resample_alg: 重采样算法（如gdal.GRA_NearestNeighbour）
    :return: 重采样后的GDAL Dataset（内存中）
    """
    # 计算目标范围
    x_min = target_geotransform[0]
    y_max = target_geotransform[3]
    x_max = x_min + target_cols * target_geotransform[1]
    y_min = y_max + target_rows * target_geotransform[5]  # geotransform[5]为负值

    # 重采样选项
    resampled_ds = gdal.Warp('', source_ds, format='MEM',
                             dstSRS=target_proj,
                             xRes=target_geotransform[1],
                             yRes=abs(target_geotransform[5]),
                             outputBounds=(x_min, y_min, x_max, y_max),
                             resampleAlg=resample_alg)
    
    scl_band = resampled_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create('/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/TSP/TSP_weining_resampling.tif', target_cols, target_rows, 1, gdal.GDT_Float32)
    out_dataset.SetGeoTransform(target_geotransform)
    out_dataset.SetProjection(target_proj)
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(scl_band)
    out_band.FlushCache()
    return resampled_ds


if __name__ == '__main__':
    reference_tif = "/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/month_mean/mosaic/weining_mosaic_06.tif"
    tsp_path = os.path.join('/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/TSP', f"TSP_weining.tif")
    reference_dataset = gdal.Open(reference_tif)
    tsp_dataset = gdal.Open(tsp_path)
    target_proj = reference_dataset.GetProjection()
    target_geotransform = reference_dataset.GetGeoTransform()
    target_cols = reference_dataset.RasterXSize
    target_rows = reference_dataset.RasterYSize
    if not check_raster_alignment(tsp_dataset, reference_dataset):
        print("SCL影像与生长期影像未对齐，正在重采样...")
        scl_dataset = resample_to_target(tsp_dataset, target_proj, target_geotransform,
                                         target_cols, target_rows, gdal.GRA_NearestNeighbour)