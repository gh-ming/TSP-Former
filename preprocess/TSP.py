import numpy as np
from osgeo import gdal, osr
import os
from datetime import datetime
from tqdm import tqdm

class ImageProcess:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dataset = gdal.Open(self.filepath, gdal.GA_ReadOnly)
        self.info = []
        self.img_data = None
        self.data_8bit = None

    def read_img_info(self):
        # 获取波段、宽、高
        img_bands = self.dataset.RasterCount
        img_width = self.dataset.RasterXSize
        img_height = self.dataset.RasterYSize
        # 获取仿射矩阵、投影
        img_geotrans = self.dataset.GetGeoTransform()
        img_proj = self.dataset.GetProjection()
        # 获取NoData值
        img_nodata = self.dataset.GetRasterBand(1).GetNoDataValue()
        self.info = [img_bands, img_width, img_height, img_geotrans, img_proj,img_nodata]
        return self.info

    def read_img_data(self):
        self.img_data = self.dataset.ReadAsArray(0, 0, self.info[1], self.info[2])
        return self.img_data

    # 影像写入文件
    @staticmethod
    def write_img(filename: str, img_data: np.array, **kwargs):
        # 判断栅格数据的数据类型
        if 'int8' in img_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        # 判读数组维数
        if len(img_data.shape) >= 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands, (img_height, img_width) = 1, img_data.shape
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        outdataset = driver.Create(filename, img_width, img_height, img_bands, datatype)
        # 写入仿射变换参数
        if 'img_geotrans' in kwargs:
            outdataset.SetGeoTransform(kwargs['img_geotrans'])
        # 写入投影
        if 'img_proj' in kwargs:
            outdataset.SetProjection(kwargs['img_proj'])
        # 写入文件
        if img_bands == 1:
            outdataset.GetRasterBand(1).WriteArray(img_data)  # 写入数组数据
        else:
            for i in range(img_bands):
                outdataset.GetRasterBand(i + 1).WriteArray(img_data[i])

        del outdataset


def read_multi_bands(image_path):
    """
    读取多波段文件
    :param image_path: 多波段文件路径
    :return: 影像对象，影像元信息，影像矩阵
    """
    # 影像读取
    image = ImageProcess(filepath=image_path)
    # 读取影像元信息
    image_info = image.read_img_info()
    # print(f"多波段影像元信息：{image_info}")
    # 读取影像矩阵
    image_data = image.read_img_data()
    print(f"多波段矩阵大小：{image_data.shape}")
    return image, image_info, image_data


def extract_doy_from_filename(filename, start_date='2023-01-01'):
    """
    从哨兵影像文件名中提取DOY（年积日）
    :param filename: 哨兵影像文件名（如S2A_MSIL2A_20230503T032541_N0509_R018_T48RXS_20230503T075656.SAFE）
    :param start_date: 起始日期字符串，格式为 'YYYY-MM-DD'
    :return: DOY（年积日，如20230503）
    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    doy_str = os.path.basename(filename).split('_')[2][:8]
    date_obj = datetime.strptime(doy_str, '%Y%m%d')
    doy = (date_obj - start_date).days
    return doy

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
    return resampled_ds

def calculate_tsp(growth_tif, growth_scl_tif,mature_tif,mature_scl_tif,Non_growing_tif, output_tif):
    """
    计算TSP指数并导出为GeoTIFF文件
    :param growth_tif: 生长期的多波段GeoTIFF文件路径
    :growth_scl_tif: 生长期的SCL波段GeoTIFF文件路径
    :param mature_tif: 成熟期的多波段GeoTIFF文件路径
    :mature_scl_tif: 成熟期的SCL波段文件路径
    :Non_growing_tif: 非生长期的多波段GeoTIFF文件路径
    :param output_tif: 输出的TSP指数GeoTIFF文件路径
    """
    # 从文件名中提取DOY
    doy_growth = extract_doy_from_filename(growth_tif)
    doy_mature = extract_doy_from_filename(mature_tif)
    GL = doy_mature - doy_growth
    print('生长时间长度：', GL)

    # 读取生长期影像并获取参考空间参数
    growth_dataset = gdal.Open(growth_tif)
    if growth_dataset is None:
        raise FileNotFoundError(f"无法打开文件: {growth_tif}")

    target_proj = growth_dataset.GetProjection()
    target_geotransform = growth_dataset.GetGeoTransform()
    target_cols = growth_dataset.RasterXSize
    target_rows = growth_dataset.RasterYSize

    # 读取生长期波段数据
    growth_red = growth_dataset.GetRasterBand(4).ReadAsArray().astype(np.float32)
    growth_nir = growth_dataset.GetRasterBand(8).ReadAsArray().astype(np.float32)
    growth_re2 = growth_dataset.GetRasterBand(6).ReadAsArray().astype(np.float32)


    # 读取成熟期影像
    mature_dataset = gdal.Open(mature_tif)
    if mature_dataset is None:
        raise FileNotFoundError(f"无法打开文件: {mature_tif}")

    # 检查成熟期影像是否需要重采样
    if not check_raster_alignment(mature_dataset, growth_dataset):
        print("成熟期影像与生长期影像未对齐，正在重采样...")
        mature_dataset = resample_to_target(mature_dataset, target_proj, target_geotransform,
                                            target_cols, target_rows, gdal.GRA_Bilinear)

    # 读取成熟期波段数据
    mature_red = mature_dataset.GetRasterBand(4).ReadAsArray().astype(np.float32)
    mature_nir = mature_dataset.GetRasterBand(8).ReadAsArray().astype(np.float32)
    mature_re2 = mature_dataset.GetRasterBand(6).ReadAsArray().astype(np.float32)
    # 计算NDVI（成熟期）
    ndvi = (mature_nir - mature_red) / (mature_nir + mature_red + 1e-10)
    # ndvi_mask2 = (ndvi > 0.2) 
    # ndvi = np.where(ndvi_mask2, ndvi, np.nan)

    # 读取非生长期影像,排除高活力植被和非植被区域的干扰
    Non_growing_image, Non_growing_info,Non_growing_data = read_multi_bands(Non_growing_tif)
    Non_growing_ndvi = NDVI(Non_growing_info, Non_growing_data).astype(np.float32)
    p1 = Non_growing_ndvi **2
    p1 = np.where(p1 > 0.1, p1, 1e-8)
    # Non_growing_ndvi = np.where(Non_growing_ndvi > 0, Non_growing_ndvi, np.nan)
    # # p1 = 1- Non_growing_ndvi ^2

    # 读取SCL影像
    growth_scl_dataset = gdal.Open(growth_scl_tif)
    if growth_scl_dataset is None:
        raise FileNotFoundError(f"无法打开文件: {growth_scl_tif}")
    # 检查SCL影像是否需要重采样
    if not check_raster_alignment(growth_scl_dataset, growth_dataset):
        print("SCL影像与生长期影像未对齐，正在重采样...")
        growth_scl_dataset = resample_to_target(growth_scl_dataset, target_proj, target_geotransform,
                                         target_cols, target_rows, gdal.GRA_NearestNeighbour)
    # 读取SCL波段数据
    growth_scl_band = growth_scl_dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)
    # 创建云阴影掩膜
    shadow_mask = (growth_scl_band == 3)

    mature_scl_dataset = gdal.Open(mature_scl_tif)
    if mature_scl_dataset is None:
        raise FileNotFoundError(f"无法打开文件: {mature_scl_tif}")
    # 检查SCL影像是否需要重采样
    if not check_raster_alignment(mature_scl_dataset, growth_dataset):
        print("SCL影像与生长期影像未对齐，正在重采样...")
        mature_scl_dataset = resample_to_target(mature_scl_dataset, target_proj, target_geotransform,
                                         target_cols, target_rows, gdal.GRA_NearestNeighbour)
    # 读取SCL波段数据
    mature_scl_band = mature_scl_dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)
    # 创建云掩膜
    cloud_mask = (mature_scl_band == 9) | (mature_scl_band == 8)

    # 计算红边-2斜率并应用掩膜
    slope_re2 = (mature_re2 - growth_re2) * 0.0001/ (GL*0.01)
    # slope_re2[shadow_mask] = np.nan
    # slope_re2[cloud_mask] = np.nan
    slope_re2 = 1 / (1 + np.exp(-slope_re2))
    # slope_re2_normalized = (slope_re2 - np.nanmin(slope_re2)) / (np.nanmax(slope_re2) - np.nanmin(slope_re2) + 1e-10)

    # 计算TSP指数
    tsp = ndvi * slope_re2 *(1-p1)
    # tsp = 1 / (1 + np.exp(-tsp))

    # tsp_normalized = (tsp - np.nanmin(tsp)) / (np.nanmax(tsp) - np.nanmin(tsp) + 1e-10)
    # tsp[np.isnan(tsp)] = 0

    # 导出结果
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_tif, target_cols, target_rows, 1, gdal.GDT_Float32)
    out_dataset.SetGeoTransform(target_geotransform)
    out_dataset.SetProjection(target_proj)
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(tsp)
    out_band.FlushCache()

    print(f"已导出TSP指数至: {output_tif}")

def NDVI(image_info, image_array, output_tif=None, block_size=1024):
    W,H = image_info[1], image_info[2]
    score = np.ones((H,W), dtype=np.float32)
    if image_array.shape[0] == 12:
        image_array = image_array[[1, 2, 3, 4, 5, 6, 7, 8, 10, 11]]
    # blue green red red-edge1 red-edge2 red-edge3 nir narrow-nir swir-1 swir-2
    total_blocks = (image_info[1] // block_size + 1) * (image_info[2] // block_size + 1)
    
    with tqdm(total=total_blocks, desc="Calculating NDVI") as pbar:
        for h_start in range(0, H, block_size):
            for w_start in range(0, W, block_size):
                h_end = min(h_start + block_size, H)
                w_end = min(w_start + block_size, W)

                block = image_array[:, h_start:h_end,w_start:w_end] * 1e-4 
                block_score = np.ones((h_end - h_start, w_end - w_start), dtype=np.float32)
                block_score = np.minimum(block_score, (block[[0, 1, 2], :, :].sum(axis=0) - 0.2) / 0.6)  # rgb
                ndvi_block = (block[6, :, :] - block[2, :, :]) / (block[6, :, :] + block[2, :, :] + 1e-8)
                # ndvi_block = np.where(cloud | dark, -1, ndvi_block)
                ndvi_block = np.clip(ndvi_block, -1, 1)
                score[h_start:h_end,w_start:w_end] = ndvi_block

                pbar.update(1)

    if output_tif:
        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(output_tif, image_info[1], image_info[2], 1, gdal.GDT_Float32)
        out_dataset.SetGeoTransform(image_info[3])
        out_dataset.SetProjection(image_info[4])
        out_band = out_dataset.GetRasterBand(1)
        out_band.WriteArray(score)
        out_band.FlushCache()
    
    return score

def CR_TSP(growth_tif, mature_tif, scl_tif, output_dir):
    """
    计算TSP指数并导出为GeoTIFF文件
    :param growth_tif: 生长期的多波段GeoTIFF文件路径
    :param mature_tif: 成熟期的多波段GeoTIFF文件路径
    :param scl_tif: SCL波段的GeoTIFF文件路径
    :param output_tif: 输出的TSP指数GeoTIFF文件路径
    """
    # 从文件名中提取DOY
    doy_growth = extract_doy_from_filename(growth_tif)
    doy_mature = extract_doy_from_filename(mature_tif)
    print('生长时间长度：', doy_mature - doy_growth)

    # 读取生长期影像并获取参考空间参数
    grow_image, grow_info,grow_data = read_multi_bands(growth_tif)
    mature_image, mature_info, mature_data = read_multi_bands(mature_tif)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    growth_path = os.path.join(output_dir, f'NDVI_RUQ_{doy_growth}_dark.tif')
    mature_path = os.path.join(output_dir, f'NDVI_RUQ_{doy_mature}_dark.tif')
    NDVI(grow_info, grow_data, growth_path)
    NDVI(mature_info, mature_data, mature_path)


# 示例调用
if __name__ == '__main__':
    growth_tif = r'/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/research/xiangcheng/2023/S2/S2A_MSIL2A_20230517T030521_N0509_R075_T49SGT_20230517T080949.SAFE.tif'  # 生长期的多波段GeoTIFF文件路径
    mature_tif = r'/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/research/xiangcheng/2023/S2/S2B_MSIL2A_20230830T030529_N0509_R075_T49SGT_20230830T064055.SAFE.tif'  # 成熟期的多波段GeoTIFF文件路径
    growth_scl_tif = r'/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/research/xiangcheng/2023/S2/SCL/Sentinel2_SCL_T48RUP_20230506.tif'  # SCL波段文件路径  
    mature_scl_tif = r'/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/research/xiangcheng/2023/S2/SCL/Sentinel2_SCL_T48RUP_20230814.tif'  # SCL波段文件路径
    Non_growing_tif = r'/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/research/xiangcheng/2023/S2/S2A_MSIL2A_20230308T030551_N0509_R075_T49SGT_20230308T062850.SAFE.tif'
    output_tif = r'/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/xiangcheng/2023/S2/TSP/TSP_SGT.tif'  # 输出TSP指数文件路径
    # output_dir = r'/mnt/e/Research/tobacco/dataset/DATA/WEINING_HEZHANG/NDVI'  # 输出TSP指数文件路径
    calculate_tsp(growth_tif, growth_scl_tif,mature_tif,mature_scl_tif,Non_growing_tif, output_tif)



