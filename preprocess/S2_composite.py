import os
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.transform import rowcol
from shapely.geometry import box, mapping
from datetime import datetime
import geopandas as gpd
from tqdm import tqdm

def process_block_box(geom, raster_folder, start_date, fixed_size=(512, 512)):
    """
    处理指定几何范围内的栅格数据，并提取与几何范围匹配的栅格块。

    参数:
        geom (shapely.geometry): 目标几何范围（如多边形）。
        raster_folder (str): 包含栅格文件的文件夹路径。
        start_date (datetime): 起始日期，用于计算 DOY（Day of Year）。
        fixed_size (tuple): 固定裁剪大小 (height, width)。

    返回:
        block_data (np.ndarray): 裁剪后的栅格数据，形状为 [H*W, T, C]。
        size (tuple): 裁剪后的实际大小 (height, width)。
    """
    block_data = []
    H_fixed, W_fixed = fixed_size  # 固定裁剪大小

    for raster_path in os.listdir(raster_folder):
        if raster_path.startswith('S2') and raster_path.endswith('.tif'):
            raster_path = os.path.join(raster_folder, raster_path)
            with rasterio.open(raster_path) as src:
                src_bounds = box(*src.bounds)
                if not src_bounds.contains(geom):
                    continue

                # 获取几何范围的左上角坐标
                minx, miny, maxx, maxy = geom.bounds
                transform = src.transform

                # 将左上角坐标转换为影像的行列号
                row_start, col_start = rowcol(transform, minx, maxy)  # 左上角
                row_end, col_end = rowcol(transform, maxx, miny)  # 右下角

                # 计算固定大小的裁剪范围
                row_start = max(0, row_start)
                col_start = max(0, col_start)
                row_end = min(src.height, row_start + H_fixed)
                col_end = min(src.width, col_start + W_fixed)

                # 裁剪影像
                window = rasterio.windows.Window(col_start, row_start, W_fixed, H_fixed)
                out_image = src.read(window=window)

                # 选择需要的波段（排除 band 1 和 band 10）
                out_image = out_image[[1, 2, 3, 4, 5, 6, 7, 8, 10, 11]]  # 假设波段索引从 0 开始

                # 处理 TSP 数据
                parts = os.path.basename(raster_path).split('_')
                with rasterio.open(os.path.join(raster_folder, f"TSP_{parts[5][-3:]}.tif")) as tsp_src:
                    tsp_image = tsp_src.read(window=window)[0]  # 读取第一个波段

                # 调整大小（如果裁剪后小于固定大小，填充 0）
                if out_image.shape[1] < H_fixed or out_image.shape[2] < W_fixed:
                    padded_image = np.zeros((10, H_fixed, W_fixed), dtype=out_image.dtype)
                    padded_image[:, :out_image.shape[1], :out_image.shape[2]] = out_image
                    out_image = padded_image

                    padded_tsp = np.zeros((H_fixed, W_fixed), dtype=tsp_image.dtype)
                    padded_tsp[:tsp_image.shape[0], :tsp_image.shape[1]] = tsp_image
                    tsp_image = padded_tsp

                # 将数据展平并拼接
                out_image = out_image.reshape(10, -1).T  # [H*W, C]
                tsp_image = tsp_image.reshape(-1, 1)  # [H*W, 1]
                out_image = np.hstack((out_image, tsp_image))  # [H*W, C+1]

                # 添加 DOY 列
                date_str = parts[2][:8]
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                doys = (date_obj - start_date).days
                doys_column = np.full((out_image.shape[0], 1), fill_value=doys)
                out_image = np.hstack((out_image, doys_column))  # [H*W, C+2]

                block_data.append(out_image)

    # 将数据堆叠并转置
    block_data = np.stack(block_data, axis=0)  # [T, H*W, C]
    block_data = np.transpose(block_data, (1, 0, 2))  # [H*W, T, C]
    size = (H_fixed, W_fixed)  # 固定大小

    return block_data, size

def composite_S2(shp_path, raster_folder):
    """
    处理所有区块的栅格数据。

    参数:
        shp_path (str): Shapefile 文件路径。
        raster_folder (str): 包含栅格文件的文件夹路径。
    """
    print("Reading shapefile...")
    blockpartition = gpd.read_file(shp_path)
    start_date = datetime.strptime('2023-01-01', '%Y-%m-%d')

    print("Processing each block in parallel...")
    for idx, row in tqdm(blockpartition.iterrows(), total=len(blockpartition), desc="Processing blocks"):
        location = row['grid_id']
        out_path = os.path.join(raster_folder, 'box2', f'data_box{location}.npy')
        if os.path.exists(out_path):
            print(f"Skipping block {location}...")
            continue

        print(f"Processing block {location}...")
        block_data, size = process_block_box(row.geometry, raster_folder, start_date)
        print(f"Block size: {size}")
        np.save(out_path, block_data)
        print(f"Saving data to {out_path}...")

if __name__ in "__main__":
    shp_path = "/mnt/e/Research/tobacco/dataset/LABEL/weining/weining_grids_4000m_wgs84_b100m.shp"
    raster_folder = "/mnt/e/Research/tobacco/dataset/DATA/WEINING_HEZHANG"
    composite_S2(shp_path,raster_folder)