import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from shapely.geometry import Point
import geopandas as gpd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import re

# --------------------------
# 工具类和预加载逻辑
# --------------------------
class TSPCache:
    def __init__(self):
        self.cache = {}

    def get_tsp_src(self, tsp_path):
        if tsp_path not in self.cache:
            if not os.path.exists(tsp_path):
                raise FileNotFoundError(f"TSP文件未找到: {tsp_path}")
            self.cache[tsp_path] = rasterio.open(tsp_path)
        return self.cache[tsp_path]
    
SELECTED_IDS = [12457,12537,17154, 17365, 17432,20238,20091,30286,30037,30631,40020,40148,50737,50574]
tsp_cache = TSPCache()

def tif_to_timeseries_npy(tif_dir, output_dir, time_axis=0):
    """
    将TIF文件重组为(T, C, H, W)格式的npy文件
    :param tif_dir: TIF文件目录路径
    :param output_dir: 输出npy目录
    :param time_axis: 时间维度位置 (0或-1)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析文件结构
    file_groups = {}
    time_pattern = re.compile(r"sample_(\d+)_date_(\d{8})\.tif")
    
    # 遍历文件并分组
    for fname in os.listdir(tif_dir):
        match = time_pattern.match(fname)
        if match:
            sample_id = int(match.group(1))
            date_str = match.group(2)
            if sample_id not in file_groups:
                file_groups[sample_id] = []
            file_groups[sample_id].append((date_str, fname))
    
    # 处理每个样本
    for sample_id, files in tqdm(file_groups.items(), desc="Processing Samples"):
        # 按时间排序
        files.sort(key=lambda x: x[0])
        sorted_files = [os.path.join(tif_dir, f[1]) for f in files]
        
        # 验证数据一致性
        ref_shape, ref_dtype = None, None
        for fpath in sorted_files:
            with rasterio.open(fpath) as src:
                if ref_shape is None:
                    ref_shape = (src.count, src.height, src.width)
                    ref_dtype = src.dtypes[0]
                else:
                    assert (src.count, src.height, src.width) == ref_shape, \
                        f"尺寸不一致: {fpath}"
                    assert src.dtypes[0] == ref_dtype, \
                        f"数据类型不一致: {fpath}"
        
        # 初始化数据立方体
        n_time = len(sorted_files)
        n_bands, height, width = ref_shape
        if time_axis == 0:
            data_cube = np.empty((n_time, n_bands, height, width), dtype=ref_dtype)
        else:
            data_cube = np.empty((n_bands, height, width, n_time), dtype=ref_dtype)
        
        # 填充数据
        timestamps = []
        for t_idx, fpath in enumerate(sorted_files):
            with rasterio.open(fpath) as src:
                # 读取数据 [C, H, W]
                img_data = src.read()
                date_str = re.search(r"date_(\d{8})", fpath).group(1)
                timestamps.append(datetime.strptime(date_str, "%Y%m%d"))
                
                if time_axis == 0:
                    data_cube[t_idx] = img_data
                else:
                    data_cube[..., t_idx] = img_data
        
        # 保存数据
        output_path = os.path.join(output_dir, f"sample_ID_{sample_id}.npy")
        np.save(output_path, data_cube)


def export_patch_to_geotiff(data, transform, crs, output_path):
    """将单个patch数据导出为GeoTIFF"""
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],  # 波段数
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=0
    ) as dst:
        dst.write(data)


def get_raster_metadata(raster_folder):
    raster_files = [f for f in os.listdir(raster_folder) if f.endswith('.tif')]
    metadata = []
    for f in raster_files:
        # if f.startswith('RUP'):
        parts = f.split('_')
        try:
            date_str = parts[-1][:2]
            print(f)
            metadata.append({
                'path': os.path.join(raster_folder, f),
                'date_str': date_str
            })
        except IndexError:
            continue
    return metadata

def check_alignment(src, tsp_src):
    return (
        tsp_src.width == src.width and
        tsp_src.height == src.height
    )

# --------------------------
# 核心处理逻辑
# --------------------------
def get_valid_window(src, x, y, patch_size=5):
    """计算有效的5x5窗口"""
    try:
        row, col = src.index(x, y)
    except rasterio.errors.RasterioIOError:
        return None

    # 计算窗口边界
    offset = patch_size // 2
    window = Window(
        col_off=col - offset,
        row_off=row - offset,
        width=patch_size,
        height=patch_size
    )

    # 检查边界有效性
    if (window.col_off >= 0 and 
        window.row_off >= 0 and 
        (window.col_off + window.width) <= src.width and 
        (window.row_off + window.height) <= src.height):
        return window
    return None

def process_block(row_dict, raster_metadata, start_date, label_field, patch_size, tsp_path, output_dir):
    """改进后的处理块函数，增加栅格导出功能"""
    row = gpd.GeoDataFrame([row_dict], geometry='geometry').iloc[0]
    x, y = row.geometry.x, row.geometry.y
    patch_data = []
    
    for meta in raster_metadata:
        raster_path = meta['path']
        date_str = meta['date_str']
        # import ipdb;ipdb.set_trace()
        with rasterio.open(raster_path) as src:
            # 获取有效窗口
            window = get_valid_window(src, x, y, patch_size)
            if not window:
                continue

            # 读取数据 [C, H, W]
            out_image = src.read(window=window)
            try:
                tsp_src = tsp_cache.get_tsp_src(tsp_path)
                tsp_window = get_valid_window(tsp_src, x, y, patch_size)
                if not tsp_window or not check_alignment(src, tsp_src):
                    raise ValueError("TSP数据无效或未对齐")
                tsp_image = tsp_src.read(1, window=tsp_window)  # [5,5]
            except Exception as e:
                print(f"跳过 {raster_path}: {str(e)}")
                continue

            # 构建时间特征
            tsp_image = tsp_image.reshape(1, patch_size, patch_size)  # [1,5,5]
            doys = np.full((1, patch_size, patch_size), int(date_str))         # [1,5,5]

            # 拼接特征维度 [12,5,5]
            combined = np.concatenate([out_image, tsp_image, doys], axis=0)
            patch_data.append(combined)

            # 导出GeoTIFF
            sample_id = row_dict['ID']  
            if sample_id in SELECTED_IDS:
                # timestamp = datetime.strptime(date_str, '%Y%m%d').strftime('%Y%m%d')
                output_filename = f"sample_{sample_id}_{date_str}.tif"
                output_path = os.path.join(output_dir, output_filename)
                # 获取地理参考信息
                crs = src.crs
                transform = rasterio.windows.transform(window, src.transform)
                # 导出原始影像patch
                export_patch_to_geotiff(out_image, transform, crs, output_path)

    return patch_data, getattr(row, label_field)


# --------------------------
# 并行调度逻辑（保持不变）
# --------------------------
def sample_time_series(shp_path, raster_folder,tsp_path,out_path, tif_output_dir,label_field,patch_size):
    # 创建输出目录
    os.makedirs(tif_output_dir, exist_ok=True)
    print("Reading shapefile...")
    blockpartition = gpd.read_file(shp_path)
    raster_metadata = get_raster_metadata(raster_folder)
    start_date = datetime.strptime('2023-01-01', '%Y-%m-%d')

    print("Processing in parallel...")
    data, label, meta_data = [], [], []
    # for row in blockpartition.itertuples():
    #     process_block(row._asdict(), raster_metadata, start_date, label_field,patch_size,tsp_path,tif_output_dir)  # 预加载

    with ProcessPoolExecutor(max_workers=os.cpu_count()//2) as executor:
        futures = [
            executor.submit(
                process_block, 
                row._asdict(),
                [m for m in raster_metadata],
                start_date,
                label_field,
                patch_size,
                tsp_path,
                tif_output_dir  # 新增输出目录参数
            )
            for row in blockpartition.itertuples()
        ]
        
        for future in tqdm(futures, total=len(futures)):
            try:
                point_data,point_label = future.result()
                data.append(point_data)
                label.append(point_label)
            except Exception as e:
                print(f"处理失败: {str(e)}")
    
    # 保存为[N, T, C, 5, 5]结构
    dataset = [data,label]
    np.save(out_path, dataset)
    
    print("Processing complete.")
    return dataset

# 使用示例（保持不变）
shp_path = '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/sample/polygon/xiangcheng_samples_0311.shp'
raster_folder = '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/research/xiangcheng/2023/S2/month_mean2'
tsp_path = '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/TSP/TSP_SGT.tif'
out_path = os.path.join("/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/sample/npy", 'XC_test.npy')
sample_tif_dir = os.path.join(os.path.dirname(out_path), 'sample_tif_dir')
label_field = 'code'
patch_size = 7
result = sample_time_series(shp_path, raster_folder,tsp_path,out_path,sample_tif_dir,label_field,patch_size)
