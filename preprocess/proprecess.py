import os
import csv
from datetime import datetime
import numpy as np
import rasterio
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # 导入 tqdm 进度条库

def save_metadata_to_csv(metadata, output_csv):
    keys = metadata[0].keys()
    metadata_sorted = sorted(metadata, key=lambda x: x['doy'])
    with open(output_csv, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(metadata_sorted)

def get_raster_metadata(raster_folder, start_date='2023-01-01'):
    raster_files = [f for f in os.listdir(raster_folder) if f.endswith('.tif')]
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    # import ipdb;ipdb.set_trace()
    metadata = []
    for f in raster_files:
        parts = f.split('_')
        try:
            date_str = parts[2][:8]
            tile_id = parts[5][-3:]
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            doys = (date_obj - start_date).days
            if tile_id == 'RVR':       

                metadata.append({
                    'doy': doys,
                    'date_str': date_str,
                    'tile_id': tile_id,
                    'path': os.path.join(raster_folder, f)
                })
        except (IndexError, ValueError):
            continue
    output_csv = os.path.join(raster_folder, 'raster_metadata.csv')
    if not os.path.exists(output_csv):
        save_metadata_to_csv(metadata, output_csv)
    return metadata

def process_month(tile_id,month, paths, select_bands):
    """处理单个月份数据的核心函数"""
    try:
        with rasterio.open(paths[0]) as ref:
            meta = ref.meta.copy()
            num_bands, shape = ref.count, ref.shape
            transform, bounds = ref.transform, ref.bounds
    except Exception as e:
        print(f"Error opening reference {paths[0]}: {e}")
        return

    valid_paths = []
    for path in paths:
        try:
            with rasterio.open(path) as src:
                if (src.transform == transform and src.shape == shape and
                    src.bounds == bounds):
                    valid_paths.append(path)
        except:
            continue

    if not valid_paths:
        return
    

    # 优化内存分配和数据类型
    avg_data = np.zeros((len(select_bands), *shape), dtype=np.float32)
    band_indices = [b for b in range(len(select_bands))]
    block_size = 1024
    total_blocks = (shape[0] // block_size + 1) * (shape[1] // block_size + 1)
    with tqdm(total=total_blocks, desc=f"Processing {tile_id} for month {month}") as pbar:
        for i in range(0, shape[0], block_size):
            for j in range(0, shape[1], block_size):
                window = rasterio.windows.Window(j, i, min(block_size, shape[1] - j), min(block_size, shape[0] - i))
                for path in valid_paths:
                    try:
                        with rasterio.open(path) as src:
                            data = src.read(select_bands, window=window).astype(np.float32)
                            avg_data[band_indices, i:i+block_size, j:j+block_size] += data
                    except Exception as e:
                        print(f"Error processing file {path} for month {month}: {e}")
                        continue
                pbar.update(1)
    # for path in valid_paths:
    #     with rasterio.open(path) as src:
    #         data = src.read(select_bands).astype(np.float32)
    #         avg_data[band_indices] += data

    avg_data[band_indices] /= len(valid_paths)
    avg_data = avg_data.astype(meta['dtype'])

    output_path = os.path.join(os.path.dirname(paths[0]), f'{tile_id}_avg_{month}.tif')
    meta.update({
        'count': len(select_bands),
        'compress': 'lzw'
    })
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(avg_data)
    return output_path

def average_rasters_by_month(metadata, workers=1):
    """并行处理每个月数据"""
    select_bands = [2, 3, 4, 5, 6, 7, 8,9, 11, 12]
    selected_months = ['01','04','05']
    
    # 按 tile_id 和月份分组
    tile_monthly_data = {}
    for item in metadata:
        date = datetime.strptime(item['date_str'], '%Y%m%d')
        month = date.strftime('%m')
        if month not in selected_months:
            continue
        tile_id = item['tile_id']
        key = (tile_id, date.strftime('%Y-%m'))
        tile_monthly_data.setdefault(key, []).append(item['path'])
    
    # import ipdb;ipdb.set_trace()
    # 使用 tqdm 显示总进度
    # for (tile_id, month), paths in tqdm(tile_monthly_data.items(), desc="Processing tiles and months"):
    #     process_month(tile_id,month, paths, select_bands)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_month, tile_id,month, paths, select_bands): month
            for (tile_id, month), paths in tile_monthly_data.items()
        }
        
        # 使用 tqdm 包装 as_completed，显示进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing months"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing month: {e}")

if __name__ == "__main__":
    raster_folder = "/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/se"
    start_date = '2023-01-01'
    
    # 获取并验证元数据
    print("Collecting metadata...")
    metadata = get_raster_metadata(raster_folder, start_date)
    
    # 使用4个进程并行计算，并显示进度条
    print("Calculating monthly averages...")
    average_rasters_by_month(metadata, workers=2)