import os
import numpy as np
import rasterio
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
from shapely.geometry import box, mapping
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

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

tsp_cache = TSPCache()

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
    # import ipdb;ipdb.set_trace()
    return (
        # tsp_src.bounds == src.bounds and
        # tsp_src.transform == src.transform and
        tsp_src.width == src.width and
        tsp_src.height == src.height
    )


# --------------------------
# 核心处理逻辑
# --------------------------
def process_block(row_dict, raster_metadata, label_field):
    # 将字典转换回行对象
    row = gpd.GeoDataFrame([row_dict], geometry='geometry').iloc[0]
    point_geom = [mapping(row.geometry)]
    point_data = []
    
    for meta in raster_metadata:
        raster_path = meta['path']
        date_str = meta['date_str']
        
        with rasterio.open(raster_path) as src:
            src_bounds = box(*src.bounds)
            if not src_bounds.contains(row.geometry):
                continue
            
            # 裁剪原始影像
            out_image, _ = mask(src, point_geom, crop=True)

            # print(out_image.shape)
            
            # 加载TSP影像
            tsp_path = os.path.join('/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/TSP', f"TSP_SGT.tif")
            # tsp_path = os.path.join('/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/research/xiangcheng/2023/S2/TSP', f"TSP_SGT.tif")
            # tsp_path = os.path.join('/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/research/xiangcheng/2023/S2/TSP', f"TSP_SGT.tif")
            # tsp_path = os.path.join('/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/xuanwei/2023/S2/RUP/TSP', f"TSP_RUP.tif")
            
            
            try:
                tsp_src = tsp_cache.get_tsp_src(tsp_path)
                if not check_alignment(src, tsp_src):
                    raise ValueError("TSP影像未对齐")
                tsp_image, _ = mask(tsp_src, point_geom, crop=True)
                tsp_image = tsp_image[0]  # [H, W]
            except Exception as e:
                print(f"跳过 {raster_path}: {str(e)}")
                continue
            
            # 拼接数据
            doys = np.full((1), int(date_str)) # [1,1]
            if out_image.shape[0] == 12:
                out_image = out_image[[1, 2, 3, 4, 5, 6, 7, 8, 10, 11]]
            out_image = out_image.reshape(10)
            tsp_image = tsp_image.reshape(1)
            combined = np.concatenate([out_image, tsp_image, doys], axis=0)
            # combined = np.concatenate([out_image, doys], axis=0)
            point_data.append(combined)

    sequence_length = len(point_data)
    # import ipdb;ipdb.set_trace()
    if sequence_length == 0:
        return None
    meta_data = [getattr(row, label_field)]
    return point_data, meta_data, getattr(row, label_field)

# --------------------------
# 并行调度逻辑
# --------------------------
def sample_time_series(shp_path, raster_folder, label_field):
    print("Reading shapefile...")
    blockpartition = gpd.read_file(shp_path)
    raster_metadata = get_raster_metadata(raster_folder)
    # import ipdb;ipdb.set_trace()
    print("Processing in parallel...")
    data, label, meta_data = [], [], []
    i = 0
    for row in blockpartition.itertuples():
        point_data, meta, point_label = process_block(row._asdict(),raster_metadata,label_field)
        data.append(point_data)
        meta_data.append(meta)
        label.append(point_label)
        i = i+1
        if i>10:
            break
    
    # with ProcessPoolExecutor(max_workers=os.cpu_count()//2) as executor:
    #     futures = [
    #         executor.submit(
    #             process_block, 
    #             row._asdict(),  # 将行对象转换为字典
    #             [m for m in raster_metadata],
    #             label_field
    #         )
    #         for row in blockpartition.itertuples()
    #     ]
        
    #     for future in tqdm(futures, total=len(futures)):
    #         try:
    #             if future.result() is not None:
    #                 point_data, meta, point_label = future.result()
    #                 data.append(point_data)
    #                 meta_data.append(meta)
    #                 label.append(point_label)
    #         except Exception as e:
    #             print(f"处理失败: {str(e)}")
    import ipdb;ipdb.set_trace()
    dataset = [data, label]
    

    out_path = os.path.join(os.path.dirname(shp_path), 'XC.npy')
    print(f"Saving data to {out_path}...")
    np.save(out_path, dataset)
    
    # meta_df = pd.DataFrame(meta_data, columns=['point_id', 'label', 'location'])
    # meta_out_path = os.path.join(os.path.dirname(shp_path), 'metadata_TSP_WN_all.xlsx')
    # print(f"Saving metadata to {meta_out_path}...")
    # meta_df.to_excel(meta_out_path, index=False)
    
    print("Processing complete.")
    return data

# # shp_path = '/root/models/SITS_MoCo/data/weining/raw_sample/tabacco_weining_bu.shp'
# shp_path = '/mnt/e/2024Work/tobacco/实验/points_train/train_val_202501008.shp'
shp_path = '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/sample/polygon/xiangcheng_samples_0311.shp'

# raster_folder = '/mnt/e/2024Work/tobacco/data/S2'
# raster_folder = '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/xuanwei/2023/S2/RUP/month_mean'
raster_folder = '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/research/xiangcheng/2023/S2/month_mean2'
label_field = 'code'

result = sample_time_series(shp_path, raster_folder, label_field)
# result = sample_time_series(shp_path, raster_folder)