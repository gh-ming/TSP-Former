import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split


def process_block(row, raster_folder, start_date, label_field, point_id):
    point_geom = [mapping(row.geometry)]
    point_data = []
    
    for raster_path in os.listdir(raster_folder):
        if raster_path.endswith('.tif'):
            raster_path = os.path.join(raster_folder, raster_path)
            with rasterio.open(raster_path) as src:
                src_bounds = box(*src.bounds)
                
                # 检查影像是否包含点几何
                if not src_bounds.contains(row.geometry):
                    continue
                
                # 解析文件名中的类型、日期和tile_id
                parts = os.path.basename(raster_path).split('_')
                type_str = parts[0]
                date_str = parts[2][:8]
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                doys = (date_obj - start_date).days
                tile_id = parts[5][-3:]
                
                # 裁剪影像到点几何区域
                out_image, out_transform = mask(src, point_geom, crop=True)
                out_image = out_image[[1, 2, 3, 4, 5, 6, 7, 8, 10, 11]]  # 排除波段1和波段10
                
                # 加载TSP影像
                tsp_path = os.path.join(raster_folder, f"TSP_{tile_id}.tif")
                if not os.path.exists(tsp_path):
                    raise FileNotFoundError(f"TSP文件未找到: {tsp_path}")
                
                with rasterio.open(tsp_path) as tsp_src:
                    # 检查TSP影像是否与原始影像对齐
                    if (tsp_src.bounds != src.bounds or
                        tsp_src.transform != src.transform or
                        tsp_src.width != src.width or
                        tsp_src.height != src.height):
                        raise ValueError("TSP影像与原始影像未对齐")
                    
                    # 裁剪TSP影像到点几何区域
                    tsp_image, _ = mask(tsp_src, point_geom, crop=True)
                    tsp_image = tsp_image[0]  # 提取TSP值
                
                # 将TSP作为新的一维加入
                out_image = np.append(out_image, tsp_image)
                
                # 添加时间信息
                out_image = np.append(out_image, doys)
                
                # 将数据展平并添加到点数据列表
                point_data.append(out_image.flatten())
    
    # 构建元数据
    sequence_length = len(point_data)
    meta_data = [point_id, sequence_length, row[label_field], row['location']]
    point_label = row[label_field]
    
    return point_data, meta_data, point_label

# def process_block_box(geom, raster_folder, start_date):
#     block_data = []
    
#     for raster_path in os.listdir(raster_folder):
#         if raster_path.endswith('.tif'):
#             raster_path = os.path.join(raster_folder, raster_path)
#             with rasterio.open(raster_path) as src:
#                 src_bounds = box(*src.bounds)
                
#                 # 检查影像是否与几何区域相交
#                 if not src_bounds.intersects(geom):
#                     continue
                
#                 # 解析文件名中的日期和tile_id
#                 parts = os.path.basename(raster_path).split('_')
#                 date_str = parts[2][:8]
#                 date_obj = datetime.strptime(date_str, '%Y%m%d')
#                 tile_id = parts[5][-3:]
#                 doys = (date_obj - start_date).days
                
#                 # 裁剪影像到几何区域
#                 out_image, out_transform = mask(src, [mapping(geom)], crop=True)
#                 _, H, W = out_image.shape
                
#                 # 选择需要的波段（排除波段1和波段10）
#                 out_image = out_image[[1, 2, 3, 4, 5, 6, 7, 8, 10, 11]]  # [C, H, W]
#                 out_image = out_image.reshape(10, -1).T  # [H*W, C]
                
#                 # 加载TSP影像
#                 tsp_path = os.path.join(raster_folder, f"TSP_{tile_id}.tif")
#                 if not os.path.exists(tsp_path):
#                     raise FileNotFoundError(f"TSP文件未找到: {tsp_path}")
                
#                 with rasterio.open(tsp_path) as tsp_src:
#                     # 检查TSP影像是否与原始影像对齐
#                     if (tsp_src.bounds != src.bounds or
#                         tsp_src.transform != src.transform or
#                         tsp_src.width != src.width or
#                         tsp_src.height != src.height):
#                         raise ValueError("TSP影像与原始影像未对齐")
                    
#                     # 裁剪TSP影像到几何区域
#                     tsp_image, _ = mask(tsp_src, [mapping(geom)], crop=True)
#                     tsp_image = tsp_image[0].reshape(-1, 1)  # [H*W, 1]
                
#                 # 将TSP作为新的一维加入
#                 out_image = np.hstack((out_image, tsp_image))  # [H*W, C+1]
                
#                 # 添加时间信息
#                 doys_column = np.full((out_image.shape[0], 1), fill_value=doys)  
#                 out_image = np.hstack((out_image, doys_column))  # [H*W, C+2]
                
#                 block_data.append(out_image)
    
#     # 将数据堆叠并调整维度
#     block_data = np.stack(block_data, axis=0)  # [T, H*W, C+2]
#     block_data = np.transpose(block_data, (1, 0, 2))  # [H*W, T, C+2]
#     size = (H, W)
    
#     return block_data, size

# def sample_time_series(shp_path, raster_folder):
#     print("Reading shapefile...")
#     blockpartition = gpd.read_file(shp_path)
    
#     data = []
#     start_date = datetime.strptime('2023-01-01', '%Y-%m-%d')
    
#     print("Processing each block in parallel...")

#     for idx, row in blockpartition.iterrows():
#         block_data,size = process_block_box(row.geometry, raster_folder, start_date)
#         data.append(block_data)
#         data.append(size)
#     # with ProcessPoolExecutor() as executor:
#     #     futures = [executor.submit(process_block_box, row.geometry, raster_folder, start_date) 
#     #                for idx, row in blockpartition.iterrows()]
        
#     #     for future in tqdm(futures, total=len(futures)):
#     #         block_data= future.result()
#     #         data.append(block_data)
    
 
#     # 计算每个通道（C）的均值和标准差
#     out_path = os.path.join(os.path.dirname(shp_path), 'data_all.npy')
#     print(f"Saving data to {out_path}...")
#     np.save(out_path, data)
    
#     print("Processing complete.")

def sample_time_series(shp_path, raster_folder, label_field):
    print("Reading shapefile...")
    blockpartition = gpd.read_file(shp_path)
    
    data = []
    label = []
    meta_data = []
    start_date = datetime.strptime('2023-01-01', '%Y-%m-%d')
    
    print("Processing each block in parallel...")
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_block, row, raster_folder, start_date, label_field, row['ID']) 
                   for idx,row in blockpartition.iterrows()]
        
        for future in tqdm(futures, total=len(futures)):
            point_data, meta,point_label = future.result()
            data.append(point_data)
            meta_data.append(meta)
            label.append(point_label)

    
    print("Converting lists to numpy arrays...")
    # data = np.array(data)
    dataset = [data, label]
    

    out_path = os.path.join(os.path.dirname(shp_path), 'data_weining_2class_TSP_train.npy')
    print(f"Saving data to {out_path}...")
    np.save(out_path, dataset)
    
    meta_df = pd.DataFrame(meta_data, columns=['point_id', 'sequence_length', 'label', 'location'])
    meta_out_path = os.path.join(os.path.dirname(shp_path), 'metadata_TSP_train.xlsx')
    print(f"Saving metadata to {meta_out_path}...")
    meta_df.to_excel(meta_out_path, index=False)
    
    print("Processing complete.")
    return data


# # shp_path = '/root/models/SITS_MoCo/data/weining/raw_sample/tabacco_weining_bu.shp'
# shp_path = '/mnt/e/2024Work/tobacco/实验/points_train/train_val_202501008.shp'
shp_path = '/mnt/e/Research/tobacco/dataset/LABEL/weining/weining_points_train.shp'

# raster_folder = '/mnt/e/2024Work/tobacco/data/S2'
raster_folder = '/mnt/e/Research/tobacco/dataset/DATA/WEINING_HEZHANG'
label_field = 'type'

result = sample_time_series(shp_path, raster_folder, label_field)
# result = sample_time_series(shp_path, raster_folder)
