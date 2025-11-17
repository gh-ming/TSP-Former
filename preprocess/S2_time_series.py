import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping
from datetime import datetime
from tqdm import tqdm

def process_block_box(geom, raster_folder, start_date):
    block_data = []
    minx,miny,maxx,maxy = geom.total_bounds
    polygon = geom.geometry[0]

    raster_list = [ raster_path for raster_path in os.listdir(raster_folder) if raster_path.endswith('.tif')]#if raster_path.startswith('RVR')
    
    for raster_path in tqdm(raster_list):
        
            raster_path = os.path.join(raster_folder, raster_path)
            with rasterio.open(raster_path) as src:
                src_bounds = box(*src.bounds)
                if not src_bounds.contains(polygon):
                    continue
                
                # else:
                #     minx, miny, maxx, maxy = geom.bounds
                #     transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, src.res[0], src.res[1])
                # parts = os.path.basename(raster_path).split('_')
                # date_str = parts[2][:8]
                # date_obj = datetime.strptime(date_str, '%Y%m%d')
                # doys = (date_obj - start_date).days
                date_str = os.path.basename(raster_path).split('_')[-1]
                month = int(date_str[:2])
                doy = month
                print(doy)
                
                window = src.window(minx,miny,maxx,maxy)
                img = src.read(window=window)
                C, H, W= img.shape
                # out_image = out_image[[1, 2, 3, 4, 5, 6, 7, 8, 10, 11]]  # excluding band 1 and band 10
                # print(out_image.shape)
                # # 计算NDVI
                # ndvi = (out_image[:,6] - out_image[:,2]) / (out_image[:,6] + out_image[:,2] + 1e-8)
                # ndvi = ndvi.clip(-1, 1)[:, np.newaxis]   # [H*W, 1]
                # out_image = np.hstack((out_image, ndvi)) # [H*W, C+1]
                tsp_path = os.path.join('/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/TSP', f"TSP_SGT.tif")
                # tsp_path = os.path.join('/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/research/xiangcheng/2023/S2/TSP', f"TSP_SGT.tif")
                with rasterio.open(tsp_path) as tsp_src:
                    tsp_src_bounds = box(*tsp_src.bounds)
                    if not tsp_src_bounds.contains(polygon):
                        continue
                    # import ipdb;ipdb.set_trace()
                    tsp_img = tsp_src.read(window=window)
                    tsp_img = tsp_img.reshape(1,img.shape[1], img.shape[2])
                doy_array = np.full((1, img.shape[1], img.shape[2]), doy, dtype=np.int32)
                out_image = np.concatenate((img, tsp_img,doy_array), axis=0)
                block_data.append(out_image)
                # print(out_image.shape)

    block_data = np.stack(block_data, axis=0)  # [T, C, h, w]
    T, C, h, w = block_data.shape
    print('block data shape:',block_data.shape)
    
    return block_data

def composite_S2(shp_path, raster_folder):
    print("Reading shapefile...")
    blockpartition = gpd.read_file(shp_path)
    
    block_size = {}
    start_date = datetime.strptime('2023-01-01', '%Y-%m-%d')
    
    print("Processing each block in parallel...")

    out_path =os.path.join(os.path.dirname(shp_path),os.path.basename(shp_path)[:-4]+'.npy') 
    # import ipdb;ipdb.set_trace()
    block_data = process_block_box(blockpartition, raster_folder, start_date)
    # print(size)
    # block_size[location] = size
    np.save(out_path, block_data)
    print(f"Saving data to {out_path}...")
    
    print("Processing complete.")

if __name__ in "__main__":
    shp_path = "/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/box/XC_box2.shp"
    raster_folder = "/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/research/xiangcheng/2023/S2/month_mean2"
    composite_S2(shp_path,raster_folder)

