import os
import glob
from osgeo import gdal
from math import ceil
from tqdm import tqdm
import numpy as np
import geopandas as gpd
from pathlib import Path
from rasterio.mask import mask
import rasterio
from collections import defaultdict
import re
from affine import Affine
import rasterio.crs

def crop_raster_by_shp(raster_path, shp_path, output_path):
    gdf = gpd.read_file(shp_path)
    geoms = gdf.geometry.values
    geoms = [geom.__geo_interface__ for geom in geoms]
    # import ipdb;ipdb.set_trace()

    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform,
                     "compress": "lzw"})

    with rasterio.open(output_path, "w", **out_meta) as dest:
        for i in range(out_image.shape[0]):
            dest.write(out_image[i], indexes=i+1)
    del dest

def GetExtent(infile):
    ds = gdal.Open(infile)
    geotrans = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    min_x, max_y = geotrans[0], geotrans[3]
    max_x, min_y = geotrans[0] + xsize * geotrans[1], geotrans[3] + ysize * geotrans[5]
    ds = None
    return min_x, max_y, max_x, min_y

def RasterMosaic(file_list,output_path, data_type=None, nodata=0):
    extents = [GetExtent(infile) for infile in file_list]
    
    min_x = min(extent[0] for extent in extents)
    max_y = max(extent[1] for extent in extents)
    max_x = max(extent[2] for extent in extents)
    min_y = min(extent[3] for extent in extents)

    in_ds = gdal.Open(file_list[0])
    in_band = in_ds.GetRasterBand(1)
    if data_type is None:
        data_type = in_band.DataType

    geotrans = list(in_ds.GetGeoTransform())
    width, height = geotrans[1], geotrans[5]
    columns = ceil((max_x - min_x) / width)
    rows = ceil((max_y - min_y) / (-height))
    bands = in_ds.RasterCount
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, columns, rows, bands, data_type)
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0] = min_x
    geotrans[3] = max_y
    out_ds.SetGeoTransform(geotrans)
    inv_geotrans = gdal.InvGeoTransform(geotrans)

    out_data = np.full((bands, rows, columns), nodata, dtype=np.float16)

    with tqdm(total=len(file_list), desc="Mosaicking") as pbar:
        for in_fn in file_list:
            in_ds = gdal.Open(in_fn)
            in_gt = in_ds.GetGeoTransform()
            offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
            x, y = map(int, offset)
            
            for b in range(bands):
                data = in_ds.GetRasterBand(b+1).ReadAsArray()
                data = data.astype(np.float32)
                data[(data == nodata) | np.isnan(data)] = -np.inf
                
                x_end = x + data.shape[1]
                y_end = y + data.shape[0]
                
                x_end = min(x_end, columns)
                y_end = min(y_end, rows)
                
                out_data[b, y:y_end, x:x_end] = np.maximum(out_data[b, y:y_end, x:x_end], data[:y_end-y, :x_end-x])
            
            pbar.update(1)

    for b in range(bands):
        out_ds.GetRasterBand(b+1).WriteArray(out_data[b])
    out_ds.FlushCache()
    out_ds = None
    


def GetList(path, suffix='.tif'):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == suffix and os.path.splitext(file)[0][:3] == 'TSP':
                file_list.append(os.path.join(root, file))
    return file_list

def group_images_by_month(image_list):
    pattern = re.compile(r'.*_(\d{2})\.tif$')
    grouped_images = defaultdict(list)
    selected_months = ['06']
    for image in image_list:
        match = pattern.match(image)
        if match:
            month = match.group(1)
            if month in selected_months:
                grouped_images[month].append(image)
    return grouped_images

if __name__ == '__main__':
    image_path = r'/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/month_mean'
    shp_path = r'/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/research/SHP/weining.shp'

    imageList = GetList(image_path)
    temp_path = os.path.join(image_path, f'weining_mosaic_tsp.tif')
    output_path = os.path.join(image_path, f'weining_clip_tsp.tif')
    import ipdb;ipdb.set_trace()
    RasterMosaic(imageList,temp_path)
    crop_raster_by_shp(temp_path, shp_path, output_path) 
        
    # grouped_images = group_images_by_month(imageList)
    # # import ipdb;ipdb.set_trace()
    # for month, images in grouped_images.items():
    #     output_folder = os.path.join(image_path,'mosaic')
    #     mosaic_temp_path = os.path.join(output_folder, f'weining_temp_{month}.tif')
    #     output_path = os.path.join(output_folder, f'weining_mosaic_{month}.tif')
    #     if os.path.exists(output_path):
    #         continue
    #     RasterMosaic(images,mosaic_temp_path)
    #     print(f"Finish mosaicing for month {month}\n")

    #     os.makedirs(output_folder, exist_ok=True)

    #     crop_raster_by_shp(mosaic_temp_path, shp_path, output_path)
    #     print(f"Finish cropping for month {month}: {output_path}\n")
    #     os.remove(mosaic_temp_path)

