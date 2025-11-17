import os 
import numpy as np 
import geopandas as gpd 
import rasterio 
from shapely.geometry import mapping
from rasterio.mask import mask 
from rasterio.warp import reproject, Resampling
from tqdm import tqdm 
import glob
from osgeo import gdal
import re
from affine import Affine
from collections import defaultdict
def read_shapefile(shp_path):
    gdf = gpd.read_file(shp_path)
    return gdf 

def resample_image(data, transform, src_crs, target_width, target_height):
    new_transform = transform * transform.scale(
        (data.shape[2] / target_width),
        (data.shape[1] / target_width)
    )

    resampled_data = np.zeros((data.shape[0], target_height, target_width), dtype=data.dtype)

    reproject(
        source=data,
        destination=resampled_data,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=new_transform,
        dst_crs=src_crs,
        resampling=Resampling.bilinear
    )

    return resampled_data, new_transform


def create_vrt(tiff_files,output_path,is_overviews=False):
    print(f"Totally {len(tiff_files)} tifs will be processed.")
    print("building vrt file.")
    # import ipdb;ipdb.set_trace()
    vrt_options = gdal.BuildVRTOptions(srcNodata=0, VRTNodata=0, hideNodata=True)
    gdal.BuildVRT(output_path, tiff_files, options=vrt_options)
    if is_overviews:
        print("building overviews.")
        vrt_ds = gdal.Open(output_path)
        vrt_ds.BuildOverviews('nearest', [2,4,8,16,32,64,128])
        vrt_ds.FlushCache()
        
def clip_vrt_with_shp(shp_path, vrt_files, output_dir):
    gdf = read_shapefile(shp_path)
    # import ipdb; ipdb.set_trace()
    geometries = [mapping(geom) for geom in gdf.geometry]         

    for vrt_file in vrt_files:
        vrt_name = os.path.basename(vrt_file)
        print(f"processing {vrt_name}")
        out_path = os.path.join(output_dir,vrt_name[:-4]+'.tif')
        with rasterio.open(vrt_file) as src:
            meta = src.meta.copy()
            out_image, out_transform = mask(src, geometries, crop=True)
            meta.update({
                "driver": 'GTiff',
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress":'lzw'
            })
        with rasterio.open(out_path,'w',**meta) as dest:
            dest.write(out_image)
        print(f"building {vrt_name} overviews.")
        out_ds = gdal.Open(out_path)
        out_ds.BuildOverviews('nearest', [2,4,8,16,32,64,128])
        out_ds.FlushCache()


def GetList(path, suffix='.tif'):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # if os.path.splitext(file)[1] == suffix:
            file_list.append(os.path.join(root, file))
    return file_list

def group_images_by_month(image_list):
    # pattern = re.compile(r'.*_(\d{2})\.tif$')
    pattern = re.compile(r'''^(.*?_avg_)\d{4}-(\d{2})(.*)$''',re.VERBOSE)
    grouped_images = defaultdict(list)
    # selected_months = ['06']
    for image in image_list:
        match = pattern.match(image)
        if match:
            month = match.group(2)
            grouped_images[month].append(image)
            # if month in selected_months:
            #     grouped_images[month].append(image)
    return grouped_images

    

if __name__ == "__main__":
    image_folder = '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/month_mean2'
    output_dir = os.path.join(image_folder,'mosaic')
    shp_path = r'/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/research/SHP/weining.shp'
    imageList = GetList(image_folder)
    grouped_images = group_images_by_month(imageList)
    # import ipdb;ipdb.set_trace()
    for month, images in grouped_images.items():
        output_folder = os.path.join(image_folder,'vrt')
        vrt_file_path = os.path.join(output_folder, f'weining_{month}.vrt')
        create_vrt(images,vrt_file_path)
        
    vrt_files_path = GetList(os.path.join(image_folder,'vrt'),'vrt')
    clip_vrt_with_shp(shp_path, vrt_files_path, output_dir)

    # vrt_files = [f'/nfs/project/netdisk/192.168.100.197/d/privite/liyx/dataset/henan/henan_s2_2023/henan_2023_vrts/henan_2023_{i:02d}.vrt' for i in range(1, 13)]
    # output_dir = "/nfs/project/netdisk/192.168.100.197/d/privite/liyx/dataset/henan/s2_clip_resampling_12"
    # os.makedirs(output_dir, exist_ok=True)
    # clip_and_stack_images(shp_path, vrt_files, output_dir)