import os
import csv
from datetime import datetime
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

        date_str = parts[2][:8]
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        doys = (date_obj - start_date).days
        tile_id = parts[5][-3:]
        if tile_id == 'RUQ':
            metadata.append({
                'doy': doys
            })
    output_csv = os.path.join(raster_folder, 'raster_doy.csv')
    if not os.path.exists(output_csv):
        save_metadata_to_csv(metadata, output_csv)
raster_folder = "/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2"
get_raster_metadata(raster_folder)