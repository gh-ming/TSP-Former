from osgeo import gdal
from osgeo import ogr
from osgeo import gdalconst
import os
import sys
from tqdm import tqdm
import shutil
import yimage

def generate_baselist(file_path, suffix):
    suffix_length = len(suffix)
    basename_list = []
    listfile = os.listdir(file_path)
    listfile.sort()
    for basename in listfile:
        if basename[(-suffix_length):] != suffix:
            continue
        basename_list.append(basename)

    return basename_list

def generate_list(file_path, basename_list, xx):
    filename_list = []
    for basename in basename_list:
        filename = file_path + '/' + basename[:-4]+xx
        filename_list.append(filename)

    return filename_list

def check_path(pathname):
    if not os.path.exists(pathname):
        os.makedirs(pathname)
        print(pathname + ' has been created!')

def generate_list_by_filelist(filelist):
    filename_list = []
    listfile = open(filelist)
    for line in listfile.readlines():
        filename_list.append(line.rstrip('\n'))
    
    return filename_list

def check_path(pathname, reset=False):
    if not os.path.exists(pathname):
        os.makedirs(pathname)
        print(pathname + ' has been created!')
    else:
        if reset:
            shutil.rmtree(pathname)
            os.makedirs(pathname)
            print(pathname + ' has been reset!')

def generate_list_by_filelist(filelist):
    filename_list = []
    listfile = open(filelist)
    for line in listfile.readlines():
        filename_list.append(line.rstrip('\n'))
    
    return filename_list
 
def Rasterize(input_shp,input_tif,output_tif,field,filed_type=gdal.GDT_Int32,NoValue=0,switch=0):
    """
    input_shp:需要转为栅格的矢量文件（矢量文件路径）
    input_tif:模板栅格，用于读取地理变换信息、栅格大小，将其应用于新的栅格上
    output_tif:输出栅格文件（栅格文件路径）
    field:字符串，栅格值的字段
    filed_type:栅格值类型，一般选择gdal.GDT_Int16,gdal.GDT_Int32,gdal.GDT_Float32,gdal.GDT_Float64等几种类型
    NoValue:整型或浮点型，矢量空白区转换后的值
    """
    data = gdal.Open(input_tif, gdalconst.GA_ReadOnly)
    # img = yimage.io.read_image(input_tif)
    geo_transform = data.GetGeoTransform()
    proj=data.GetProjection()
    ct = gdal.ColorTable()
    
    img, geo = yimage.io.read_image(input_tif, only_image_info=True, with_geo_info=True)
    width = img['width']
    height = img['height']
    color_table = [(0,0,0),(197,0,255),(0,122,255),(85,255,0),(255,0,0)]
    # color_table = [(255,255,255),(255,144,40)]
    for i, color in enumerate(color_table):
        ct.SetColorEntry(i, color)
    # import ipdb;ipdb.set_trace()
    open_shp = ogr.Open(input_shp)

    shp_ly = open_shp.GetLayer()
    spatial_ref = shp_ly.GetSpatialRef()

    x_min, x_max, y_min, y_max = shp_ly.GetExtent()
    # import ipdb;ipdb.set_trace()
    pixel_size1 = geo_transform[1]
    pixel_size2 = geo_transform[5]
    x_res = int((x_max - x_min) / pixel_size1)
    y_res = int((y_max - y_min) / (-pixel_size2))
    # import ipdb;ipdb.set_trace()
    if x_res <= 0 or y_res <= 0:
        return None
    if switch == 0:
        target_ds = gdal.GetDriverByName('GTiff').Create(output_tif, x_res, y_res, 1, filed_type,options=['COMPRESS=LZW'], )
        target_ds.SetGeoTransform((x_min, pixel_size1, 0.0, y_max, 0.0, pixel_size2))
        target_ds.SetProjection(spatial_ref.ExportToWkt())
    else:
        target_ds = gdal.GetDriverByName('GTiff').Create(output_tif, width, height, 1, filed_type,options=['COMPRESS=LZW'], )
        target_ds.SetGeoTransform((geo['coord'][0], pixel_size1, 0.0, geo['coord'][3], 0.0, pixel_size2))
        target_ds.SetProjection(proj)
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoValue)
    band.FlushCache()
    
    if field is None:
        gdal.RasterizeLayer(target_ds, [1], shp_ly, None)
        print(gdal.RasterizeLayer(target_ds, [1], shp_ly, None))
    else:
        OPTIONS = ['ATTRIBUTE=' + field]
        print(OPTIONS)
        gdal.RasterizeLayer(target_ds, [1], shp_ly, options=OPTIONS)
    _band = target_ds.GetRasterBand(1)
    data = _band.ReadAsArray()
    # data[data > 0] = 255
    _band.WriteArray(data)
    # _band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    # _band.SetColorTable(ct)
    # dataset = gdal.Warp(output_tif.split('.')[0]+'_wgs84.tif', target_ds, dstSRS='EPSG:4326',resampleAlg = gdal.GRIORA_Bilinear,dstNodata = 0,multithread = True,warpOptions = ['NUM_THREADS=ALL_CPUS'],creationOptions = ['BIGTIFF=YES'])
    target_ds.BuildOverviews('nearest',[2,4,8,16,32,64,128])  
    del target_ds

def line2shp(raster_filename, shapefile_filename, pred_band=1):
    raster_dataset = gdal.Open(raster_filename)
    if raster_dataset is None:
        print('[FATAL] GDAL open file failed. [%s]'%raster_filename)
        exit(1)

    driver = ogr.GetDriverByName('ESRI Shapefile')
    if driver is None:
        print('[FATAL] OGR create driver failed. [%s]'%'ESRI Shapefile')
        exit(1)

    shape_dataset = driver.CreateDataSource(shapefile_filename)
    if shape_dataset is None:
        print('[FATAL] OGR create file failed. [%s]'%shapefile_filename)
        exit(1)

    proj_ref = raster_dataset.GetProjectionRef()
    proj_shp = osr.SpatialReference()
    proj_shp.ImportFromWkt(proj_ref)
    layer = shape_dataset.CreateLayer('pred', proj_shp, ogr.wkbPolygon)
    field_name = ogr.FieldDefn('objects', ogr.OFTInteger)
    layer.CreateField(field_name)
    band = raster_dataset.GetRasterBand(pred_band)
    gdal.Polygonize(band, band, layer, 0)
    del shape_dataset


def run():
    # shapefile_path, tiffile_path, out_path = process_arguments(sys.argv)
    # Rasterize(shapefile_path, tiffile_path, out_path, 'type',switch=1)

    # input_dir = "/nfs/project/netdisk/192.168.100.193/d/private/gaohm/tabacco/segmentation_label/label_shp"
    # output_dir = "/nfs/project/netdisk/192.168.100.193/d/private/gaohm/tabacco/segmentation_label/label_box"
    # os.makedirs(output_dir, exist_ok=True)
    # # tiffile_path = '/nfs/project/netdisk/100/workspace/segmentation/project/Saudi_Arabia/GF02_reg_Saudi_Arabia_tif_pick1/GF02_PA1_036453_20210517_MY150_01_084_L1A_01_Reg.tif'
    # tif_dir = "/nfs/project/netdisk/192.168.100.193/d/private/gaohm/tabacco/segmentation_label/image_box"
    # shp_names = [shp for shp in os.listdir(input_dir) if shp.endswith('.shp')]
    # for shp_name in shp_names:
    #     # import ipdb; ipdb.set_trace()
    #     print(shp_name)
    #     shp_file = os.path.join(input_dir, shp_name)
    #     tiffile_path = os.path.join(tif_dir, shp_name.replace('.shp', '.tif'))
    #     output_file = os.path.join(output_dir, shp_name.replace('.shp', '.tif'))
    #     Rasterize(shp_file, tiffile_path, output_file, 'type', switch=1)

    shp_path = "/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/box/XC_box2_GT.shp"
    tiff_path = "/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/predict/RF_XC_box2_20250414_2222.tif"
    output_file = "/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/predict/XC_box2_GT.tif"
    Rasterize(shp_path, tiff_path, output_file, 'type', switch=1)

    # check_path(out_path)
    # # base_list = ['T_1670_810_6681_3242.shp','T_1670_810_6681_3243.shp','T_1670_811_6681_3244.shp','T_1670_811_6681_3245.shp']
    # base_list1 = generate_baselist(out_path, '.tif')
    
    # base_list = generate_baselist(shapefile_path, '.shp')

    
    # # if os.path.isfile(tiffile_path):
    # #     shapefile_list = generate_list_by_filelist(tiffile_path)
    # #     base_list = []
    # #     for filename in tiffile_list:
    # #         base_name = filename.split('/')[-1]
    # #         base_name = base_name[:-4]+'.tif'
    # #         base_list.append(base_name)
    # # else:
    # #     base_list = generate_baselist(tiffile_path, '.tif')
    # #     tiffile_list = generate_list(tiffile_path, base_list, '.tif')
    # # print(base_list1)
    # base_list_2 = []
    # for basename in base_list:
    #     if basename[:-4]+'.tif' in base_list1:
    #         continue
    #     base_list_2.append(basename)
    # print(base_list)
    # shapefile_list = generate_list(shapefile_path, base_list, '.shp')
    # tiffile_list = [tiffile_path]
    # # # shapefile_list = [shapefile_path]


    # # # tiffile_list = generate_list(tiffile_path, base_list_2, '.tif')
    # for i in tqdm(range(len(shapefile_list))):
    # #     # if base_list[i] not in base_list1:
    # #     #     continue
    # #     # outtif_name = out_path + base_list1[i] + '.tif'
    #     outtif_name = out_path + shapefile_list[i].split('/')[-1][:-4]+'.tif'
        #     if os.path.isfile(outtif_name):
        #         continue
        #     print(shapefile_list[i], tiffile_list[i])
            # if len(base_list1[i]) != len(base_list1[0]):
            #     Rasterize(shapefile_list[i], tiffile_path+base_list1[i][:19]+'.tif', outtif_name, 'type')
            # else:
            # if '(' in tiffile_list[i]:

            #     Rasterize(shapefile_list[i], tiffile_list[i].split(' (')[0]+'.tif.tif', outtif_name, 'type')
            # else:
            #     Rasterize(shapefile_list[i], tiffile_list[i]+'.tif', outtif_name, 'type')
        # print(outtif_name)
    




def process_arguments(argv):
    # if len(argv) < 4:
    #     help()
    # shapefile_path = argv[1]
    # tiffile_path = argv[2]
    # out_path = argv[3]

    # shapefile_path = '/nfs/project/netdisk/192.168.100.192/d/private/tianxy/guizhou/test/dn_test_all_v3/infer2/'
    # tiffile_path = '/nfs/project/netdisk/192.168.100.192/d/private/tianxy/guizhou/select_gf/S_813_427_3253_1711.tif'
    # # tiffile_path = '/nfs/project/netdisk/192.168.10.225/d/project/henan_wheat_2023/image_20230326/GF2GF7_dehazed_reg_mosaic/'
    # out_path = '/nfs/project/netdisk/192.168.100.192/d/private/tianxy/guizhou/test/dn_test_all_v3/infer2/tif/'
    shapefile_path = '/nfs/project/netdisk/192.168.100.192/d/project/guizhou_agriculture/20240530/数据/raster/label.shp'
    tiffile_path = '/nfs/project/netdisk/100/workspace/segmentation/project/Saudi_Arabia/GF02_reg_Saudi_Arabia_tif_pick1/GF02_PA1_036453_20210517_MY150_01_084_L1A_01_Reg.tif'
    # tiffile_path = '/nfs/project/netdisk/192.168.10.225/d/project/henan_wheat_2023/image_20230326/GF2GF7_dehazed_reg_mosaic/'
    out_path = '/nfs/project/netdisk/192.168.100.192/d/project/guizhou_agriculture/20240530/数据/raster/label.tif'
    # import ipdb; ipdb.set_trace()
    # check_path(out_path)


    # print(tiffile_path)
    return shapefile_path, tiffile_path, out_path


if __name__ == '__main__':
    run()


