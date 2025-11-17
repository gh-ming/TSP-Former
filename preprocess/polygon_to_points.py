from osgeo import gdal, ogr
from shapely.geometry import Polygon, Point, MultiPolygon
import random
from tqdm import tqdm

# 定义世界坐标到像素坐标的转换函数
def world_to_pixel(geo_transform, x, y):
    ulX = geo_transform[0]
    ulY = geo_transform[3]
    xDist = geo_transform[1]
    yDist = geo_transform[5]
    rtnX = geo_transform[2]
    rtnY = geo_transform[4]
    pixel = int((x - ulX) / xDist)
    line = int((y - ulY) / yDist)
    return (pixel, line)



tif_file = '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/TSP2/TSP_weining_resampling_0307.tif'
shp_file = '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/sample/polygon/weining_samples_0308.shp'
output_shp_file = '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/sample/polygon/weining_samples_0308_3000.shp'

# 读取tif影像
dataset = gdal.Open(tif_file)
transform = dataset.GetGeoTransform()
spatial_ref = dataset.GetProjection()

# 读取shp文件
driver = ogr.GetDriverByName('ESRI Shapefile')
dataSource = driver.Open(shp_file, 0)
layer = dataSource.GetLayer()

# 创建输出Shapefile
output_driver = ogr.GetDriverByName('ESRI Shapefile')
if output_driver.DeleteDataSource(output_shp_file):
    print(f"Deleted existing output file: {output_shp_file}")
output_data_source = output_driver.CreateDataSource(output_shp_file)
output_layer = output_data_source.CreateLayer('sample_points', geom_type=ogr.wkbPoint, srs=ogr.osr.SpatialReference(spatial_ref))

# 添加字段
# field_defn_id = ogr.FieldDefn('ID', ogr.OFTInteger)
# output_layer.CreateField(field_defn_id)
field_defn_type = ogr.FieldDefn('code', ogr.OFTString)
output_layer.CreateField(field_defn_type)

# 计算总面积
A_total = sum(feature.GetField('MJ') for feature in layer)

# 总采样点数量
N_total = 3000 # 示例值

# 存储所有采样点
sample_points = []

# 对每个矢量面进行采样
for feature in tqdm(layer, desc="Processing features"):
    A_i = feature.GetField('MJ')
    N_i = round((A_i / A_total) * N_total)
    if N_i == 0 :
        continue
    feature_type = feature.GetField('code')
    
    # 获取矢量面几何形状
    geom = feature.GetGeometryRef()
    
    # 处理空洞
    if geom.GetGeometryName() == 'POLYGON':
        exterior_ring = geom.GetGeometryRef(0)
        exterior_polygon = Polygon(exterior_ring.GetPoints())
        
        # 处理空洞，假设空洞是MultiPolygon类型
        holes = MultiPolygon([Polygon(geom.GetGeometryRef(i).GetPoints()) for i in range(1, geom.GetGeometryCount())])
        
        # 使用unary_union来合并所有空洞，然后从外多边形中减去
        exterior_polygon = exterior_polygon.difference(holes)

    for _ in range(N_i):
        # 在矢量面内生成随机点
        minx, miny, maxx, maxy = exterior_polygon.bounds
        while True:
            point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if exterior_polygon.contains(point):
                # 将地理坐标转换为像素坐标
                x, y = world_to_pixel(transform, point.x, point.y)
                sample_points.append((point.x, point.y))
                
                # 创建新的点要素并添加到输出图层
                featureDefn = output_layer.GetLayerDefn()
                outFeature = ogr.Feature(featureDefn)
                # outFeature.SetField('ID', len(sample_points))
                outFeature.SetField('code', feature_type)
                point_geom = ogr.Geometry(ogr.wkbPoint)
                point_geom.AddPoint(point.x, point.y)
                outFeature.SetGeometry(point_geom)
                output_layer.CreateFeature(outFeature)
                outFeature = None
                
                break

# 关闭数据源
dataSource = None
output_data_source = None
dataset = None

print(f"Sampling completed. Output file: {output_shp_file}")
