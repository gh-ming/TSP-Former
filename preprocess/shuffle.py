import geopandas as gpd
import pandas as pd

def shuffle_and_export_shp(input_shp, output_shp, fraction, columns_to_keep):
    # 读取shp文件
    gdf = gpd.read_file(input_shp)

    # 打乱条目
    gdf = gdf.sample(frac=1)

    # 根据比例或者绝对数量选择条目
    if fraction < 1:
        gdf = gdf.sample(frac=fraction)
    else:
        gdf = gdf.sample(n=int(fraction))

    # # 计算每个'type'值的频率
    # gdf['type_count'] = gdf.groupby('XBZW')['XBZW'].transform('count')

    # # 根据'type'的频率进行排序，然后为排序后的'type'分配整数ID
    # gdf = gdf.sort_values('type_count', ascending=False)
    # gdf['type_id'], _ = pd.factorize(gdf['XBZW'])

    # # 打印'type'和对应的'type_id'
    # type_id_df = gdf[['XBZW', 'type_id']].drop_duplicates()
    # print(type_id_df)

    # 保留部分属性表的值
    if 'geometry' not in columns_to_keep:
        columns_to_keep.append('geometry')
    gdf = gdf[columns_to_keep]
    print(gdf.head())

    # 导出成新的shp文件
    gdf.to_file(output_shp, encoding='utf-8')

# 使用示例
input_shp = '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/sample/xuanwei/xuanwei_points_0317.shp'
output_shp = '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/sample/xuanwei/xuanwei_points_0317_500.shp'
fraction = 0.5  # 选择50%的条目
# columns_to_keep = ['type','location']  # 保留的属性表的列
columns_to_keep = ['code']  # 保留的属性表的列
shuffle_and_export_shp(input_shp, output_shp, fraction, columns_to_keep)