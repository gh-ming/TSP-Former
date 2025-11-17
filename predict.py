"""
This script is for time series classification task.
"""
import copy
import argparse
from tqdm import tqdm
from joblib import dump, load
from datetime import datetime
import torch
import torch.optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import *
import rasterio
from rasterio.transform import from_origin
from osgeo import gdal, ogr
import numpy as np
import os   
import pandas as pd
import geopandas as gpd

def visualize_differences(gt_path, pred_path, output_path):
    """
    将GT和预测的TP、FP、FN用不同颜色表示，并保存为TIFF文件。

    :param gt_path: Ground truth TIFF 文件路径
    :param pred_path: 预测 TIFF 文件路径
    :param output_path: 输出文件路径
    """
    with rasterio.open(gt_path) as gt_dataset, rasterio.open(pred_path) as pred_dataset:
        gt = gt_dataset.read(1)
        pred = pred_dataset.read(1)
        
        output = np.zeros((3, gt.shape[0], gt.shape[1]), dtype=np.uint8)

        # 定义颜色 (R, G, B)
        TP_color = (0, 255, 0)  # 绿色
        FP_color = (255, 0, 0)  # 红色
        FN_color = (0, 0, 255)  # 蓝色

        TP = (gt == 1) & (pred == 1)
        FP = (gt == 0) & (pred == 1)
        FN = (gt == 1) & (pred == 0)

        output[0][TP] = TP_color[0]
        output[1][TP] = TP_color[1]
        output[2][TP] = TP_color[2]

        output[0][FP] = FP_color[0]
        output[1][FP] = FP_color[1]
        output[2][FP] = FP_color[2]

        output[0][FN] = FN_color[0]
        output[1][FN] = FN_color[1]
        output[2][FN] = FN_color[2]

        with rasterio.open(output_path, 'w', driver='GTiff', 
                           width=gt_dataset.width, height=gt_dataset.height,
                           count=3, dtype='uint8', crs=gt_dataset.crs,
                           transform=gt_dataset.transform) as output_dataset:
            output_dataset.write(output[0], 1)
            output_dataset.write(output[1], 2)
            output_dataset.write(output[2], 3)

    print(f"生成彩色输出完成，保存为 {output_path}")

def matrix_to_tiff(shp_path, matrix, output_tif):
    print("Writing array to Tiff...")
    # 读取SHP文件的范围和坐标系
    shp_ds = ogr.Open(shp_path)
    layer = shp_ds.GetLayer()

    feature = layer.GetNextFeature()
    
    geom = feature.GetGeometryRef()
    extent = geom.GetEnvelope()  # (minX, maxX, minY, maxY)
    srs = layer.GetSpatialRef()
    srs_wkt = srs.ExportToWkt()

    # 矩阵的行列数
    rows, cols = matrix.shape

    # 计算每个像元的分辨率
    x_res = (extent[1] - extent[0]) / cols
    y_res = (extent[2] - extent[3]) / rows  # 结果为负数，因为Y向下减少

    # 创建TIFF文件

    driver = gdal.GetDriverByName('GTiff')
    tiff_ds = driver.Create(output_tif, cols, rows, 1, gdal.GDT_Byte)
    tiff_ds.SetGeoTransform((extent[0], x_res, 0, extent[3], 0, y_res))
    tiff_ds.SetProjection(srs_wkt)

    # 将矩阵写入波段（假设矩阵行顺序与栅格一致，即第一行对应最北端）
    band = tiff_ds.GetRasterBand(1)
    band.WriteArray(matrix)
    band.FlushCache()

    # 关闭数据集
    tiff_ds = None
    shp_ds = None



def extract_patches(
    x: torch.Tensor,
    patch_size: int,
    padding_mode: str = 'constant',
    padding_value: float = 0
) -> torch.Tensor:
    """
    对四维张量进行基于中心像素的补丁提取
    
    参数：
    x : Tensor 
        输入张量 [T, C, H, W]
    patch_size : int 
        补丁尺寸（支持奇偶）
    padding_mode : str 
        填充方式（'constant', 'reflect'等）
    padding_value : float
        常数填充时的填充值
        
    返回：
    Tensor [T, C, H, W, K, K]
    """
    x = torch.from_numpy(x)
    T, C, H, W = x.shape
    # 计算所需填充量
    pad = patch_size // 2
    padding = (pad, pad, pad, pad)  # 左右上下对称填充
    
    # 执行填充
    if padding_mode == 'constant':
        x_padded = F.pad(x, padding, mode=padding_mode, value=padding_value)
    else:
        x_padded = F.pad(x, padding, mode=padding_mode)
    
    # 在高度维度展开
    x_unfold_h = x_padded.unfold(2, patch_size, 1)  # [T, C, H, W, K]
    
    # 在宽度维度展开
    x_unfold_hw = x_unfold_h.unfold(3, patch_size, 1)  # [T, C, H, W, K, K]

    patches = x_unfold_hw.permute(2, 3, 0, 1, 4, 5).reshape(H*W, T, C, patch_size, patch_size)
    patches = patches.numpy()
    
    return patches


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal prediction with different models")
    parser.add_argument('--datapath', type=str, default='/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/box', help='Directory containing temporal TIFF files')
    parser.add_argument('--shp_path', type=str, default="/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/box/XC_box2.shp",help='help to make tiff.')
    parser.add_argument('--model', type=str, default="STNET",help='select model architecture.')
    parser.add_argument('--model_path', type=str, default = "/nfs/project/netdisk/192.168.100.192/d/private/gaohm/SITS_MoCo/results/STNet_C2_512_0.001_center_10_2", help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default = "/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/predict", help='Output directory for predictions')
    parser.add_argument('-m', '--mode', type=str, default="center",help=' mode in center or patch')   
    parser.add_argument('--ndims', type=int, default=10, help='number of input channel dimensions')
    parser.add_argument('-p', '--patch_size', type=int, default=3,help='patch_size (number of time series patched )') 
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--batchsize', type=int, default= 512, help='Prediction batch size')
    return parser.parse_args()

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datapath = args.datapath
    shp_path = args.shp_path
    model_path = args.model_path
    mode = args.mode
    ndims = args.ndims
    patch_size = args.patch_size
    num_classes = args.num_classes
    base_name = os.path.basename(shp_path)[:-4]
    data_path = os.path.join(datapath,f'{base_name}.npy')
    data = np.load(data_path, allow_pickle=True)
    time = datetime.now().strftime("%Y%m%d_%H%M")
    tif_path = f"/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/predict/{args.model}_{base_name}_{time}.tif"
    
    T,C,H,W = data.shape
    print('data shape:',data.shape)
    patch_path = f"/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/predict/{base_name}_{patch_size}.npy"
    if os.path.exists(patch_path):
        patches = np.load(patch_path,allow_pickle=True)
    else:
        patches = extract_patches(data,patch_size)
        np.save(patch_path,patches)
    # pixel_data = data.transpose(2, 3, 0, 1)  # [h, w, T, C]
    # N = pixel_data.shape[0] * pixel_data.shape[1]
    # pixel_data = pixel_data.reshape(N, -1, data.shape[1])  # [N, T, C]
    # print('pixel data shape:',pixel_data.shape)

    # 统一参数配置
    common_args = {
        'mode': mode,
        'patch_size': patch_size,
        'datapath': datapath,
        'scale_factor': 1e-4,
        'ndims': ndims,
        'use_cache': True
    }
    if args.mode == 'center':
        if args.model in ['rf', 'RF']:
            best_model_path = os.path.join(model_path , 'model_best.joblib')
            model = load(best_model_path)
            features = []
            dataset = TABACCO_Crops(data=patches, **common_args)
            for x, _ in dataset:
                spectral = x[0].numpy().reshape(-1) #[T,C] -> [-1]
                features.append(spectral)
            rf_data = np.stack(features)
            y_pred = model.predict(rf_data)
            y_pred = np.array(y_pred).reshape(H, W).astype(np.uint8)
        else:
            model = get_model(args.model, ndims, num_classes, device,args.patch_size)
            best_model_path = os.path.join(model_path, 'model_best.pth')
            print('Restoring best model weights for testing...')
            checkpoint = torch.load(best_model_path)
            state_dict = {k: v for k, v in checkpoint['model_state'].items()}
            criterion = checkpoint['criterion']
            model.load_state_dict(state_dict)
            model.eval()
            print("=> creating model '{}'".format(args.model))
            dataset = TABACCO_Crops(data=patches, **common_args)
            # import ipdb;ipdb.set_trace()
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize * 10, shuffle=False)
            with torch.no_grad():
                y_pred = []
                with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
                    for idx, (X, y) in iterator:
                        X = recursive_todevice(X, device)
                        logits = model(X)
                        # import ipdb;ipdb.set_trace()
                        out = F.log_softmax(logits, dim=-1)
                        y_pred.append(out.argmax(-1).cpu().numpy())
                y_pred = np.concatenate(y_pred, axis=0)
                y_pred = y_pred.reshape(H, W).astype(np.uint8)
    else:
        model = get_model(args.model, ndims, num_classes, device,args.patch_size)
        best_model_path = os.path.join(model_path, 'model_best.pth')
        print('Restoring best model weights for testing...')
        checkpoint = torch.load(best_model_path)
        state_dict = {k: v for k, v in checkpoint['model_state'].items()}
        criterion = checkpoint['criterion']
        model.load_state_dict(state_dict)
        model.eval()
        print("=> creating model '{}'".format(args.model))
        dataset = TABACCO_Crops(data=patches, **common_args)
        # import ipdb;ipdb.set_trace()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize * 10, shuffle=False)
        with torch.no_grad():
            y_pred = []
            with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
                for idx, (X, y) in iterator:
                    main_input = X[0].to(device)
                    doy = X[1].to(device)
                    ndvi_gt = X[2].to(device)
                    y = y.to(device)
                    # logits, ndvi_pred, tsp_pred = model(main_input, doy)  
                    tsp_gt = X[3][:,0,:,:].to(device)
                    logits = model(main_input, doy,tsp_gt,ndvi_gt) 
                    out = F.log_softmax(logits, dim=-1)
                    y_pred.append(out.argmax(-1).cpu().numpy())
            y_pred = np.concatenate(y_pred, axis=0)
            y_pred = y_pred.reshape(H, W).astype(np.uint8)
    matrix_to_tiff(shp_path, y_pred, tif_path)
    gt_path = '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/predict/XC_box2_GT.tif'
    output_path =  f"/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/predict/{args.model}_{base_name}_{time}_Dif.tif"
    visualize_differences(gt_path, tif_path, output_path)
    





def main():
    args = parse_args()
    predict(args)

if __name__ == '__main__':
    main()
