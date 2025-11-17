import rasterio
import numpy as np

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

if __name__ == "__main__":
    gt_path = '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/predict/XC_box2_GT.tif'
    pred_path = '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/predict/XC_TSP_Former3.tif'
    output_path = '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/predict/XC_TSP_Former3_Dif.tif'
    
    visualize_differences(gt_path, pred_path, output_path)
