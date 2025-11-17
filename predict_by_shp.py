"""
shp_predict.py - 基于shp范围的时序影像预测
"""
import numpy as np
import torch
import rasterio
from pathlib import Path
from tqdm import tqdm
from models import *
from sklearn.ensemble import RandomForestClassifier
from datasets.uscrops import TABACCO_Crops
from utils import get_model
from datetime import datetime
from rasterio.windows import Window
from typing import List, Dict
class TabaccoPredictor:
    def __init__(self, cfg):
        """
        Args:
            cfg: 配置字典
                model_path: 模型路径
                data_root: 包含时序TIFF和TSP数据的目录
                tsp_path: TSP数据文件路径
                output_dir: 输出目录
                patch_size: 分块尺寸（奇数）
                ndims: 光谱通道数（不含TSP/DOY）
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_paths()
        self._load_georeference()
        self._init_model()
        self._load_tsp_data()
        self.metadata = self._get_raster_metadata()

    def _init_paths(self):
        self.output_dir = Path(self.cfg['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_georeference(self):
        """从第一个TIFF文件获取地理参考"""
        sample_tif = next(Path(self.cfg['data_root']).glob('*.tif'))
        with rasterio.open(sample_tif) as src:
            self.transform = src.transform
            self.crs = src.crs
            self.height = src.height
            self.width = src.width

    def _load_tsp_data(self):
        """加载全局TSP数据"""
        self.tsp_src = rasterio.open(self.cfg['tsp_path'])
        assert self.tsp_src.count == 1, "TSP数据应为单波段"
        
    def _get_raster_metadata(self) -> List[Dict]:
        """解析时序TIFF元数据"""
        metadata = []
        for f in Path(self.cfg['data_root']).glob('*.tif'):
            # if not f.name.startswith('Rup'):
            #     continue
                
            # 解析日期（示例文件名：'Rup_20230415.tif'）
            date_str = f.stem.split('_')[1]
            try:
                date = datetime.strptime(date_str, "%Y%m%d")
                doy = date.timetuple().tm_yday  # 年积日
            except:
                continue
                
            metadata.append({
                'path': str(f),
                'doy': doy,
                'date_str': date_str
            })
        return sorted(metadata, key=lambda x: x['date_str'])

    def _get_valid_window(self, src: rasterio.DatasetReader, x: int, y: int, size: int) -> Window:
        """计算有效数据窗口"""
        return Window(
            x_off=x,
            y_off=y,
            width=min(size, src.width - x),
            height=min(size, src.height - y)
        )

    def _load_temporal_patch(self, x: int, y: int, size: int) -> np.ndarray:
        """加载时空数据块 [T, C+2, H, W]"""
        temporal_data = []
        
        for meta in tqdm(self.metadata, desc="加载时序数据", leave=False):
            # 加载光谱数据
            with rasterio.open(meta['path']) as src:
                window = self._get_valid_window(src, x, y, size)
                spectral = src.read(window=window)  # [C, H, W]
                
            # 加载TSP
            tsp_window = self._get_valid_window(self.tsp_src, x, y, size)
            tsp = self.tsp_src.read(1, window=tsp_window)  # [H, W]
            
            # 对齐检查
            if tsp.shape != spectral.shape[1:]:
                tsp = np.zeros(spectral.shape[1:], dtype=np.float32)
                
            # 生成DOY矩阵
            doy = np.full(spectral.shape[1:], meta['doy'], dtype=np.int16)
            
            # 拼接特征 [C+2, H, W]
            combined = np.concatenate([
                spectral, 
                tsp[np.newaxis], 
                doy[np.newaxis]
            ], axis=0)
            
            temporal_data.append(combined)
            
        return np.stack(temporal_data, axis=0)  # [T, C+2, H, W]

    def _preprocess(self, patch: np.ndarray) -> torch.Tensor:
        """复用TABACCO_Crops的内部方法进行预处理"""
        # 动态创建虚拟数据集实例
        dummy_data = np.expand_dims(patch, axis=0)  # [1, T, C+2, H, W]
        
        # 初始化与训练配置一致的dataset实例
        crop_dataset = TABACCO_Crops(
            data=dummy_data,
            labels=None,
            mode=self.cfg['mode'],
            patch_size=self.cfg['patch_size'],
            datapath=Path(self.cfg['data_root']),
            scale_factor=1e-4,
            ndims=self.cfg['ndims'],
            use_cache=False
        )
        
        # 复用特征提取流程
        spectral, tsp, doy = crop_dataset._extract_features(dummy_data[0])  # [T,C,H,W], [T,H,W], [T,H,W]
        
        # 复用特征处理流程
        processed = crop_dataset._process_patch(spectral, tsp, doy)
        
        # 重组为模型输入格式
        if self.cfg['mode'] == 'center':
            # [T, C] -> [1, T, C] (添加batch维度)
            return (processed[0].unsqueeze(0), 
                    processed[1].unsqueeze(0),
                    processed[2].unsqueeze(0),
                    processed[3].unsqueeze(0))
        else:
            # [T, C, H, W] -> [1, T, C, H, W]
            return (processed[0].unsqueeze(0),
                    processed[1].unsqueeze(0),
                    processed[2].unsqueeze(0),
                    processed[3].unsqueeze(0))


    def _predict_patch(self, x: int, y: int, size: int) -> np.ndarray:
        """预测单个空间分块"""
        # 1. 加载时空数据
        patch_data = self._load_temporal_patch(x, y, size)  # [T, C+2, H, W]
        
        # 2. 数据预处理
        inputs = self._preprocess(patch_data).to(self.device)
        
        # 3. 模型推理
        with torch.no_grad():
            outputs = self.model(inputs.unsqueeze(0))  # 增加batch维度
            pred = outputs.argmax(1).cpu().numpy()[0]
            
        return pred

    def run(self):
        """端到端预测流程"""
        size = self.cfg['patch_size']
        pred_map = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # 分块预测
        for y in tqdm(range(0, self.height, size), desc="垂直分块"):
            for x in tqdm(range(0, self.width, size), desc="水平分块", leave=False):
                # 预测当前分块
                patch_pred = self._predict_patch(x, y, size)
                
                # 获取实际分块尺寸
                h, w = patch_pred.shape
                
                # 写入预测图
                pred_map[y:y+h, x:x+w] = patch_pred
                
        # 保存结果
        self._save_geotiff(pred_map)

    def _save_geotiff(self, data: np.ndarray):
        """保存地理参考结果"""
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': 0,
            'count': 1,
            'height': self.height,
            'width': self.width,
            'crs': self.crs,
            'transform': self.transform,
            'compress': 'lzw'
        }
        
        with rasterio.open(self.output_dir/'prediction.tif', 'w', **profile) as dst:
            dst.write(data, 1)

if __name__ == '__main__':
    config = {
        'model_path': "/nfs/project/netdisk/192.168.100.192/d/private/gaohm/SITS_MoCo/results/G_TSP_TransNet_R2_Seed1_512_0.003_7_0.1_P/model_best.pth",
        'model_type': 'TSP',  
        'data_root': '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/month_mean2/mosaic',
        'output_dir': '/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/predict',
        'patch_size': 7,
        'mode': 'patch',
        'ndims': 10,
        'batch_size': 32
    }
    
    predictor = TabaccoPredictor(config)
    predictor.run()


