import numpy as np
import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
from typing import Optional, Tuple, Union

class TABACCO_Crops(Dataset):
    def __init__(self,
                 data: np.ndarray,
                 labels: Optional[np.ndarray] = None,
                 mode: str = 'center',
                 patch_size: int = 7,  # 默认patch尺寸
                 datapath: Optional[Union[str, Path]] = None,
                 scale_factor: float = 1e-4,
                 ndims: int = 10,
                 use_cache: bool = True):
        super().__init__()
        
        # 参数验证
        assert mode in ['center', 'patch'], "模式应为center或patch"
        assert patch_size % 2 == 1, "patch尺寸必须为奇数"
        
        self.data = data
        self.labels = labels
        self.mode = mode
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.scale = scale_factor
        self.datapath = Path(datapath) if datapath else None
        self.original_channels = ndims
        self.cache = {} if use_cache else None
        
        # 空间维度
        self.H, self.W = data.shape[-2:]
        self.center_h = self.H // 2
        self.center_w = self.W // 2
        
        self.mean , self.std = self._compute_stats(mode)
        # 边界检查
        if mode == 'patch':
            assert self.H >= patch_size, f"高度{self.H}小于patch尺寸{patch_size}"
            assert self.W >= patch_size, f"宽度{self.W}小于patch尺寸{patch_size}"
        
    def _compute_stats(self, mode: str) -> Tuple[np.ndarray, np.ndarray]:
        """计算光谱通道的均值和标准差"""
        stats_file = self.datapath / f'spectral_stats_{mode}.npz' if self.datapath else None
        
        if stats_file and stats_file.exists():
            stats = np.load(stats_file)
            return stats['mean'], stats['std']
            
        # 根据模式选择数据范围
        # self.data [N,T,C,H,W]
        if mode == 'center':
            spectral_data = self.data[..., :self.original_channels, 
                                    self.center_h, self.center_w]
            # 计算统计量 [T,C]
            mean = spectral_data.mean(axis=(0)) * self.scale # [T,C]
            std = spectral_data.std(axis=(0)) * self.scale   # [T,C]
        else:  # 'patch'模式
            spectral_data = self.data[..., :self.original_channels, :, :]
            # 计算统计量 [T,C]
            mean = spectral_data.mean(axis=(0,3,4)) * self.scale # [T,C]
            std = spectral_data.std(axis=(0,3,4)) * self.scale

        if stats_file:
            np.savez(stats_file, mean=mean, std=std)
            
        return mean, std

    def _extract_features(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """分解输入数据的三个部分"""
        # x形状: [T, C+2, H, W]
        spectral = x[:, :self.original_channels] * self.scale  # 光谱数据 [T,C,H,W]
        tsp = x[:, self.original_channels]+ 1e-8     # 物候特征 [T,H,W]
        doy = x[:, self.original_channels+1]* 30     # 时间 [T,H,W] 
        return spectral, tsp, doy

    def _process_patch(self, spectral: np.ndarray, tsp: np.ndarray,doy) -> np.ndarray:
        """处理时空特征"""
        # 标准化光谱数据
        if self.mode == 'center':
            spectral = (spectral - self.mean) / (self.std + 1e-8)
        else:
            spectral = (spectral - self.mean[:, :, None, None]) / (self.std[:, :, None, None] + 1e-8)
        # 计算时空权重
        ndvi = self._compute_ndvi(spectral)  # [T,H,W] or [T]
        # 转换为Tensor
        spectral_tensor = torch.from_numpy(spectral).float()
        ndvi_tensor = torch.from_numpy(ndvi).float()
        tsp_tensor = torch.from_numpy(tsp).float()
        doy_tensor = torch.from_numpy(doy).long()
        
        return spectral_tensor, doy_tensor, ndvi_tensor, tsp_tensor

    def _compute_ndvi(self, spectral: np.ndarray) -> np.ndarray:
        """计算NDVI指数"""
        # 假设第3通道是红波段（索引2），第7通道是近红外（索引6）
        if self.mode == 'center':
            red = spectral[:, 2]
            nir = spectral[:, 6]
            ndvi = (nir - red) / (nir + red + 1e-8)
        elif self.mode == 'patch':
            red = spectral[:, 2,:,:]
            nir = spectral[:, 6,:,:]
            ndvi = (nir - red) / (nir + red + 1e-8)
        return np.clip(ndvi, -1, 1)

    def __getitem__(self, index: int):
        if self.cache is not None and index in self.cache:
            return self.cache[index]
            
        x = self.data[index]  # [T, C+2, H, W]
        
        # 分解特征
        spectral, tsp, doy = self._extract_features(x)
        
        # 空间模式选择
        if self.mode == 'center':
            spectral = spectral[..., self.center_h,self.center_w]  # [T,C]
            tsp = tsp[..., self.center_h,self.center_w]            # [T] 
            doy = doy[..., self.center_h,self.center_w]            # [T]
        elif self.mode == 'patch':
            # 截取中心patch
            # import ipdb;ipdb.set_trace()
            h_slice = slice(self.center_h-self.half_patch, self.center_h+self.half_patch+1)
            w_slice = slice(self.center_w-self.half_patch, self.center_w+self.half_patch+1)
            spectral = spectral[..., h_slice, w_slice]  # [T,C,P,P]
            tsp = tsp[..., h_slice, w_slice]            # [T,P,P]
            doy = doy[..., h_slice, w_slice]            # [T,P,P]
            
        # 处理特征
        X = self._process_patch(spectral, tsp, doy)
        
        # 标签处理
        Y = torch.tensor(self.labels[index], dtype=torch.long) if self.labels is not None else -1
        
        # 缓存
        if self.cache is not None:
            self.cache[index] = (X, Y)
            
        return X, Y

    def __len__(self) -> int:
        return len(self.data)

