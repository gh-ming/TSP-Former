import torch
import torch.nn.functional as F
import numpy as np
import glob
import os
from osgeo import gdal
from datetime import datetime
from models import *
from utils import *
from sklearn.ensemble import RandomForestClassifier
import time
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal prediction with different models")
    parser.add_argument('--image_dir', type=str, default='/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/month_mean/mosaic', help='Directory containing temporal TIFF files')
    parser.add_argument('--model', type=str, default="tsp",help='select model architecture.')
    parser.add_argument('--model_path', type=str, default = "/nfs/project/netdisk/192.168.100.193/d/private/gaohm/code/SITS_MoCo/results/G_TSP_TransNet_R2_Seed111_512/model_best.pth", help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default = "/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/predict", help='Output directory for predictions')
    parser.add_argument('--model_type', choices=['pixel', 'patch'], default = 'pixel', help='Model input type')
    parser.add_argument('--patch_size', type=int, default=512, help='Patch size for patch-based model')
    parser.add_argument('-seq', '--sequencelength', type=int, default=12,help='Maximum length of time series data (default 12)')
    parser.add_argument('--stride', type=int, default=256, help='Stride for sliding window')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default= 512, help='Prediction batch size')
    return parser.parse_args()

class ImageProcess:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dataset = gdal.Open(self.filepath, gdal.GA_ReadOnly)
        self.info = []
        self.img_data = None
        self.data_8bit = None

    def read_img_info(self):
        # 获取波段、宽、高
        img_bands = self.dataset.RasterCount
        img_width = self.dataset.RasterXSize
        img_height = self.dataset.RasterYSize
        # 获取仿射矩阵、投影
        img_geotrans = self.dataset.GetGeoTransform()
        img_proj = self.dataset.GetProjection()
        # 获取NoData值
        img_nodata = self.dataset.GetRasterBand(1).GetNoDataValue()
        self.info = [img_bands, img_width, img_height, img_geotrans, img_proj,img_nodata]
        return self.info

    def read_img_data(self):
        self.img_data = self.dataset.ReadAsArray(0, 0, self.info[1], self.info[2])
        return self.img_data

    # 影像写入文件
    @staticmethod
    def write_img(filename: str, img_data: np.array, **kwargs):
        # 判断栅格数据的数据类型
        if 'int8' in img_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        # 判读数组维数
        if len(img_data.shape) >= 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands, (img_height, img_width) = 1, img_data.shape
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        outdataset = driver.Create(filename, img_width, img_height, img_bands, datatype)
        # 写入仿射变换参数
        if 'img_geotrans' in kwargs:
            outdataset.SetGeoTransform(kwargs['img_geotrans'])
        # 写入投影
        if 'img_proj' in kwargs:
            outdataset.SetProjection(kwargs['img_proj'])
        # 写入文件
        if img_bands == 1:
            outdataset.GetRasterBand(1).WriteArray(img_data)  # 写入数组数据
        else:
            for i in range(img_bands):
                outdataset.GetRasterBand(i + 1).WriteArray(img_data[i])

        del outdataset

def read_multi_bands(image_path):
    """
    读取多波段文件
    :param image_path: 多波段文件路径
    :return: 影像对象，影像元信息，影像矩阵
    """
    # 影像读取
    image = ImageProcess(filepath=image_path)
    # 读取影像元信息
    image_info = image.read_img_info()
    # print(f"多波段影像元信息：{image_info}")
    # 读取影像矩阵
    image_data = image.read_img_data()
    return image, image_info, image_data


class TemporalPredictor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tif_files, self.metadata = self.load_temporal_meta()
        self.ndims = self.metadata[0]
        self.model = self.load_model()
        self.start = time.time()
        


    def get_model(self,modelname, ndims, num_classes,device):
        modelname = modelname.lower()  # make case invariant
        if modelname == 'transformer':
            model = TransformerModel(input_dim=ndims, num_classes=num_classes).to(device)
        elif modelname == 'tempcnn':
            model = TempCNN(input_dim=ndims, num_classes=num_classes).to(device)
        elif modelname == 'lstm':
            model = LSTM(input_dim=ndims, num_classes=num_classes).to(device)
        elif modelname == 'ltae':
            model = LTAE(input_dim=ndims, num_classes=num_classes).to(device)
        elif modelname == 'rf':
            model = RandomForestClassifier(n_estimators=500, max_depth=25)
        elif modelname == 'stnet':
            model = STNet(input_dim=ndims, num_classes=num_classes).to(device)
        elif modelname == 'tsp':
            model = TSP_TransNet(input_dim=ndims, num_classes=num_classes).to(device)
        else:
            raise ValueError(
                "invalid model argument. choose from 'Transformer', 'TempCNN', 'LSTM', 'LTAE', 'RF', or 'STNet' ")

        return model
     
    def load_model(self):
        model = self.get_model(args.model, self.ndims, self.args.num_classes, self.device)
        checkpoint = torch.load(self.args.model_path)
        state_dict = {k: v for k, v in checkpoint['model_state'].items()}
        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model

    def parse_time(self, filename,start_date ='2023-01-01'):
        base = os.path.basename(filename)
        if base.startswith('S2'):
            parts = base.split('_')
            date_str = parts[2][:8]
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            date = datetime.strptime(date_str, "%Y%m%d")
            doy = (date - start_date).days
        else:
            date_str = base.split('_')[-1]
            month = int(date_str[:2])
            doy = (month - 1) * 30
            # print(doy)
        return doy

    # def load_temporal_data(self):
    #     tif_files = glob.glob(os.path.join(self.args.image_dir, '*.tif'))
    #     tif_files.sort(key=self.parse_time)
    #     # doy_int = [self.parse_time(f) for f in tif_files]
        
    #     temporal_data = []
    #     meta = None
    #     for f in tif_files:
    #         _, meta, img = read_multi_bands(f)
    #         # 注意修改
    #         if meta[0] == 12:
    #             img = img[[1,2,3,4,5,6,7,8,10,11]]
                
    #         doy = self.parse_time(f)
    #         doy_array = np.full((1, img.shape[1], img.shape[2]), doy, dtype=np.int32)
    #         img = np.concatenate((img, doy_array), axis=0)
    #         temporal_data.append(img)
            
    #     return np.stack(temporal_data, axis=self.ndims), meta
    
    def load_temporal_meta(self):
        tif_files = glob.glob(os.path.join(self.args.image_dir, '*.tif'))
        tif_files.sort(key=self.parse_time)
        _, meta, _ = read_multi_bands(tif_files[0])
        
        return tif_files,meta
        
        

    # def preprocess_pixel(self):
    #     T, C, H, W = self.temporal_data.shape
    #     pixel_data = self.temporal_data.transpose(2, 3, 0, 1)  # [H, W, T, C]
    #     return pixel_data.reshape(-1, T, C)  # [N, T, C]

    def preprocess_patch(self):
        T, C, H, W = self.temporal_data.shape
        pad_h = (self.args.patch_size - H % self.args.patch_size) % self.args.patch_size
        pad_w = (self.args.patch_size - W % self.args.patch_size) % self.args.patch_size
        
        padded = np.pad(self.temporal_data, 
                       ((0,0), (0,0), (0,pad_h), (0,pad_w)),
                       mode='constant',
                       constant_values=self.metadata[-1])
        
        patches = []
        for i in range(0, padded.shape[2], self.args.stride):
            for j in range(0, padded.shape[3], self.args.stride):
                patch = padded[:, :, i:i+self.args.patch_size, j:j+self.args.patch_size]
                if patch.shape[2:] == (self.args.patch_size, self.args.patch_size):
                    patches.append(patch)
        return patches, padded.shape

    def predict_pixel(self,patch=True):
        # 初始化分块参数
        # import ipdb;ipdb.set_trace()
        W, H = self.metadata[1], self.metadata[2]
        block_size = 1024  # 根据GPU显存调整
        predictions = np.zeros((H, W), dtype=np.uint8)
        total_start = time.time()
        tsp_path = os.path.join('/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/weining/2023/S2/TSP2', f"TSP_weining_resampling_0307.tif")
        # 分块处理
        import math
        total_block_num = math.ceil(H/block_size)*math.ceil(W/block_size)
        with tqdm(total=total_block_num,desc="precesing blocks data") as pbar: 
            for h_start in range(0, H, block_size):
                h_end = min(h_start + block_size, H)
                for w_start in range(0, W, block_size):
                    block_start = time.time()
                    w_end = min(w_start + block_size, W)
                    # 1. 按需加载区块数据
                    block_data = []
                    for t_idx in range(len(self.tif_files)):
                        ds = gdal.Open(self.tif_files[t_idx], gdal.GA_ReadOnly)
                        img = ds.ReadAsArray(w_start, h_start, 
                                        w_end - w_start, 
                                        h_end - h_start)
                        if img.shape[0] == 12:
                            img = img[[1,2,3,4,5,6,7,8,10,11]]
                        tsp = gdal.Open(tsp_path, gdal.GA_ReadOnly)
                        tsp_img = tsp.ReadAsArray(w_start, h_start, 
                                        w_end - w_start, 
                                        h_end - h_start).reshape(1,img.shape[1], img.shape[2])
                        doy = self.parse_time(self.tif_files[t_idx])
                        doy_array = np.full((1, img.shape[1], img.shape[2]), doy, dtype=np.int32)
                        img = np.concatenate((img, tsp_img,doy_array), axis=0)
                        # print(img.shape)
                        block_data.append(img)
                    block_data = np.stack(block_data, axis=0)  # [T, C, h, w]
                    T, C, h, w = block_data.shape
                    print('block data shape:',block_data.shape)
                    
                    # 2. 转换数据格式
                    if patch:
                        patch_size = 5
                        padding = patch_size//2
                        if isinstance(block_data,np.ndarray):
                            block_data = torch.from_numpy(block_data.copy()).float().to(self.device) #[T,C,h,w]
                        paddding_block = F.pad(block_data,(padding,padding,padding,padding),mode='reflect') 
                        patches_data = []
                        # unfold methods
                        # for t in range(T):
                        #     patches_t = paddding_block[t].unfold(1,patch_size,1).unfold(2,patch_size,1)
                        #     patches_t = patches_t.permute(1,2,0,3,4)  # [C,H,W,patch_size,patch_size]
                        #     patches_data.append(patches_t)
                        # for -i -j
                        for i in range(h):
                            for j in range(w):
                                patch = paddding_block[:,:,i:i+patch_size,j:j+patch_size]
                                patches_data.append(patch)
                        patches_data = torch.stack(patches_data)
                        patches_data = patches_data.cpu().numpy()
                        del block_data,paddding_block,patches_t
                        torch.cuda.empty_cache() 
                        print('patch data shape:',patches_data.shape)
                    else:
                        pixel_data = block_data.transpose(2, 3, 0, 1)  # [h, w, T, C]
                        N = pixel_data.shape[0] * pixel_data.shape[1]
                        pixel_data = pixel_data.reshape(N, -1, block_data.shape[1])  # [N, T, C]
                        print('pixel data shape:',pixel_data.shape)

                    # 3. 分批预测                      
                    dataset = TABACCO_Crops(data=patches_data,nclasses=self.args.num_classes,datapath=self.args.image_dir)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size*16, shuffle=False) 
                    block_pred = []            
                    with torch.no_grad():
                        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
                            for idx, (X, y) in iterator:
                                X = recursive_todevice(X, self.device)
                                logits = self.model(X)
                                out = F.log_softmax(logits, dim=-1)
                                block_pred.append(out.argmax(-1).cpu().byte().numpy())

                    # 4. 重组并写入结果
                    block_pred = np.concatenate(block_pred, axis=0).reshape(h_end - h_start, w_end - w_start)
                    predictions[h_start:h_end, w_start:w_end] = block_pred
                    block_end = time.time()
                
                    block_time = (block_end - block_start)/60
                    used_time = (block_end - total_start)/60
                    total_time = block_time * total_block_num
                    remain_time = total_time - used_time
                    print(f"use time in one block prediction:{block_time:.2f}min")
                    print(f"use time in predictions:{used_time:.2f}min")
                    print(f"remain time in predictions:{remain_time:.2f}min")
                    
                    pbar.update(1)
        
                
        return predictions


    def predict_patch(self):
        patches, padded_shape = self.preprocess_patch()
        predictions = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(patches), self.args.batch_size)):
                batch = torch.FloatTensor(np.array(patches[i:i+self.args.batch_size])).to(self.device)
                outputs = self.model(batch)
                predictions.append(torch.argmax(outputs, dim=1).cpu())
                
        return self.merge_patches(torch.cat(predictions).numpy(), padded_shape)

    def merge_patches(self, preds, padded_shape):
        T, C, H, W = padded_shape
        output = np.zeros((H, W), dtype=np.uint8)
        count = np.zeros((H, W), dtype=np.uint8)
        
        idx = 0
        for i in range(0, H - self.args.patch_size + 1, self.args.stride):
            for j in range(0, W - self.args.patch_size + 1, self.args.stride):
                patch = preds[idx]
                output[i:i+self.args.patch_size, j:j+self.args.patch_size] += patch
                count[i:i+self.args.patch_size, j:j+self.args.patch_size] += 1
                idx += 1
                
        output = (output / count.max()).astype(np.uint8)
        return output[:self.temporal_data.shape[2], :self.temporal_data.shape[3]]

    def save_result(self, prediction, filename):
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(filename,
                             prediction.shape[1],
                             prediction.shape[0],
                             1,
                             gdal.GDT_Byte)
        out_ds.SetGeoTransform(self.metadata[3])
        out_ds.SetProjection(self.metadata[4])
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(prediction)
        out_ds.FlushCache()
        
        self.end = time.time()
        total_time = (self.end - self.start)/3600
        print(f"total time :{total_time:.2f}h")

    def run(self):
        if self.args.model_type == 'pixel':
            pred = self.predict_pixel()
        else:
            pred = self.predict_patch()
            
        os.makedirs(self.args.output_dir, exist_ok=True)
        output_path = os.path.join(self.args.output_dir, f'pred_{self.args.model_type}_0307.tif')
        self.save_result(pred, output_path)
        apply_color_table_to_tif(output_path, self.args.num_classes)


def apply_color_table_to_tif(output_path, num_classes):
    colors = color_table(num_classes)
    ds = gdal.Open(output_path, gdal.GA_Update)
    band = ds.GetRasterBand(1)
    ct = gdal.ColorTable()
    for i, color in enumerate(colors):
        ct.SetColorEntry(i, tuple(color))
    band.SetRasterColorTable(ct)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    ds.FlushCache()
    ds = None

def color_table(num_classes, background_color=(0, 0, 0, 0)):
    colors = np.zeros((num_classes, 4), dtype=np.uint8)
    predefined_colors = [
        (255, 0, 0, 255),    # Red
        (0, 255, 0, 255),    # Green
        (0, 0, 255, 255),    # Blue
        (255, 255, 0, 255),  # Yellow
        (255, 0, 255, 255),  # Magenta
        (0, 255, 255, 255),  # Cyan
        (128, 0, 0, 255),    # Maroon
        (0, 128, 0, 255),    # Dark Green
        (0, 0, 128, 255),    # Navy
        (128, 128, 0, 255),  # Olive
        (128, 0, 128, 255),  # Purple
        (0, 128, 128, 255),  # Teal
        (192, 192, 192, 255) # Silver
    ]
    for i in range(num_classes):
        if i < len(predefined_colors):
            colors[i] = predefined_colors[i]
        else:
            colors[i] = np.random.randint(0, 256, size=3).tolist() + [255]
    colors[0] = background_color
    return colors

if __name__ == "__main__":
    try:
        args = parse_args()
    except SystemExit as e:
        print(f"Error parsing arguments: {e}")
        import sys
        sys.exit(1)
    predictor = TemporalPredictor(args)
    predictor.run()
