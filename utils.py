import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Union
from sklearn.ensemble import RandomForestClassifier

import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from datasets import *
from models import *
from prettytable import PrettyTable
import csv
import os

SITES = ['WN','HZ','XC','XS','XW']
CLASS = ['others', 'tabbcco']
# -------------------------- #
#          dataset           #
# -------------------------- #
def get_data(data_npy):
    loaded_array = np.load(data_npy, allow_pickle=True)
    data = np.stack(loaded_array[0],axis=0)
    labels = loaded_array[1].astype(int)
    return data, labels

def get_tabacco_dataloader(modelname: str,
                          datapath: Union[str, Path],
                          batchsize: int,
                          patch_size: int = 7,
                          mode: str = 'center',
                          nidms: int = 10,
                          seed: int = 111):
    """支持多站点测试的数据加载接口
    
    参数:
        modelname: 模型名称 ['rf', 'RF'] 或其他深度学习模型
        datapath: 数据目录路径
        patch_size: patch尺寸
        nclasses: 类别数量
        mode: 数据模式 ['center', 'patch']
        nidms:spectral bands
        seed: 随机种子
    """
    datapath = Path(datapath)

    # 统一加载函数
    def load_site_data(site: str, split: str):
        """加载指定站点的split数据"""
        npy_path = datapath / f"{site}_{split}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"数据文件 {npy_path} 不存在")
        data, labels = get_data(npy_path)
        print(f"[{site}] {split} 集尺寸:", data.shape)
        return data, labels

    # 加载训练集和验证集（仅用WN站点）
    X_train, y_train = load_site_data('WN', 'train')
    X_val, y_val = load_site_data('WN', 'val')

    # 加载所有测试站点（排除WN）
    test_sites = [site for site in SITES if site != 'WN']
    test_data = {
        site: load_site_data(site, 'test')
        for site in test_sites
    }

    # 统一参数配置
    common_args = {
        'mode': mode,
        'patch_size': patch_size,
        'datapath': datapath,
        'scale_factor': 1e-4,
        'ndims': nidms,
        'use_cache': True
    }
    # 处理不同模型类型
    if modelname.lower() == 'rf':
        # 随机森林需要展平特征
        def flatten_dataset(data, labels):
            dataset = TABACCO_Crops(data=data, labels=labels, **common_args)
            features = []
            for x, _ in dataset:
                spectral = x[0].numpy().reshape(-1) #[T,C] -> [-1]
                # doy = x[1].numpy().astype(float)
                # import ipdb;ipdb.set_trace()
                # spectral = x[0].numpy()
                # weight =  x[3]
                # weight[torch.isnan(weight)] = 0
                # tsp = weight.numpy()
                # new_spl =np.column_stack((spectral,tsp)).reshape(-1)
                # ndvi = x[2].numpy()
                features.append(spectral)
            return np.stack(features), labels

        # 处理训练数据
        X_train, y_train = flatten_dataset(X_train, y_train)
        # X_val, y_val = flatten_dataset(X_val, y_val)
        
        # 处理测试数据
        test_features = {}
        for site, (data, labels) in test_data.items():
            test_features[site] = flatten_dataset(data, labels)
        
        return (X_train, y_train), test_features

    else:
        # 深度学习模式
        train_ds = TABACCO_Crops(X_train, y_train, **common_args)
        val_ds = TABACCO_Crops(X_val, y_val, **common_args)
        # test = train_ds[0]
        
        # 自动生成统计文件
        if not (datapath/'spectral_stats.npz').exists():
            print("初始化全局统计量...")
            _ = train_ds.mean, train_ds.std  # 触发计算
        
        # 创建数据加载器
        loader_args = {
            'batch_size': batchsize,
            'pin_memory': True,
            'num_workers': 4
        }
        train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
        
        # 创建测试集加载器字典
        test_loaders = {
            site: DataLoader(
                TABACCO_Crops(data, labels, **common_args),
                shuffle=False,
                **loader_args
            ) for site, (data, labels) in test_data.items()
        }
        
        return (train_loader, val_loader), test_loaders




# -------------------------- #
#           Model            #
# -------------------------- #
def get_model(modelname, ndims, num_classes, device,patch_size=None):
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
        model = TSP_TransNet(patch_size=patch_size).to(device)
    else:
        raise ValueError(
            "invalid model argument. choose from 'Transformer', 'TempCNN', 'LSTM', 'LTAE', 'RF', or 'STNet' ")

    return model




# -------------------------- #
#           Utils            #
# -------------------------- #
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, site_name=None, export_csv=None):
    CLASS = ['others', 'tobacco']
    num_classes = len(CLASS)
    # 获取基础指标
    num = target.shape[0]
    output = np.array(output).astype(int)
    target = target.astype(int)
    confusion_matrix = get_confusion_matrix(output, target, num_classes)
    
    # 计算各项指标
    TP = confusion_matrix.diagonal()
    FP = confusion_matrix.sum(1) - TP
    FN = confusion_matrix.sum(0) - TP    
    # 整体指标
    po = TP.sum() / num
    pe = (confusion_matrix.sum(0) * confusion_matrix.sum(1)).sum() / num ** 2
    if pe == 1:
        kappa = 1
    else:
        kappa = (po - pe) / (1 - pe)

    p = TP / (TP + FP + 1e-12)
    r = TP / (TP + FN + 1e-12)
    f1 = 2 * p * r / (p + r + 1e-12)

    oa = po
    kappa = kappa
    macro_f1 = f1.mean()
    weight = confusion_matrix.sum(0) / confusion_matrix.sum()
    weighted_f1 = (weight * f1).sum()
    class_f1 = f1
    table = PrettyTable()
    table.field_names = ["Class","Precision", "Recall","F1"]

    for i, class_name in enumerate(CLASS):
        table.add_row([class_name, f"{p[i]:.4f}",f"{r[i]:.4f}",f"{f1[i]:.4f}"])
    table.add_row(["OA", "",f"{oa:.4f}", ""])
    table.add_row(["macro_f1", "",f"{macro_f1:.4f}", ""])
    table.add_row(["weighted_f1","", f"{weighted_f1:.4f}", ""])
    table.add_row(["kappa","", f"{kappa:.4f}", ""])
    print(table)

    result = dict(
        oa=oa,
        kappa=kappa,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        class_f1=class_f1,
        confusion_matrix=confusion_matrix
    )
    
    # 获取tobacco类别索引（假设是第1类）
    tobacco_idx = CLASS.index('tobacco') if 'tobacco' in CLASS else 1
    
    # 构建结果字典
    result_tobacco = {
        'site_name': site_name,
        'tobacco P': p[tobacco_idx],
        'tobacco R': r[tobacco_idx],
        'F1': weighted_f1,
        'OA': oa,
        'kappa': kappa
    }
    
    # 自动导出CSV逻辑
    if export_csv:
        file_exists = Path(export_csv).exists()
        with open(export_csv, 'a' if file_exists else 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result_tobacco.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_tobacco)
    
    return result


def get_confusion_matrix(y_pred, y_true, num_classes=21):
    idx = y_pred * num_classes + y_true
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def save(model, path="model.pth", **kwargs):
    print(f"saving model to {str(path)}\n")
    model_state = model.state_dict()
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(dict(model_state=model_state, **kwargs), path)


def overall_performance(logdir):
    overall_metrics = defaultdict(list)

    for seed in [111, 222, 333, 444, 555]:
        log_dir = Path(logdir.replace(re.findall('Seed\d+', str(logdir))[0], f'Seed{seed}'))
        log_fn = log_dir / f'testlog.csv'
        if log_fn.exists():
            test_metrics = pd.read_csv(log_fn).iloc[0].to_dict()
            for metric, value in test_metrics.items():
                overall_metrics[metric].append(value)

    print(f'Overall result across 5 trials:')
    for metric, values in overall_metrics.items():
        values = np.array(values)
        if isinstance(values[0], (str)) or np.any(np.isnan(values)):
            continue
        if 'loss' in metric or 'f1' in metric or 'kappa' in metric:
            print(f"{metric}: {np.mean(values):.4}")
        else:
            values *= 100
            print(f"{metric}: {np.mean(values):.2f}")

    print(f'{np.mean(overall_metrics["oa"])*100:.2f}\t'
          f'{np.mean(overall_metrics["kappa"]):.4f}\t'
          f'{np.mean(overall_metrics["weighted_f1"]):.4f}')
    print()

