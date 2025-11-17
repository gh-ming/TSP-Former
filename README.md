# TSP-Former 项目介绍

## 项目简介
TSP-Former 是一个用于时序遥感影像分类与预测的深度学习项目，主要应用于烟草等作物的识别。项目集成了多种时序建模方法（如 LSTM、Transformer、STNet、TempCNN、LTAE、TSP_TransNet 等），并支持完整的数据预处理、模型训练与预测流程。

## 目录结构
- `datasets/`：数据集处理与加载相关代码
- `models/`：各类时序深度学习模型实现
- `preprocess/`：遥感影像数据预处理脚本
- `results/`：模型输出结果与实验记录
- 主程序脚本：
  - `train_tobacco.py`：模型训练主脚本
  - `predict.py`、`predict_by_shp.py`、`big_tif_predict.py`：模型预测脚本
  - `main_ghm.py`：主流程脚本
  - `utils.py`、`plot.py`：工具与可视化

## 数据预处理
数据预处理相关脚本位于 `preprocess/` 目录，包括：
- 影像镶嵌与重采样（`mosaic.py`, `resampling.py`）
- NDVI 等植被指数计算（`cal_ndvi.py`）
- 时序影像拼接与采样（`S2_time_series.py`, `sampling_S2.py` 等）
- 栅格与矢量数据转换（`shp2tif.py`, `polygon_to_points.py`）
- 数据分析与整理（`anlysis.py`, `proprecess.py`）

典型流程：
1. 使用 `mosaic.py`、`resampling.py` 对原始影像进行拼接和重采样。
2. 用 `cal_ndvi.py` 计算 NDVI 等特征。
3. 通过 `S2_time_series.py`、`sampling_S2.py` 生成时序样本。
4. 利用 `shp2tif.py`、`polygon_to_points.py` 进行标签数据处理。

## 模型训练
- 训练脚本为 `train_tobacco.py`，支持多种模型选择（如 LSTM、Transformer、TSP_TransNet 等）。
- 可通过命令行参数指定模型类型、数据路径、超参数等。
- 训练过程自动保存模型权重与日志到 `results/` 目录。
- 训练示例：
  ```bash
  python train_tobacco.py --model TSP_TransNet --data_path ./datasets/xxx --epochs 100 --batch_size 32
  ```

## 预测与评估
- 预测脚本包括 `predict.py`、`predict_by_shp.py`、`big_tif_predict.py`，支持对单幅影像、矢量范围或大面积影像进行预测。
- 预测结果保存在 `results/` 目录下，支持多种格式输出。
- 评估与可视化可通过 `plot.py` 实现。

## 依赖环境
- 推荐使用 Anaconda 环境，环境依赖见 `environment.yml`。
- 主要依赖：PyTorch、numpy、rasterio、GDAL、scikit-learn 等。


