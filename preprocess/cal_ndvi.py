import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def load_data(file):
    data = []
    labels = []
    loaded_array = np.load(file, allow_pickle=True)
    data = loaded_array[0]
    all_doy = sorted(list(set(d[-1] for sample in data for d in sample)))
    labels = loaded_array[1]

    return data, labels, all_doy


def interpolated_data(file, out_dir, interpolate=False):
    """
    对光谱值进行处理，支持设置为 0 或线性插值
    :param file: 输入数据文件路径
    :param out_dir: 输出数据文件夹路径
    :param interpolate: 是否使用线性插值（True：插值，False：设置为 0）
    """
    data, labels, all_doy = load_data(file)
    interpolated_data = []
    spectral_len = len(data[0][0]) - 1  # 光谱值的长度（去掉 doy）

    for sample in tqdm(data, desc="Processing samples"):
        # 提取当前样本的 doy 和光谱值
        sample_doy = [time_point[-1] for time_point in sample]
        sample_spectral = [time_point[:-1] for time_point in sample]

        # 初始化插值后的光谱值
        interpolated_spectral = np.zeros((len(all_doy), spectral_len))

        if interpolate:
            # 使用线性插值
            for channel in range(spectral_len):
                # 对每个光谱通道进行线性插值
                f = interp1d(sample_doy, [spectral[channel] for spectral in sample_spectral],
                             kind='linear', fill_value="extrapolate")
                interpolated_spectral[:, channel] = f(all_doy)  # 对统一的 doy 轴进行插值
        else:
            # 设置为 0
            for time_point, spectral in zip(sample_doy, sample_spectral):
                index = all_doy.index(time_point)  # 找到对应的 doy 索引
                interpolated_spectral[index] = spectral  # 填充光谱值

        interpolated_data.append(interpolated_spectral)

    # 将 list 转换为 NumPy 数组 [N, T, C]
    interpolated_data = np.stack(interpolated_data)
    # dataset = [interpolated_data, label]

    # 保存插值后的数据
    data_path =  os.path.join(out_dir,'data_zero.npy')
    label_path = os.path.join(out_dir,'label_zero.npy')
    if not os.path.exists(data_path):
        # os.remove(out_file)
        np.save(data_path, interpolated_data)
        np.save(label_path,labels)

    print(f"Processed data saved to {data_path}")

    # 保存 all_doy 为 TXT 文件
    all_doy_file = data_path.replace('.npy', '_all_doy.txt')
    with open(all_doy_file, 'w') as f:
        for doy in all_doy:
            f.write(f"{doy}\n")

    print(f"DOY data saved to {all_doy_file}")

    return interpolated_data, labels, all_doy

def save_data_as_csv(data, labels, all_doys, save_dir, file_name):
    """
    将数据保存为 CSV 文件，每一行是一个样本点，包含自生成的 ID 和标签
    :param data: 数据，形状为 [N, T]
    :param labels: 标签数据，形状为 [N, T]
    :param all_doys: 统一的 doy 轴
    :param save_dir: 保存 CSV 文件的目录路径
    :param file_name: 保存的文件名
    """
    # 创建 DataFrame
    df = pd.DataFrame(data, columns=all_doys)  # 列名为 doy

    # 添加 ID 列和 Label 列
    df.insert(0, 'ID', range(1, len(df) + 1))  # 自生成 ID，从 1 开始
    df.insert(1, 'Label', labels)  # 在第二列插入标签

    # 保存为 CSV 文件
    csv_path = os.path.join(save_dir, file_name)
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

def calculate_ndvi(interpolated_data):
    """
    计算 NDVI
    :param interpolated_data: 插值后的数据，形状为 [N, T, C]
    :return: NDVI 数据，形状为 [N, T]
    """
    nir = interpolated_data[:, :, 6]  # 第 7 个波段（索引 6）
    red = interpolated_data[:, :, 2]  # 第 3 个波段（索引 2）
    
    # 计算 NDVI
    ndvi = (nir - red) / (nir + red + 1e-10)  # 添加小常数避免除零
    return ndvi


def plot_spectral_boxplots(data_dir, save_dir=None):
    """
    根据插值后的数据，绘制光谱值和 NDVI 的箱型图，并保存数据和图像
    :param data_dir: 数据目录路径
    :param save_dir: 保存图像的目录路径，如果为 None 则不保存
    """
    # 加载数据
    data_path = os.path.join(data_dir, 'data.npy')
    label_path = os.path.join(data_dir, 'label.npy')
    interpolated_data = np.load(data_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)
    all_doy_file = data_path.replace('.npy', '_all_doy.txt')

    # 读取 all_doy 文件
    with open(all_doy_file, 'r') as f:
        all_doys = [int(line.strip()) for line in f.readlines()]

    # 提取 label 为 1 的样本
    label_1_indices = np.where(labels == 2)[0]  # 找到 label 为 1 的样本索引
    label_1_data = interpolated_data[label_1_indices]  # 提取对应的数据

    # 获取波段数量
    num_bands = label_1_data.shape[2]

    # 创建保存目录
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 绘制每个波段的箱型图
    for band in range(num_bands):
        plt.figure(figsize=(15, 6))
        sns.set_style("whitegrid")  # 设置样式
        sns.set_palette("pastel")  # 设置颜色

        # 创建 DataFrame 用于绘图
        df = pd.DataFrame({
            'DOY': np.tile(all_doys, len(label_1_data)),
            'Spectral Value': label_1_data[:, :, band].flatten()
        })

        # 绘制箱型图
        sns.boxplot(
            x='DOY',
            y='Spectral Value',
            data=df,
            color='skyblue',
            width=0.6,
            linewidth=1.5,
            showfliers=False
        )

        # 美化图表
        plt.title(f'Boxplot of Spectral Band {band + 1} (Label = 1)', fontsize=16, fontweight='bold')
        plt.xlabel('DOY', fontsize=14, fontweight='bold')
        plt.ylabel('Spectral Value', fontsize=14, fontweight='bold')
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线

        # 保存图像
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'spectral_band_{band + 1}.png'), dpi=300, bbox_inches='tight')

        # 保存数据为 CSV 文件
        if save_dir:
            save_data_as_csv(
                label_1_data[:, :, band],  # 当前波段的数据
                labels[label_1_indices],  # 对应的标签
                all_doys,  # 统一的 doy 轴
                save_dir,  # 保存目录
                f'spectral_band_{band + 1}.csv'  # 文件名
            )

        plt.show()

    # 计算 NDVI
    ndvi_data = calculate_ndvi(interpolated_data)
    label_1_ndvi = ndvi_data[label_1_indices]  # 提取 label 为 1 的 NDVI 数据

    # 绘制 NDVI 箱型图
    plt.figure(figsize=(15, 6))
    sns.set_style("whitegrid")  # 设置样式
    sns.set_palette("pastel")  # 设置颜色

    # 创建 DataFrame 用于绘图
    df_ndvi = pd.DataFrame({
        'DOY': np.tile(all_doys, len(label_1_ndvi)),
        'NDVI': label_1_ndvi.flatten()
    })

    # 绘制箱型图
    sns.boxplot(
        x='DOY',
        y='NDVI',
        data=df_ndvi,
        color='skyblue',
        width=0.6,
        linewidth=1.5,
        showfliers=False
    )

    # 美化图表
    plt.title('Boxplot of NDVI (Label = 1)', fontsize=16, fontweight='bold')
    plt.xlabel('DOY', fontsize=14, fontweight='bold')
    plt.ylabel('NDVI', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线

    # 保存图像
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'ndvi_boxplot.png'), dpi=300, bbox_inches='tight')

    # 保存 NDVI 数据为 CSV 文件
    if save_dir:
        save_data_as_csv(
            ndvi_data,  # NDVI 数据
            labels,  # 标签
            all_doys,  # 统一的 doy 轴
            save_dir,  # 保存目录
            'ndvi_data.csv'  # 文件名
        )

    plt.show()

def plot_combined_boxplots(data_dir, save_dir=None, sample_size=1000):
    """
    绘制不同类别的波段值在一张箱型图中，随机抽取指定数量的样本，并过滤掉小于5%和大于95%分位数的数据
    :param data_dir: 数据目录路径
    :param save_dir: 保存图像的目录路径，如果为 None 则不保存
    :param sample_size: 随机抽取的样本数量，默认为 200
    """
    # 加载数据
    data_path = os.path.join(data_dir, 'data.npy')
    label_path = os.path.join(data_dir, 'label.npy')
    interpolated_data = np.load(data_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)
    all_doy_file = data_path.replace('.npy', '_all_doy_selected.txt')

    # 读取 all_doy 文件
    with open(all_doy_file, 'r') as f:
        all_doys = [int(line.strip()) for line in f.readlines()]

    # 获取波段数量
    num_bands = interpolated_data.shape[2]

    # 创建保存目录
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 绘制每个波段的箱型图
    for band in range(num_bands):
        # 提取当前波段的数据
        band_data = interpolated_data[:, :, band]  # 形状为 [N, T]

        # 创建 DataFrame 用于绘图
        data_list = []
        for label in np.unique(labels):  # 遍历所有类别（第一维是类别）
            if label > 0:
                label_indices = np.where(labels == label)[0]  # 找到当前类别的样本索引
                label_data = band_data[label_indices]  # 提取当前类别的数据

                # 随机抽取样本
                # if len(label_data) > sample_size:
                #     sampled_indices = np.random.choice(len(label_data), sample_size, replace=False)
                #     label_data = label_data[sampled_indices]

                # 检查数据是否有效
                if len(label_data) == 0:
                    print(f"Warning: No valid data for label {label} in band {band + 1}.")
                    continue

                # 清理数据，去除 NaN 和 Inf
                label_data = label_data[~np.isnan(label_data).any(axis=1)]  # 去除包含 NaN 的行
                label_data = label_data[~np.isinf(label_data).any(axis=1)]  # 去除包含 Inf 的行

                # 检查清理后的数据是否有效
                if len(label_data) == 0:
                    print(f"Warning: No valid data after cleaning for label {label} in band {band + 1}.")
                    continue

                # 对每个 doy 单独计算分位数并过滤数据
                for j, doy in enumerate(all_doys):
                    # 提取当前 doy 的光谱值
                    label_values = label_data[:, j] 
                    spectral_values = label_values[label_values != 0]

                    # 计算当前 doy 的 5% 和 95% 分位数
                    try:
                        lower_bound = np.percentile(spectral_values, 5)  # 5% 分位数
                        upper_bound = np.percentile(spectral_values, 95)  # 95% 分位数
                    except ValueError as e:
                        print(f"Error calculating percentiles for label {label} in band {band + 1}, DOY {doy}: {e}")
                        continue

                    # 过滤掉小于 5% 和大于 95% 分位数的数据
                    valid_indices = (spectral_values >= lower_bound) & (spectral_values <= upper_bound)
                    filtered_values = spectral_values[valid_indices]

                    # 将过滤后的数据添加到 DataFrame 中
                    for value in filtered_values:
                        data_list.append({
                            'DOY': doy,
                            'Spectral Value': value,
                            'Label': label
                        })

        df = pd.DataFrame(data_list)

        # 检查 DataFrame 是否为空
        if df.empty:
            print(f"Warning: No valid data to plot for band {band + 1}.")
            continue

        # 绘制箱型图
        plt.figure(figsize=(15, 6))
        sns.set_style("whitegrid")  # 设置样式
        num_classes = len(df['Label'].unique())  # 获取类别数量
        custom_palette = sns.color_palette("husl", num_classes)  # 使用 husl 调色板

        sns.boxplot(
            x='DOY',
            y='Spectral Value',
            hue='Label',  # 根据类别区分
            data=df,
            width=0.6,
            linewidth=1.5,
            showfliers=False,
            palette=custom_palette
        )

        # 美化图表
        plt.title(f'Boxplot of Spectral Band {band + 1} (Combined Labels)', fontsize=16, fontweight='bold')
        plt.xlabel('DOY', fontsize=14, fontweight='bold')
        plt.ylabel('Spectral Value', fontsize=14, fontweight='bold')
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线
        plt.legend(title='Label', fontsize=12)  # 添加图例

        # 保存图像
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'combined_spectral_band_{band + 1}.png'), dpi=300, bbox_inches='tight')

        plt.show()


if __name__ == "__main__":
    file = '/mnt/e/2024Work/tobacco/实验/points_train/three-class/data_weining_3class.npy'
    # file = '/root/models/SITS_MoCo/data/weining/2023/data.npy'
    out_dir = '/mnt/e/2024Work/tobacco/实验/points_train/three-class'
    map_dir = '/mnt/e/2024Work/tobacco/实验/points_train/three-class/boxplots_threeclass_tobacco'
    # interpolated_data(file, out_dir)
    plot_spectral_boxplots(out_dir, map_dir)
    # plot_combined_boxplots(out_dir, map_dir)
