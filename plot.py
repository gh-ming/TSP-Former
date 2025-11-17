import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_attention_maps(mean_features, labels, n_head, T, figure_path='./figures'):
    """
    绘制多头注意力图

    参数:
    - mean_features: 形状为 [num_classes, n_head, T] 的张量
    - labels: 类别标签列表
    - n_head: 注意力头的数量
    - T: 时间序列长度
    - figure_path: 图像保存路径
    """
    os.makedirs(figure_path, exist_ok=True)


    # 绘制每个头的注意力图
    for head in range(n_head):
        plt.figure(figsize=(12, 8))
        for cls_idx, label in enumerate(labels):
            plt.plot(np.arange(T), mean_features[cls_idx, head], label=label)
        
        plt.title(f"Attention Map for Head {head + 1}")
        plt.xlabel("Time")
        plt.ylabel("Mean Feature Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{figure_path}/attention_head_{head + 1}.png", bbox_inches='tight')
        plt.close()

    # 绘制综合所有头的注意力图
    plt.figure(figsize=(12, 8))
    for cls_idx, label in enumerate(labels):
        mean_combined = mean_features[cls_idx].mean(axis=0)
        plt.plot(np.arange(T), mean_combined, label=label)
    
    plt.title("Combined Attention Map for All Heads")
    plt.xlabel("Time")
    plt.ylabel("Mean Feature Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{figure_path}/combined_attention.png", bbox_inches='tight')
    plt.close()

def attention_plot(attention, x_texts, y_texts=None, figsize=(15, 10), annot=False, figure_path='./figures',
                   figure_name='attention_weight.png'):
    """
    绘制注意力热力图

    参数:
    - attention: 注意力权重矩阵
    - x_texts: x 轴标签
    - y_texts: y 轴标签
    - figsize: 图像尺寸
    - annot: 是否显示数值
    - figure_path: 图像保存路径
    - figure_name: 图像文件名
    """
    os.makedirs(figure_path, exist_ok=True)

    plt.clf()
    sns.set(font_scale=1.25)
    plt.figure(figsize=figsize)
    sns.heatmap(
        attention,
        cbar=True,
        cmap="RdBu_r",
        annot=annot,
        square=True,
        fmt='.2f',
        annot_kws={'size': 10},
        # yticklabels=y_texts,
        # xticklabels=x_texts
    )
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, figure_name), bbox_inches='tight')
    plt.close()

# 示例用法
if __name__ == "__main__":
    # 假设我们已经计算出 mean_features
    num_classes = 5
    n_head = 8
    T = 20
    mean_features = torch.randn(num_classes, n_head, T)
    labels = [f"Class {i}" for i in range(num_classes)]
    
    plot_attention_maps(mean_features, labels, n_head, T, figure_path='attention_plots')

    # 生成一个注意力权重矩阵示例（假设为 num_classes x n_head 矩阵）
    attention_weights = torch.randn(num_classes, n_head).softmax(dim=1)
    x_texts = [f"Head {i+1}" for i in range(n_head)]
    y_texts = [f"Class {i}" for i in range(num_classes)]
    
    attention_plot(
        attention=attention_weights.detach().cpu().numpy(),
        x_texts=x_texts,
        y_texts=y_texts,
        annot=True,
        figure_path='attention_plots',
        figure_name='attention_weights.png'
    )