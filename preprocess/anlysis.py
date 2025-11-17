import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = '/mnt/e/2024Work/tobacco/实验/points_train/three-class/统计.xlsx'  # 替换为你的 Excel 文件路径
df = pd.read_excel(file_path,header=[0, 1])  # 读取多级表头

# 提取数据
data = {}
for group in df.columns.levels[0]:  # 遍历所有组
    data[group] = {
        'tobacco': df[group]['tobacco'].dropna().tolist(),
        'corn': df[group]['corn'].dropna().tolist()
    }

# 定义相同的区间划分
bins = np.linspace(3000, 7000, 50)

# 计算每组数据的重叠面积
overlap_areas = {}

for group, values in data.items():
    hist_tobacco, _ = np.histogram(values['tobacco'], bins=bins, density=True)
    hist_corn, _ = np.histogram(values['corn'], bins=bins, density=True)
    
    bin_widths = np.diff(bins)
    overlap = np.minimum(hist_tobacco, hist_corn) * bin_widths
    total_overlap = np.sum(overlap)
    
    overlap_areas[group] = total_overlap

    # 可视化直方图
    plt.hist(values['tobacco'], bins=bins, density=True, alpha=0.5, label='Tobacco')
    plt.hist(values['corn'], bins=bins, density=True, alpha=0.5, label='Corn')
    plt.title(f'{group} Histogram Overlap (Area = {total_overlap:.2f})')
    plt.legend()
    plt.show()

    plt.savefig(f'/mnt/e/2024Work/tobacco/实验/points_train/three-class/{group}_histogram.png', dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    plt.close()  # 关闭当前图像，避免内存泄漏

# 将结果保存到 Excel 文件
result_df = pd.DataFrame(list(overlap_areas.items()), columns=['Group', 'Overlap Area'])
result_df.to_excel('/mnt/e/2024Work/tobacco/实验/points_train/three-class/overlap_areas.xlsx', index=False)

# 输出重叠面积
print(result_df)