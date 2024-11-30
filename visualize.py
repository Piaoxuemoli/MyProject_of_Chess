import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator  # 从matplotlib.ticker导入AutoMinorLocator

# 读取CSV文件
data = pd.read_csv('losses.csv')

# 提取epoch和loss列
epochs = data['epoch']
losses = data['loss']

# 定义两个训练阶段的起点
start_epoch_first_phase = 3  # 第一次训练从第3个epoch开始
start_epoch_second_phase = 2  # 第二次训练从第2个epoch开始

# 找到第一次训练的终点（第800个epoch）
end_epoch_first_phase = 800

# 计算两次训练的数据分割点
split_point = data[data['epoch'] == end_epoch_first_phase].index[0]

# 分割数据为两部分
early_data = data.loc[:split_point]  # 包括第800行
late_data = data.loc[split_point + 1:]  # 从第801个epoch开始

# 绘制图表
plt.figure(figsize=(14, 6), dpi=300)  # 更宽的图表，更高的分辨率

# 绘制第一次训练的数据，颜色为蓝色，更细的线和更小的点
plt.plot(early_data['epoch'], early_data['loss'], label=f'First Training Phase (from {start_epoch_first_phase} to {end_epoch_first_phase})', color='blue', linewidth=0.5, marker='o', markersize=2)

# 绘制第二次训练的数据，颜色为红色，更细的线和更小的点
plt.plot(late_data['epoch'], late_data['loss'], label=f'Second Training Phase (from {start_epoch_second_phase + (split_point+1-start_epoch_first_phase)})', color='red', linewidth=0.5, marker='o', markersize=2)

# 确保799和800这两个点之间没有连线
plt.scatter([early_data['epoch'].iloc[-1]], [early_data['loss'].iloc[-1]], color='blue', s=10)  # 最后一个蓝点
plt.scatter([late_data['epoch'].iloc[0]], [late_data['loss'].iloc[0]], color='red', s=10)        # 第一个红点

# 添加标签和标题
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs with Discontinuous Line at 800th Epoch')

# 显示图例
plt.legend()

# 设置主要和次要网格线
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 启用次要网格线
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.3)

# 调整x轴的刻度以适应次要网格线
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(100))  # 主要刻度间隔
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(20))   # 次要刻度间隔

# 增加y轴的次要刻度线密度
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())  # 使用导入的AutoMinorLocator

# 显示图形
plt.show()