import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator  # ��matplotlib.ticker����AutoMinorLocator

# ��ȡCSV�ļ�
data = pd.read_csv('losses.csv')

# ��ȡepoch��loss��
epochs = data['epoch']
losses = data['loss']

# ��������ѵ���׶ε����
start_epoch_first_phase = 3  # ��һ��ѵ���ӵ�3��epoch��ʼ
start_epoch_second_phase = 2  # �ڶ���ѵ���ӵ�2��epoch��ʼ

# �ҵ���һ��ѵ�����յ㣨��800��epoch��
end_epoch_first_phase = 800

# ��������ѵ�������ݷָ��
split_point = data[data['epoch'] == end_epoch_first_phase].index[0]

# �ָ�����Ϊ������
early_data = data.loc[:split_point]  # ������800��
late_data = data.loc[split_point + 1:]  # �ӵ�801��epoch��ʼ

# ����ͼ��
plt.figure(figsize=(14, 6), dpi=300)  # �����ͼ�����ߵķֱ���

# ���Ƶ�һ��ѵ�������ݣ���ɫΪ��ɫ����ϸ���ߺ͸�С�ĵ�
plt.plot(early_data['epoch'], early_data['loss'], label=f'First Training Phase (from {start_epoch_first_phase} to {end_epoch_first_phase})', color='blue', linewidth=0.5, marker='o', markersize=2)

# ���Ƶڶ���ѵ�������ݣ���ɫΪ��ɫ����ϸ���ߺ͸�С�ĵ�
plt.plot(late_data['epoch'], late_data['loss'], label=f'Second Training Phase (from {start_epoch_second_phase + (split_point+1-start_epoch_first_phase)})', color='red', linewidth=0.5, marker='o', markersize=2)

# ȷ��799��800��������֮��û������
plt.scatter([early_data['epoch'].iloc[-1]], [early_data['loss'].iloc[-1]], color='blue', s=10)  # ���һ������
plt.scatter([late_data['epoch'].iloc[0]], [late_data['loss'].iloc[0]], color='red', s=10)        # ��һ�����

# ��ӱ�ǩ�ͱ���
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs with Discontinuous Line at 800th Epoch')

# ��ʾͼ��
plt.legend()

# ������Ҫ�ʹ�Ҫ������
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# ���ô�Ҫ������
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.3)

# ����x��Ŀ̶�����Ӧ��Ҫ������
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(100))  # ��Ҫ�̶ȼ��
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(20))   # ��Ҫ�̶ȼ��

# ����y��Ĵ�Ҫ�̶����ܶ�
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())  # ʹ�õ����AutoMinorLocator

# ��ʾͼ��
plt.show()