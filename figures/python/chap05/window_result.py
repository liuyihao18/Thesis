import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hamming

# 0. 字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 信号参数设置
fs = 1000                     # 采样频率 (Hz)
T = 1.0                       # 信号时长 (s)
t = np.linspace(0, T, int(fs * T), endpoint=False)

# 信号 A: 10Hz, 振幅 1.0
# 信号 B: 14Hz, 振幅 0.1
sig = 1.0 * np.sin(2 * np.pi * 10 * t) + 0.1 * np.sin(2 * np.pi * 14 * t)

# 2. 傅里叶变换处理
N = len(t)
xf = fftfreq(N, 1/fs)[:N//2]

# 方案一：标准傅里叶变换 (Rectangular Window)
yf_standard = fft(sig)
mag_standard = 2.0/N * np.abs(yf_standard[0:N//2])
db_standard = 20 * np.log10(mag_standard + 1e-6) # 转换为dB，加微小值防止log(0)

# 方案二：汉明窗 + 傅里叶变换
win = hamming(N)
yf_hamming = fft(sig * win)
# 补偿窗函数增益损失 (Hamming窗相干增益约为0.54)
mag_hamming = (2.0 / np.sum(win)) * np.abs(yf_hamming[0:N//2])
db_hamming = 20 * np.log10(mag_hamming + 1e-6)

# 3. 绘图
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)

# 绘制左侧轴：标准傅里叶变换 (橙色)
color_std = '#D35400' # 类似图中的橙红色
lns1 = ax1.plot(xf, db_standard, color=color_std, linewidth=2, label='标准傅里叶变换')
ax1.set_xlabel('频率(Hz)', fontsize=14)
ax1.set_ylabel('幅度(dB)', color=color_std, fontsize=14)
ax1.tick_params(axis='y', labelcolor=color_std)
ax1.set_ylim(0, 50) # 根据你的图调整刻度
ax1.grid(True, linestyle='-', alpha=0.6)

# 绘制右侧轴：汉明窗 (蓝色)
ax2 = ax1.twinx()
color_ham = '#0072BD' # 类似图中的深蓝色
lns2 = ax2.plot(xf, db_hamming, color=color_ham, linewidth=3, label='汉明窗+')
ax2.set_ylabel('幅度(dB)', color=color_ham, fontsize=14)
ax2.tick_params(axis='y', labelcolor=color_ham)
ax2.set_ylim(-40, 40) # 右侧坐标轴范围通常不同

# 4. 细节修饰
plt.xlim(2, 45) # 只显示感兴趣的频率范围
ax1.set_xticks([10, 20, 30, 40])

# 合并图例
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper right', frameon=True, edgecolor='black', prop={'size': 12})

plt.title('双信号频谱分析对比', fontsize=15)
plt.tight_layout()
plt.show()