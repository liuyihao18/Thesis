import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

# --- 0. 字体设置 ---
font_list = ['SimSun', 'Times New Roman', 'sans-serif']  # SimHei
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['font.serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False

title_size = 18
label_size = 18
tick_size = 16
legend_size = 16

# --- 1. 参数设置 ---
N = 100  # 窗口长度 (样本点数)
Fs = 1.0  # 采样频率 (为了进行归一化频率计算，可以设为任意值，如 1.0)
n = np.arange(N) # 0 到 N-1 的样本点

# --- 2. 生成窗函数 ---
# 汉明窗 (Hamming Window)
# 公式: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
w_hamming = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) - 0.006 * np.cos(2 * np.pi * n / (N - 1)) ** 3

# 矩形窗 (Rectangular Window) - 对应图中的“标准傅里叶变换”
# (在时域中，标准的傅里叶变换相当于加了一个矩形窗，权重恒为 1)
w_rect = np.ones(N)

# --- 3. 计算频域响应 ---
# 进行 FFT。为了更好地显示细节，FFT 长度通常远大于窗口长度 N
# N_fft 必须是 2 的幂次，以提高 FFT 效率，比如 N_fft=1024
N_fft = 1024

# 对窗口函数进行 FFT
W_hamming = fft(w_hamming, N_fft)
W_rect = fft(w_rect, N_fft)

# 使用 fftshift 将零频率分量移动到频谱中心
W_hamming_shifted = fftshift(W_hamming)
W_rect_shifted = fftshift(W_rect)

# 计算幅度谱的 dB 值：20 * log10(|W| / max(|W|))
# 注意：通常对窗口函数分析时，会将其最大值归一化到 0 dB
# 我们使用 np.abs(W_rect).max() 作为归一化基准
max_amplitude = np.abs(W_rect).max()

# 归一化后的 dB 响应
# 汉明窗：
dB_hamming = 20 * np.log10(np.abs(W_hamming_shifted) / max_amplitude)
# 矩形窗：
dB_rect = 20 * np.log10(np.abs(W_rect_shifted) / max_amplitude)

# 生成归一化频率轴：频率范围从 -Fs/2 到 +Fs/2
# 这里我们只绘制从 0 到 1 的归一化频率 (通常是单边谱)
# N_fft/2 是单边谱的有效点数
f = np.linspace(0, Fs, N_fft, endpoint=False) # 0 到 Fs-dF

# 为了匹配图 (b) 中的 0 到 1 归一化频率，我们取前半部分，并用 N_fft/2 归一化
# 频域点的个数
num_points = N_fft // 2
# 归一化频率轴 (0 到 1)
normalized_freq = np.linspace(0, 1, num_points, endpoint=False)


# --- 4. 绘图 ---

# 创建包含两个子图的画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) # 宽 12，高 5

# --- (a) 时域权重图 ---
ax1.plot(n, w_hamming, label='汉明窗+', color='tab:blue')
ax1.plot(n, w_rect, label='标准傅里叶变换', color='tab:red')

ax1.set_title('(a) 时域权重', fontsize=title_size)
ax1.set_xlabel('样本点', fontsize=label_size)
ax1.set_ylabel('权重', fontsize=label_size)
ax1.set_ylim(0, 1.1)  # 匹配图示的 Y 轴范围
ax1.set_xlim(0, N)
ax1.tick_params(labelsize=tick_size)
ax1.legend(fontsize=legend_size)
ax1.grid(True)


# --- (b) 频域响应图 ---
# 注意：我们只绘制单边频谱 (从 0 到 N_fft/2)
# 需要截取 dB_hamming 和 dB_rect 的前 N_fft/2 个点
dB_hamming_one_side = dB_hamming[N_fft // 2 : N_fft // 2 + num_points]
dB_rect_one_side = dB_rect[N_fft // 2 : N_fft // 2 + num_points]

# 为了匹配图示，我们通常只看主瓣右侧和旁瓣，因此取归一化频率轴的 0 到 1/2 部分
# 假设图 (b) 中的归一化频率是 f / (Fs/2)，即 0 到 2*f/Fs，最大到 1
# 重新计算频率轴，使最大值为 1 (对应 Fs/2)
normalized_freq_plot = np.linspace(0, 1, num_points, endpoint=False)

ax2.plot(normalized_freq_plot, dB_hamming_one_side, label='汉明窗+', color='tab:blue')
ax2.plot(normalized_freq_plot, dB_rect_one_side, label='标准傅里叶变换', color='tab:red')

ax2.set_title('(b) 频率响应', fontsize=title_size)
ax2.set_xlabel('归一化频率', fontsize=label_size)
ax2.set_ylabel('频率响应', fontsize=label_size)
ax2.set_ylim(-80, 5)  # 匹配图示的 Y 轴范围
ax2.set_xlim(0, 1)
ax2.tick_params(labelsize=tick_size)
ax2.legend(fontsize=legend_size)
ax2.grid(True)

dirname = '../../chap03/'
file_name = 'window.pdf'
plt.savefig(os.path.join(dirname, file_name),
            format='pdf',          # 明确指定保存格式为PDF
            bbox_inches='tight',   # 自动裁剪掉图像周围多余的空白区域
            dpi=300)               # 对于矢量图（PDF/SVG）来说 DPI 影响不大，但可以保持习惯

# 调整子图间距
plt.tight_layout()
plt.show()
