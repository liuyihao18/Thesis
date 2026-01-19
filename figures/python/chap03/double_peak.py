import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# --- 0. 字体设置 ---
font_list = ['SimSun', 'Times New Roman', 'sans-serif']  # SimHei
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['font.serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False

title_size = 16
label_size = 16
tick_size = 14

# --- 1. 参数设置 ---
fs = 200e3       # 采样率 200 KHz (根据Y轴范围 -100到100 设置)
duration = 0.02  # 持续时间 20ms
t = np.arange(0, duration, 1/fs) # 时间轴

# --- 2. 构造信号 ---
# 我们需要构造一个复数信号 (Complex Signal) 才能区分正负频率
signal = np.zeros_like(t, dtype=complex)

# 定义两段不同的频率特征
# 第一阶段 (0-10ms): 中心频率 50kHz, 边带 +/- 20kHz
mask1 = t < 0.010
f_c1 = 50e3
signal[mask1] = (1.0 * np.exp(1j * 2 * np.pi * f_c1 * t[mask1]) +           # 主载波 (强)
                 0.4 * np.exp(1j * 2 * np.pi * (f_c1 - 20e3) * t[mask1]) +  # 下边带 (弱)
                 0.4 * np.exp(1j * 2 * np.pi * (f_c1 + 20e3) * t[mask1]))   # 上边带 (弱)

# 第二阶段 (10-20ms): 中心频率 -50kHz, 边带 +/- 20kHz
mask2 = t >= 0.010
f_c2 = -50e3
signal[mask2] = (1.0 * np.exp(1j * 2 * np.pi * f_c2 * t[mask2]) +           # 主载波
                 0.4 * np.exp(1j * 2 * np.pi * (f_c2 - 20e3) * t[mask2]) +  # 边带
                 0.4 * np.exp(1j * 2 * np.pi * (f_c2 + 20e3) * t[mask2]))   # 边带

# --- 3. 计算 STFT (短时傅里叶变换) ---
# nperseg 决定了频率分辨率（越高越细腻），noverlap 决定时间平滑度
f, t_spec, Zxx = stft(signal, fs, window='hann', nperseg=128, noverlap=100, return_onesided=False)

# --- 4. 数据调整 ---
# stft默认输出是 [0, fs/2, -fs/2, 0]，需要使用 fftshift 移位，使 0 频率在中心
Zxx = np.fft.fftshift(Zxx, axes=0)
f = np.fft.fftshift(f)

# 单位转换：频率转为 KHz，时间转为 ms
f_khz = f / 1e3
t_ms = t_spec * 1e3

# --- 5. 绘图 ---
plt.figure(figsize=(6, 5))

# 绘制热力图
# vmin/vmax 用来控制颜色对比度，使背景更蓝，线条更亮
plt.pcolormesh(t_ms, f_khz, np.abs(Zxx), shading='gouraud', cmap='jet', vmin=0, vmax=0.8)

# 设置坐标轴标签和范围
plt.title('STFT 频谱图', fontsize=title_size)
plt.xlabel('时间 (ms)', fontsize=label_size)
plt.ylabel('频率 (KHz)', fontsize=label_size)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.ylim(-100, 100) # 对应图片的Y轴范围
plt.xlim(0, 20)     # 对应图片的X轴范围

# 保存文件
# 保存
dirname = '../../chap03/'
file_name = 'double_peak.pdf'
plt.savefig(os.path.join(dirname, file_name),
            format='pdf',          # 明确指定保存格式为PDF
            bbox_inches='tight',   # 自动裁剪掉图像周围多余的空白区域
            dpi=300)               # 对于矢量图（PDF/SVG）来说 DPI 影响不大，但可以保持习惯

# 调整布局并显示
plt.tight_layout()
plt.show()