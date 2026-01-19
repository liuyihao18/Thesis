import os

import numpy as np
import matplotlib.pyplot as plt

# --- 1. 设置中文字体 (推荐使用以下方法) ---
# 尝试设置 'SimHei'（黑体）。如果你的系统没有这个字体，请更改为系统中已有的中文字体名。
font_list = ['SimSun', 'Times New Roman', 'sans-serif']  # SimHei
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['font.serif'] = font_list

# 用于正常显示负号（通常字体库不支持负号显示时需要）
plt.rcParams['axes.unicode_minus'] = False

# 如果你想单独调整某些关键部分的默认大小：
# 标题 (Title) 的默认字体大小
plt.rcParams['axes.titlesize'] = 20
# 轴标签 (X/Y Label) 的默认字体大小
plt.rcParams['axes.labelsize'] = 16
# 图例 (Legend) 的默认字体大小
plt.rcParams['legend.fontsize'] = 14
# 刻度标签 (Tick Labels) 的默认字体大小
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# --- 1. 定义信号参数 (Parameters) ---
# 信号 H1: 振幅 A1, 相位 theta1 (弧度)
A1 = 1.5
theta1_deg = 30  # 角度
theta1_rad = np.deg2rad(theta1_deg)

# 信号 H2: 振幅 A2, 相位 theta2 (弧度)
A2 = 1.0
theta2_deg = 120 # 角度
theta2_rad = np.deg2rad(theta2_deg)

# --- 2. 计算复向量 (Complex Vectors) ---
# H1 = A1 * exp(j*theta1)
H1 = A1 * np.exp(1j * theta1_rad)
# H2 = A2 * exp(j*theta2)
H2 = A2 * np.exp(1j * theta2_rad)

# 结果向量 R = H1 + H2
R = H1 + H2

# 打印结果
print(f"H1 = {H1:.3f}")
print(f"H2 = {H2:.3f}")
print(f"R = H1 + H2 = {R:.3f} (Magnitude: {np.abs(R):.3f}, Phase: {np.rad2deg(np.angle(R)):.1f}°)")

# --- 3. 绘图设置 (Plotting Setup) ---
fig, ax = plt.subplots(figsize=(8, 8))

# 设置轴的限制，确保所有向量都可见
max_lim = max(np.abs(H1), np.abs(H2), np.abs(R)) * 1.2
ax.set_xlim((-max_lim / 4, max_lim * 0.8))
ax.set_ylim((-max_lim / 4, max_lim * 0.8))

# 绘制复平面轴
ax.axhline(0, color='gray', linewidth=0.5) # 实轴 (Real)
ax.axvline(0, color='gray', linewidth=0.5) # 虚轴 (Imaginary)

# 设置网格
ax.grid(True, linestyle='--', alpha=0.6)

# 设置轴标签
ax.set_xlabel("实部（I）")
ax.set_ylabel("虚部（Q）")
# ax.set_title(r"同频率信号相量求和 $R = H_1 + H_2$")

# 设置轴刻度
ax.set_xticklabels([])
ax.set_yticklabels([])

# --- 4. 绘制向量 (Drawing Vectors) ---
# 绘制 H1: 从原点 (0, 0) 开始
ax.quiver(0, 0, H1.real, H1.imag,
          angles='xy', scale_units='xy', scale=1,
          color='blue', width=0.005, headwidth=8, headlength=10,
          label=r'$H_1 = A_1 e^{j\theta_1}$')

# 绘制 H2: 遵循三角形法则，从 H1 的终点 (H1.real, H1.imag) 开始
# 这表示将 H2 向量平移到 H1 的头部
ax.quiver(H1.real, H1.imag, H2.real, H2.imag,
          angles='xy', scale_units='xy', scale=1,
          color='green', width=0.005, headwidth=8, headlength=10,
          label=r'$H_2 = A_2 e^{j\theta_2}$ (平移后)')

# 绘制 结果 R: 从原点 (0, 0) 到 H2 的终点 (R.real, R.imag)
ax.quiver(0, 0, R.real, R.imag,
          angles='xy', scale_units='xy', scale=1,
          color='red', width=0.007, headwidth=10, headlength=12,
          label=r'$R = H_1 + H_2$', zorder=3) # zorder=3 确保 R 在最上层

# --- 5. 添加文字标注和图例 (Labels and Legend) ---
# 标注 H1, H2, R 的终点
ax.text(H1.real * 1.05, H1.imag * 1.05, r'$H_1$', color='blue', fontsize=16)
ax.text(R.real * 0.9, R.imag * 0.77, r'$R$', color='red', fontsize=16) # 标注 R
ax.text(H1.real + H2.real * 0.5, H1.imag + H2.imag * 0.6, r'$H_2$', color='green', fontsize=16) # 标注平移后的 H2

# 创建图例 (Legend)
ax.legend(loc='lower left')

# 确保 x, y 轴比例相等，使圆和角度不失真
ax.set_aspect('equal', adjustable='box')

# 保存
dirname = '../../chap02/'
file_name = 'vector2D.pdf'
plt.savefig(os.path.join(dirname, file_name),
            format='pdf',          # 明确指定保存格式为PDF
            bbox_inches='tight',   # 自动裁剪掉图像周围多余的空白区域
            dpi=300)               # 对于矢量图（PDF/SVG）来说 DPI 影响不大，但可以保持习惯

plt.show()

