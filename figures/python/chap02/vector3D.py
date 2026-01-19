import os

import numpy as np
import matplotlib.pyplot as plt

# --- 1. 全局字体和大小设置 ---
font_list = ['SimSun', 'Times New Roman', 'sans-serif']  # SimHei
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['font.serif'] = font_list
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# 如果你想单独调整某些关键部分的默认大小：
# 标题 (Title) 的默认字体大小
plt.rcParams['axes.titlesize'] = 22
# 轴标签 (X/Y Label) 的默认字体大小
plt.rcParams['axes.labelsize'] = 18
# 图例 (Legend) 的默认字体大小
plt.rcParams['legend.fontsize'] = 16
# 刻度标签 (Tick Labels) 的默认字体大小
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# --- 2. 定义信号参数 ---
# 信号 H1: 振幅 A1, 相位 theta1 (弧度)
A1 = 1.5
theta1_deg = 30
theta1_rad = np.deg2rad(theta1_deg)
H1 = A1 * np.exp(1j * theta1_rad)

# 信号 H2: 振幅 A2, 相位 theta2 (弧度)
A2 = 1.5
theta2_deg = 60
theta2_rad = np.deg2rad(theta2_deg)
H2 = A2 * np.exp(1j * theta2_rad)

print(H2)

# --- 3. 创建 3D 图 ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d') # 创建一个3D子图

# 设置轴标签
ax.set_xlabel("实部（I）")
ax.set_ylabel("虚部（Q）")
ax.set_zlabel("频率轴")
# ax.set_title(r"不同频率信号相量叠加 $H_1 \oplus H_2$")

# --- 4. 绘制 3D 向量 ---
# 向量 H1: 从原点 (0,0,0) 到 (H1.real, H1.imag, 0)
ax.quiver(0, 0, 0,         # 起始点 (x,y,z)
          H1.real, H1.imag, 0, # 向量分量 (dx,dy,dz)
          color='blue', arrow_length_ratio=0.1, label=r'$H_{1} = A e^{j (2 \pi f_1t + \theta_1)}$', linewidth=2)

# 向量 H2: 从 H1 终点 (H1.real, H1.imag, 0) 到 (H1.real+H2.real, H1.imag+H2.imag, 0)
ax.quiver(0, 0, 0.1, # 起始点
          H2.real, H2.imag, 0.1, # 向量分量
          color='green',
          arrow_length_ratio=0.1, label=r'$H_{2} = A e^{j (2 \pi f_2t + \theta_2)}$', linewidth=2)

# --- 5. 添加文本标注 ---
# 标注 H1
# noinspection PyTypeChecker
ax.text(H1.real * 0.5, H1.imag * 0.5, 0.02, r'$H_{1}$', color='blue', fontsize=20)
# 标注 H2 (在 H1 终点附近)
# noinspection PyTypeChecker
ax.text(H2.real * 0.8, H2.imag * 0.8, 0.24, r'$H_{2}$', color='green', fontsize=20)

# --- 6. 设置轴的范围 (确保所有向量可见) ---
# 需要根据 H1, H2, H3 的最大分量来设置
ax.set_xlim((0, 1.5))
ax.set_ylim((0, 1.5))
ax.set_zlim((0, 0.5)) # Z轴从0开始，并确保有一定高度

# --- 7. 添加图例 ---
ax.legend(loc='upper left')

# --- 8. 隐藏刻度 (可选，但3D图通常会保留一部分刻度来提供方向感) ---
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([]) # 隐藏Z轴刻度
ax.invert_yaxis()

# 旋转视角，可以更好地观察3D效果
ax.view_init(elev=20, azim=-45) # elev 仰角，azim 方位角

# 保存
dirname = '../../chap02/'
file_name = 'vector3D.pdf'
plt.savefig(os.path.join(dirname, file_name),
            format='pdf',          # 明确指定保存格式为PDF
            # bbox_inches='tight',   # 自动裁剪掉图像周围多余的空白区域
            dpi=300)               # 对于矢量图（PDF/SVG）来说 DPI 影响不大，但可以保持习惯

plt.show()