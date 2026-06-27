#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
电磁波极化波 3D 动态仿真模型
====================================
支持五种极化状态：
  1. 线极化波 (Linearly Polarized)
  2. 右旋圆极化波 (Right-Hand Circularly Polarized)
  3. 左旋圆极化波 (Left-Hand Circularly Polarized)
  4. 右旋椭圆极化波 (Right-Hand Elliptically Polarized)
  5. 左旋椭圆极化波 (Left-Hand Elliptically Polarized)

物理约定：
  - 波沿 +z 方向传播
  - 电场在 x-y 平面内振动
  - 旋向判断：沿传播方向(+z)观察，顺时针为右旋，逆时针为左旋 (IEEE标准)
  - 相位差 δ = φ_y - φ_x
      δ < 0 -> 右旋
      δ > 0 -> 左旋
      δ = 0 或 π -> 线极化

运行环境：Python 3.8+, matplotlib, numpy
"""

import os
import sys
if sys.platform == 'win32':
    import ctypes
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except:
        pass
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, Slider
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
#  3D 箭头类（用于绘制电场矢量）
# ============================================================
class Arrow3D(FancyArrowPatch):
    """3D 空间中的箭头绘制类"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

    def set_data(self, xs, ys, zs):
        self._verts3d = xs, ys, zs


# ============================================================
#  极化波仿真主类
# ============================================================
class PolarizationWaveSimulator:
    """电磁波极化状态 3D 动态仿真器"""

    def __init__(self):
        # ---- 物理参数 ----
        self.Ex0 = 2.0       # x 分量振幅
        self.Ey0 = 2.0       # y 分量振幅
        self.delta = -np.pi/2  # 相位差 δ = φ_y - φ_x (初始：右旋圆极化)
        self.k = 1.0         # 波数
        self.omega = 1.0     # 角频率
        self.z_obs = 3.0     # 观察平面位置
        self.t = 0.0         # 当前时间

        # ---- 空间范围 ----
        self.z_min = 0.0
        self.z_max = 6.0
        self.z_num = 200

        # ---- 动画参数 ----
        self.dt = 0.05       # 时间步长
        self.animation_speed = 1.0

        # ---- 创建图形界面 ----
        self._setup_figure()

    # --------------------------------------------------------
    #  界面搭建
    # --------------------------------------------------------
    def _setup_figure(self):
        """创建图形窗口与控件"""
        self.fig = plt.figure(figsize=(14, 9), facecolor='#f5f5f5')
        self.fig.suptitle('电磁波极化状态 3D 动态仿真', fontsize=16, fontweight='bold', y=0.98)

        # 3D 主图区域
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_position([0.05, 0.20, 0.65, 0.72])
        self.ax.set_facecolor('#fafafa')

        # ---- 按钮区域（右侧）----
        button_y_start = 0.85
        button_height = 0.06
        button_gap = 0.015
        button_left = 0.78
        button_width = 0.18

        self.buttons = {}
        button_configs = [
            ('line', '线极化', '#4CAF50'),
            ('rhcp', '右旋圆极化', '#2196F3'),
            ('lhcp', '左旋圆极化', '#9C27B0'),
            ('rhep', '右旋椭圆极化', '#FF9800'),
            ('lhep', '左旋椭圆极化', '#F44336'),
        ]

        for i, (key, label, color) in enumerate(button_configs):
            ax_btn = self.fig.add_axes([button_left, button_y_start - i*(button_height+button_gap),
                                        button_width, button_height])
            btn = Button(ax_btn, label, color=color, hovercolor='#cccccc')
            btn.label.set_color('white')
            btn.label.set_fontweight('bold')
            btn.on_clicked(lambda event, k=key: self._set_preset(k))
            self.buttons[key] = btn

        # ---- 滑动条区域（底部）----
        slider_left = 0.12
        slider_width = 0.55
        slider_bottom = 0.12
        slider_height = 0.03
        slider_gap = 0.012

        # 振幅比滑动条
        self.ax_amp = self.fig.add_axes([slider_left, slider_bottom + 3*(slider_height+slider_gap),
                                         slider_width, slider_height])
        self.slider_amp = Slider(self.ax_amp, '振幅比 Ey0/Ex0', 0.2, 2.0,
                                  valinit=1.0, valstep=0.05, color='#FF5722')
        self.slider_amp.on_changed(self._update_amp_ratio)

        # 相位差滑动条
        self.ax_delta = self.fig.add_axes([slider_left, slider_bottom + 2*(slider_height+slider_gap),
                                           slider_width, slider_height])
        self.slider_delta = Slider(self.ax_delta, '相位差 δ (°)', -180, 180,
                                    valinit=-90, valstep=5, color='#3F51B5')
        self.slider_delta.on_changed(self._update_delta)

        # 观察平面滑动条
        self.ax_zobs = self.fig.add_axes([slider_left, slider_bottom + 1*(slider_height+slider_gap),
                                          slider_width, slider_height])
        self.slider_zobs = Slider(self.ax_zobs, '观察平面 z0', 0, 6,
                                   valinit=3.0, valstep=0.1, color='#009688')
        self.slider_zobs.on_changed(self._update_zobs)

        # 动画速度滑动条
        self.ax_speed = self.fig.add_axes([slider_left, slider_bottom,
                                           slider_width, slider_height])
        self.slider_speed = Slider(self.ax_speed, '动画速度', 0.1, 3.0,
                                    valinit=1.0, valstep=0.1, color='#795548')
        self.slider_speed.on_changed(self._update_speed)

        # ---- 信息文本 ----
        self.info_text = self.fig.text(0.78, 0.45, '', fontsize=11,
                                       verticalalignment='top',
                                       bbox=dict(boxstyle='round,pad=0.5',
                                                 facecolor='#fff9c4', alpha=0.9))

        # ---- 初始化 3D 轴 ----
        self._setup_3d_axes()

        # ---- 创建图形对象 ----
        self._create_plot_objects()

        # ---- 初始化动画 ----
        self.anim = FuncAnimation(self.fig, self._animate, frames=1000,
                                  interval=30, blit=False)

    def _setup_3d_axes(self):
        """设置 3D 坐标轴"""
        self.ax.set_xlabel('z (传播方向)', fontsize=11, labelpad=10)
        self.ax.set_ylabel('E_x', fontsize=11, labelpad=10)
        self.ax.set_zlabel('E_y', fontsize=11, labelpad=10)

        self.ax.set_xlim(self.z_min, self.z_max)
        self.ax.set_ylim(-3.5, 3.5)
        self.ax.set_zlim(-3.5, 3.5)

        # 设置视角
        self.ax.view_init(elev=20, azim=-65)

        # 网格
        self.ax.grid(True, alpha=0.3)

    def _create_plot_objects(self):
        """创建所有图形对象"""
        z = np.linspace(self.z_min, self.z_max, self.z_num)

        # ---- 电场 x 分量波形（红色）----
        ex = self.Ex0 * np.cos(self.omega * self.t - self.k * z)
        self.line_ex, = self.ax.plot(z, ex, np.zeros_like(z),
                                      color='#e53935', linewidth=2.5, label='E_x 分量')

        # ---- 电场 y 分量波形（绿色）----
        ey = self.Ey0 * np.cos(self.omega * self.t - self.k * z + self.delta)
        self.line_ey, = self.ax.plot(z, np.zeros_like(z), ey,
                                      color='#43a047', linewidth=2.5, label='E_y 分量')

        # ---- 合成电场波形（蓝色）----
        self.line_total, = self.ax.plot(z, ex, ey,
                                         color='#1e88e5', linewidth=2,
                                         label='合成 E 矢量', alpha=0.7)

        # ---- 观察平面（半透明灰色）----
        xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 2), np.linspace(-3.5, 3.5, 2))
        zz = np.full_like(xx, self.z_obs)
        self.obs_plane = self.ax.plot_surface(zz.T, xx, yy,
                                               alpha=0.15, color='#9e9e9e',
                                               shade=False)

        # ---- 极化轨迹（橙色，在观察平面上）----
        tau = np.linspace(0, 2*np.pi, 100)
        locus_ex = self.Ex0 * np.cos(self.omega * tau - self.k * self.z_obs)
        locus_ey = self.Ey0 * np.cos(self.omega * tau - self.k * self.z_obs + self.delta)
        self.locus_line, = self.ax.plot(np.full_like(tau, self.z_obs), locus_ex, locus_ey,
                                         color='#fb8c00', linewidth=2.5,
                                         label='极化轨迹')

        # ---- 电场矢量（蓝色箭头）----
        ex_t = self.Ex0 * np.cos(self.omega * self.t - self.k * self.z_obs)
        ey_t = self.Ey0 * np.cos(self.omega * self.t - self.k * self.z_obs + self.delta)
        self.e_arrow = Arrow3D([self.z_obs, self.z_obs], [0, ex_t], [0, ey_t],
                                mutation_scale=20, lw=3, arrowstyle="-|>",
                                color='#1565c0')
        self.ax.add_artist(self.e_arrow)

        # ---- 矢量端点（蓝色小球）----
        self.end_point, = self.ax.plot([self.z_obs], [ex_t], [ey_t],
                                        'o', color='#0d47a1', markersize=10, zorder=10)

        # ---- 图例 ----
        self.ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    # --------------------------------------------------------
    #  预设极化模式
    # --------------------------------------------------------
    def _set_preset(self, mode):
        """设置预设极化模式"""
        presets = {
            'line':  {'Ex0': 2.0, 'Ey0': 2.0, 'delta': 0.0},
            'rhcp':  {'Ex0': 2.0, 'Ey0': 2.0, 'delta': -np.pi/2},
            'lhcp':  {'Ex0': 2.0, 'Ey0': 2.0, 'delta': np.pi/2},
            'rhep':  {'Ex0': 2.0, 'Ey0': 1.2, 'delta': -np.pi/3},
            'lhep':  {'Ex0': 2.0, 'Ey0': 1.2, 'delta': np.pi/3},
        }

        if mode in presets:
            p = presets[mode]
            self.Ex0 = p['Ex0']
            self.Ey0 = p['Ey0']
            self.delta = p['delta']

            # 同步滑动条
            self.slider_amp.set_val(self.Ey0 / self.Ex0)
            self.slider_delta.set_val(np.degrees(self.delta))

    # --------------------------------------------------------
    #  滑动条回调
    # --------------------------------------------------------
    def _update_amp_ratio(self, val):
        ratio = val
        self.Ey0 = self.Ex0 * ratio

    def _update_delta(self, val):
        self.delta = np.radians(val)

    def _update_zobs(self, val):
        self.z_obs = val
        # 更新观察平面
        self._update_obs_plane()

    def _update_speed(self, val):
        self.animation_speed = val

    def _update_obs_plane(self):
        """更新观察平面位置"""
        self.obs_plane.remove()
        xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 2), np.linspace(-3.5, 3.5, 2))
        zz = np.full_like(xx, self.z_obs)
        self.obs_plane = self.ax.plot_surface(zz.T, xx, yy,
                                               alpha=0.15, color='#9e9e9e',
                                               shade=False)

    # --------------------------------------------------------
    #  极化类型判断
    # --------------------------------------------------------
    def _get_polarization_type(self):
        """判断当前极化类型"""
        ratio = self.Ey0 / self.Ex0 if self.Ex0 > 0 else 0
        delta_deg = np.degrees(self.delta) % 360
        if delta_deg > 180:
            delta_deg -= 360

        # 线极化判断
        if abs(delta_deg) < 2 or abs(abs(delta_deg) - 180) < 2:
            return '线极化', '*'

        # 圆极化判断
        is_circular = abs(ratio - 1.0) < 0.05 and abs(abs(delta_deg) - 90) < 5

        # 旋向判断（沿 +z 观察）
        if delta_deg < 0:
            hand = '右旋'
        else:
            hand = '左旋'

        if is_circular:
            return f'{hand}圆极化', 'o'
        else:
            return f'{hand}椭圆极化', '@'

    # --------------------------------------------------------
    #  更新信息文本
    # --------------------------------------------------------
    def _update_info_text(self):
        """更新信息显示文本"""
        ptype, icon = self._get_polarization_type()

        info = f"""
  +---------------------+
  |  {icon}  {ptype:^10s}  |
  +---------------------+
  |  E_x0 = {self.Ex0:.2f}            |
  |  E_y0 = {self.Ey0:.2f}            |
  |  振幅比 = {self.Ey0/self.Ex0:.2f}       |
  |  相位差 d = {np.degrees(self.delta):.0f} deg      |
  |  观察 z0 = {self.z_obs:.1f}        |
  |  时间 t = {self.t:.2f} s        |
  +---------------------+

  旋向说明（沿+z方向看）：
    d < 0 -> 顺时针 -> 右旋
    d > 0 -> 逆时针 -> 左旋
        """
        self.info_text.set_text(info)

    # --------------------------------------------------------
    #  动画帧更新
    # --------------------------------------------------------
    def _animate(self, frame):
        """每一帧的更新函数"""
        # 更新时间
        self.t += self.dt * self.animation_speed

        z = np.linspace(self.z_min, self.z_max, self.z_num)

        # ---- 更新电场 x 分量波形 ----
        ex = self.Ex0 * np.cos(self.omega * self.t - self.k * z)
        self.line_ex.set_data(z, ex)
        self.line_ex.set_3d_properties(np.zeros_like(z))

        # ---- 更新电场 y 分量波形 ----
        ey = self.Ey0 * np.cos(self.omega * self.t - self.k * z + self.delta)
        self.line_ey.set_data(z, np.zeros_like(z))
        self.line_ey.set_3d_properties(ey)

        # ---- 更新合成电场波形 ----
        self.line_total.set_data(z, ex)
        self.line_total.set_3d_properties(ey)

        # ---- 更新极化轨迹 ----
        tau = np.linspace(0, 2*np.pi, 100)
        locus_ex = self.Ex0 * np.cos(self.omega * tau - self.k * self.z_obs)
        locus_ey = self.Ey0 * np.cos(self.omega * tau - self.k * self.z_obs + self.delta)
        self.locus_line.set_data(np.full_like(tau, self.z_obs), locus_ex)
        self.locus_line.set_3d_properties(locus_ey)

        # ---- 更新电场矢量 ----
        ex_t = self.Ex0 * np.cos(self.omega * self.t - self.k * self.z_obs)
        ey_t = self.Ey0 * np.cos(self.omega * self.t - self.k * self.z_obs + self.delta)
        self.e_arrow.set_data([self.z_obs, self.z_obs], [0, ex_t], [0, ey_t])

        # ---- 更新矢量端点 ----
        self.end_point.set_data([self.z_obs], [ex_t])
        self.end_point.set_3d_properties([ey_t])

        # ---- 更新信息文本 ----
        self._update_info_text()

        return (self.line_ex, self.line_ey, self.line_total,
                self.locus_line, self.end_point)

    # --------------------------------------------------------
    #  运行
    # --------------------------------------------------------
    def run(self):
        """启动仿真"""
        self._update_info_text()
        print("=" * 60)
        print("电磁波极化状态 3D 动态仿真")
        print("=" * 60)
        print("操作说明：")
        print("* 点击右侧按钮切换预设极化模式")
        print("* 拖动底部滑动条调节参数")
        print("* 鼠标拖拽 3D 图可旋转视角")
        print("* 关闭窗口退出程序")
        print("=" * 60)
        plt.show()


# ============================================================
#  主程序入口
# ============================================================
if __name__ == '__main__':
    sim = PolarizationWaveSimulator()
    sim.run()