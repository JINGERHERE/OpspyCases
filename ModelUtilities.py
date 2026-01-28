#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：ModelUtilities.py
@Date    ：2026/01/26 19:40:43
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.font_manager import FontProperties

from typing import Union, Optional, Literal, List

import opstool as opst
import ops_utilities as opsu
from ops_utilities.post import AdjustPlot

"""
# --------------------------------------------------
# ========== < ModelUtilities > ==========
# --------------------------------------------------
"""

# 全局单位系统
UNIT = opst.pre.UnitSystem(
    length='m', force='kn', time='sec'
    )

# 全局模型管理器
MM = opsu.pre.ModelManager(include_start=False)

"===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
# 数据后处理方法
class PostProcess:
    
    def __init__(self, manager: opsu.pre.ModelManager, label: str):
        
        self.MM = manager.get_tag(category='uniaxialMaterial')
        self.ele_tag = manager.get_tag(category='element', label=label)[0]
        self.strain_stages = manager.get_param(category='section', label=label, key='strain_stages')
        
        self.SMS = opsu.post.SecMatStates(odb_tag=label, ele_tag=self.ele_tag, integ=1)
    
    
    def get_fiber_resp(self, mat_tag: int, data_type: Literal['Strains', 'Stresses']):
        
        """使用 ModelManager.get_tag(category='uniaxialMaterial') 获取材料标签列表"""
        
        return self.SMS.get_data_mat(mat_tag=mat_tag, data_type=data_type)
    
    
    def get_sec_resp(
        self,
        data_type: Union[
            Literal['all'],
            Literal['secDefo', 'secForce'],
            List[Literal['secDefo', 'secForce']]
            ],
        dofs: Literal['P', 'Mz', 'My', 'T']
        ):
        
        """使用 ModelManager.get_tag(category='uniaxialMaterial') 获取材料标签列表"""
        
        return self.SMS.get_data(data_type=data_type, dofs=dofs)
    
    
    # def determine_damage(self, odb_tag: Union[int, str]):
        
    #     sec_damage = self.SMS.get_combined_steps_mat(
    #         mat_config=self.strain_stages,
    #         data_type='Strains',
    #         warn=True
    #         )
    
    def equivalent_bilinear(self):
        '''直接在这里判断损伤'''
        sec_damage = self.SMS.get_combined_steps_mat(
            mat_config=self.strain_stages,
            data_type='Strains',
            warn=True
            )
    



"===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
# 绘图方法
class PlotyHub:
    
    # 全局长度单位转换为：英寸
    UNIT = opst.pre.UnitSystem(length='inch')
    
    # ----- ----- ----- ----- -----
    # / 内部方法 / < 坐标轴 > 调整方法
    # ----- ----- ----- ----- -----
    @classmethod
    def _adjust_ax(
        cls, ax: Axes,
        title: str = '', title_size: int = 18,
        title_x: float = 0.5, title_y: float = -0.25,
        
        lower_label: str = '', left_label: str = '',
        label_size: int = 20, label_pad: float = 7.,
        
        lower_ticks: bool = True, left_ticks: bool = True,
        ticks_size: int = 18, ticks_pad: float = 12.,

        ) -> Axes:
        
        """
        调整坐标轴内的标题、背景、轴线刻度、标签、数字等。
        
        Args:
            ax (Axes): 要调整的 Axes 对象。
            title (str): 坐标轴标题。`默认值：`""。
            title_size (int): 坐标轴标题字体大小。`默认值：`18。
            title_x (float): 坐标轴标题水平位置。`默认值：`0.5。
            title_y (float): 坐标轴标题垂直位置。`默认值：`-0.25。
            
            lower_label (str): 下 x轴 标签。`默认值：`""。
            left_label (str): 左 y轴 标签。`默认值：`""。
            label_size (int): 坐标轴标签字体大小。`默认值：`20。
            label_pad (float): 坐标轴标签与轴线的距离。`默认值：`7.0。
            
            lower_ticks (bool): 是否显示下 x轴 刻度。`默认值：`True。
            left_ticks (bool): 是否显示左 y轴 刻度。`默认值：`True。
            ticks_size (int): 坐标轴刻度字体大小。`默认值：`18。
            ticks_pad (float): 坐标轴刻度与轴线的距离。`默认值：`12.0。
        
        Returns:
            Axes: 返回调整后的 Axes 对象。
        
        """
        
        # 坐标系 标题
        if not title:
            title = ax.get_title()
        ax.set_title(title, fontsize=title_size, color='black', x=title_x, y=title_y)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 背景调整
        ax = AdjustPlot.adjust_bg(ax)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 坐标轴调整
        AdjustPlot.adjust_x(ax, lower_label, label_size, label_pad, lower_ticks, ticks_size, ticks_pad)
        AdjustPlot.adjust_y(ax, left_label, label_size, label_pad, left_ticks, ticks_size, ticks_pad)
        
        # "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # # 共享 下x轴
        # y_right = ax.twinx()
        # cls._adjust_y(y_right, right_label, label_size, label_pad, right_ticks, ticks_size, ticks_pad)
        # # 共享 左y轴
        # x_upper = ax.twiny()
        # cls._adjust_x(x_upper, upper_label, label_size, label_pad, upper_ticks, ticks_size, ticks_pad)
        '''返回的共享轴需要有数据，否则 ticks 范围只有 [0, 1]'''
        
        return ax
    
    # ----- ----- ----- ----- -----
    # < 单坐标轴 > 绘图参数调整方法
    # ----- ----- ----- ----- -----
    @classmethod
    def adjust_single(cls, fig: Figure, ax: Axes, leg: Optional[Legend] = None) -> Figure:
        
        """
        单坐标轴绘图参数调整方法。
        
        Args:
            fig (Figure): 输入的 matplotlib.figure.Figure 对象。
            ax (Axes): 输入的 matplotlib.axes.Axes 对象。
            leg (Legend, optional): 输入的 `leg = ax.legend()` 对象。
        
        Returns:
            Figure: 返回调整后的 matplotlib.figure.Figure 对象。
        
        """
        
        # 画布参数
        fig_params = dict(
            # 创建画布（将厘米转换为英寸：1厘米=0.3937英寸）
            fig_width = 27 * cls.UNIT.cm, # 单位厘米
            fig_height = 18 * cls.UNIT.cm,
            # 子图间距
            w_space = 5.4 * cls.UNIT.cm, h_space = 6.3 * cls.UNIT.cm,
            # 边缘距离
            left = 3.5 * cls.UNIT.cm, right = 1.5 * cls.UNIT.cm,
            top = 1.5 * cls.UNIT.cm, bottom = 2.5 * cls.UNIT.cm,
            )
        # 画布调整
        fig_adj = AdjustPlot.adjust_fig(fig, **fig_params)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 坐标轴 参数
        ax_params = {
            'title_size': 18, 'title_x': 0.5, 'title_y': -0.25,
            'label_size': 20, 'label_pad': 7.,
            'ticks_size': 18, 'ticks_pad': 12.,
            }
        # 坐标轴参数调整
        cls._adjust_ax(ax, **ax_params)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 图例参数
        leg_size = 20
        # 图例
        if leg is not None:
            AdjustPlot.adjust_leg(leg, font_size=leg_size)
        
        return fig_adj
    
    # ----- ----- ----- ----- -----
    # < 双坐标轴 > 绘图参数调整方法
    # ----- ----- ----- ----- -----
    @classmethod
    def adjust_dual(
        cls, fig: Figure,
        ax1: Axes, ax2: Axes,
        leg1: Optional[Legend] = None, leg2: Optional[Legend] = None
        ) -> Figure:
        """
        双坐标轴绘图参数调整方法。
        
        Args:
            fig (Figure): 输入的 matplotlib.figure.Figure 对象。
            ax1 (Axes): 输入的 matplotlib.axes.Axes 对象。
            ax2 (Axes): 输入的 matplotlib.axes.Axes 对象。
            leg1 (Legend, optional): 输入的 `leg1 = ax1.legend()` 对象。
            leg2 (Legend, optional): 输入的 `leg2 = ax2.legend()` 对象。
        
        Returns:
            Figure: 返回调整后的 matplotlib.figure.Figure 对象。
        
        """
        fig_params = {
            # 创建画布（将厘米转换为英寸：1厘米=0.3937英寸）
            'fig_width': 2 * 27 * cls.UNIT.cm, # 单位厘米
            'fig_height': 18 * cls.UNIT.cm,
            # 子图间距
            'w_space': 15.3 * cls.UNIT.cm, 'h_space': 6.3 * cls.UNIT.cm,
            # 边缘距离
            'left': 4.5 * cls.UNIT.cm, 'right': 1.5 * cls.UNIT.cm,
            'top': 1.5 * cls.UNIT.cm, 'bottom': 5.5 * cls.UNIT.cm,
            }
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 画布参数调整
        fig_adj = AdjustPlot.adjust_fig(fig, **fig_params)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 坐标轴 参数
        ax_params = {
            'title_size': 38, 'title_x': 0.5, 'title_y': -0.4,
            'label_size': 36, 'label_pad': 9.,
            'ticks_size': 25, 'ticks_pad': 12.,
            }
        # 坐标轴1 调整
        cls._adjust_ax(ax1, **ax_params)
        # 坐标轴2 调整
        cls._adjust_ax(ax2, **ax_params)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 图例参数
        leg_size = 30
        # 图例调整
        if leg1 is not None:
            AdjustPlot.adjust_leg(leg1, font_size=leg_size)
        if leg2 is not None:
            AdjustPlot.adjust_leg(leg2, font_size=leg_size)
        
        return fig_adj

    @classmethod
    def _annotate(cls, ax: Axes):
        
        arrowprops = dict(
            arrowstyle='->',  # 使用箭头 'fancy', '->'
            linewidth=2,  # 线宽
            mutation_scale=20,  # 箭头大小
            color='k',  # 箭头颜色
            connectionstyle='arc3,rad=-0.2',  # 带弧度的箭头
            )
        
        # 注释箭头
        ax.annotate(
            text='center point',  # 标签
            color='k',  # 标签颜色
            xy=(0., 0.),  # 要标记的数据坐标
            xycoords='data',  # 'data' 使用被注释对象（参数为xy）的坐标系统
            xytext=(2.0, 0.5),  # 标签偏移量
            textcoords='data',  # 'offset points' 以点为单位的偏移量
            fontsize=16,  # 字体大小
            arrowprops=arrowprops,
            zorder=5
            )

"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""

if __name__ == "__main__":
    
    print(f'全局单位：\n{UNIT}\n')
    print(f'全局模型管理器：\n{MM}\n')
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    import matplotlib.pyplot as plt
    
    plt.close('all')
    # 画布
    fig = plt.figure(dpi=100)
    fig.suptitle('Example Plot', fontsize=22, color='black', x=0.5, y=0.98, alpha=1.0)
    # 坐标轴
    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1, facecolor='none')
    ax.plot([1, 2, 3, 4, 5], [6, -2, 3, -4, 5], label='Line 1')
    ax.set_xlim(0, 6)
    ax.set_ylim(-7, 7)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    # 图例
    leg = ax.legend(
        loc='lower right',
        bbox_to_anchor=(1.0, 0.0), labelcolor=(0, 0, 0, 1),
        frameon=True, fancybox=True, shadow=False,
        )
    # 调整
    adj_fig = PlotyHub.adjust_single(fig, ax, leg)
    
    # plt.show()
    adj_fig.savefig('./ExamplePlot_single.png', dpi=320)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    plt.close('all')
    # 画布
    fig = plt.figure(dpi=100)
    # fig.suptitle('Example Plot', fontsize=22, color='black', x=0.5, y=0.98, alpha=1.0)
    # 坐标轴
    ax1 = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1, facecolor='none')
    ax2 = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1, facecolor='none')
    
    ax1.set_title('(a) Subplot 1')
    ax1.plot([1, 2, 3, 4, 5], [6, -2, 3, -4, 5], label='Line 1')
    ax1.set_xlim(0, 6)
    ax1.set_ylim(-7, 7)
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    leg1 = ax1.legend(
        loc='lower right',
        bbox_to_anchor=(1.0, 0.0), labelcolor=(0, 0, 0, 1),
        frameon=True, fancybox=True, shadow=False,
        )
    
    ax2.set_title('(b) Subplot 2')
    ax2.plot([1, 2, 3, 4, 5], [-6, 2, -3, 4, -5], label='Line 2')
    ax2.set_xlim(0, 6)
    ax2.set_ylim(-7, 7)
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    leg2 = ax2.legend(
        loc='lower right',
        bbox_to_anchor=(1.0, 0.0), labelcolor=(0, 0, 0, 1),
        frameon=True, fancybox=True, shadow=False,
        )
    
    # 调整
    adj_fig = PlotyHub.adjust_dual(fig, ax1, ax2, leg1, leg2)
    
    # plt.show()
    adj_fig.savefig('./ExamplePlot_dual.png', dpi=320)

