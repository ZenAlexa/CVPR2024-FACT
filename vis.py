import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

def visualize_loss_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # 定义颜色方案
    colors = {
        'frame': '#FF6B6B',
        'seg': '#4ECDC4',
        'token': '#45B7D1',
        'attention': '#96CEB4',
        'smooth': '#FECA57',
        'total': '#5F27CD'
    }
    
    # FACT模型主要组件
    # 帧级特征
    frame_box = FancyBboxPatch((1, 7), 2, 1.5, 
                               boxstyle="round,pad=0.1",
                               facecolor=colors['frame'], 
                               edgecolor='black',
                               alpha=0.7)
    ax.add_patch(frame_box)
    ax.text(2, 7.75, 'Frame Features\n(T×D)', ha='center', va='center', fontsize=10, weight='bold')
    
    # 段级特征(TDU)
    seg_box = FancyBboxPatch((4.5, 7), 2, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['seg'],
                             edgecolor='black',
                             alpha=0.7)
    ax.add_patch(seg_box)
    ax.text(5.5, 7.75, 'Segment Features\n(S×D)', ha='center', va='center', fontsize=10, weight='bold')
    
    # 动作令牌
    token_box = FancyBboxPatch((8, 7), 2, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['token'],
                               edgecolor='black',
                               alpha=0.7)
    ax.add_patch(token_box)
    ax.text(9, 7.75, 'Action Tokens\n(N×D)', ha='center', va='center', fontsize=10, weight='bold')
    
    # 交叉注意力
    # F2A
    f2a_arrow = FancyArrowPatch((5.5, 6.5), (8.5, 6.5),
                                connectionstyle="arc3,rad=.3",
                                arrowstyle='->,head_width=.4,head_length=.4',
                                color=colors['attention'],
                                linewidth=3)
    ax.add_patch(f2a_arrow)
    ax.text(7, 6, r'$\mathcal{L}_{f2a}$', ha='center', fontsize=12, weight='bold')
    
    # A2F
    a2f_arrow = FancyArrowPatch((8.5, 5.5), (5.5, 5.5),
                                connectionstyle="arc3,rad=-.3",
                                arrowstyle='->,head_width=.4,head_length=.4',
                                color=colors['attention'],
                                linewidth=3)
    ax.add_patch(a2f_arrow)
    ax.text(7, 5, r'$\mathcal{L}_{a2f}$', ha='center', fontsize=12, weight='bold')
    
    # 各损失函数框
    y_loss = 3.5
    
    # 帧级损失
    loss_frame = Rectangle((0.5, y_loss), 2, 0.8, 
                          facecolor=colors['frame'], 
                          alpha=0.5,
                          edgecolor='black')
    ax.add_patch(loss_frame)
    ax.text(1.5, y_loss+0.4, r'$\mathcal{L}_{frame}$', ha='center', va='center', fontsize=11)
    ax.text(1.5, y_loss-0.3, r'CE($y_{frame}, \hat{y}_{frame}$)', ha='center', va='center', fontsize=9)
    
    # 段级损失
    loss_seg = Rectangle((3, y_loss), 2, 0.8,
                        facecolor=colors['seg'],
                        alpha=0.5,
                        edgecolor='black')
    ax.add_patch(loss_seg)
    ax.text(4, y_loss+0.4, r'$\mathcal{L}_{seg}$', ha='center', va='center', fontsize=11)
    ax.text(4, y_loss-0.3, r'CE($y_{seg}, \hat{y}_{seg}$)', ha='center', va='center', fontsize=9)
    
    # 令牌损失
    loss_token = Rectangle((5.5, y_loss), 2.5, 0.8,
                          facecolor=colors['token'],
                          alpha=0.5,
                          edgecolor='black')
    ax.add_patch(loss_token)
    ax.text(6.75, y_loss+0.4, r'$\mathcal{L}_{token}$', ha='center', va='center', fontsize=11)
    ax.text(6.75, y_loss-0.3, 'Hungarian-CE', ha='center', va='center', fontsize=9)
    
    # 平滑损失
    loss_smooth = Rectangle((8.5, y_loss), 2, 0.8,
                           facecolor=colors['smooth'],
                           alpha=0.5,
                           edgecolor='black')
    ax.add_patch(loss_smooth)
    ax.text(9.5, y_loss+0.4, r'$\mathcal{L}_{smooth}$', ha='center', va='center', fontsize=11)
    ax.text(9.5, y_loss-0.3, r'$\sum_t ||p_t - p_{t-1}||_1$', ha='center', va='center', fontsize=9)
    
    # 权重系数
    weights_y = 2.3
    for i, (name, weight) in enumerate([
        ('λ_f', 1), ('λ_s', 1), ('λ_t', 1), 
        ('λ_{f2a}', 1), ('λ_{a2f}', 1), ('λ_{sm}', 5)
    ]):
        ax.text(1 + i*1.8, weights_y, f'{name}={weight}', 
                ha='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    # 总损失
    total_loss = FancyBboxPatch((3, 0.5), 5, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['total'],
                                edgecolor='black',
                                alpha=0.8)
    ax.add_patch(total_loss)
    ax.text(5.5, 1, r'$\mathcal{L} = \sum_i \lambda_i \mathcal{L}_i$', 
            ha='center', va='center', fontsize=13, weight='bold', color='white')
    
    # 连接线
    for x in [1.5, 4, 6.75, 9.5]:
        ax.plot([x, 5.5], [y_loss, 1.5], 'k--', alpha=0.3)
    
    # 设置
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_title('FACT Loss Function Architecture', fontsize=16, weight='bold', pad=20)
    
    plt.tight_layout()
    return fig

# 生成图表
fig1 = visualize_loss_architecture()
plt.show()