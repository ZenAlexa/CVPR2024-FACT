import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# Set matplotlib to support Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # Use SimHei font for Chinese
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

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
    ax.text(2, 7.75, '帧级特征\n(T×D)', ha='center', va='center', fontsize=10, weight='bold')
    
    # 段级特征(TDU)
    seg_box = FancyBboxPatch((4.5, 7), 2, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['seg'],
                             edgecolor='black',
                             alpha=0.7)
    ax.add_patch(seg_box)
    ax.text(5.5, 7.75, '段级特征\n(S×D)', ha='center', va='center', fontsize=10, weight='bold')
    
    # 动作令牌
    token_box = FancyBboxPatch((8, 7), 2, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['token'],
                               edgecolor='black',
                               alpha=0.7)
    ax.add_patch(token_box)
    ax.text(9, 7.75, '动作令牌\n(N×D)', ha='center', va='center', fontsize=10, weight='bold')
    
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
    ax.text(6.75, y_loss-0.3, '匈牙利算法-CE', ha='center', va='center', fontsize=9)
    
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
    ax.set_title('FACT损失函数架构', fontsize=16, weight='bold', pad=20)
    
    plt.tight_layout()
    return fig

# 生成图表
fig1 = visualize_loss_architecture()
plt.show()

def visualize_loss_weights():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 损失组成饼图
    losses = ['帧级', '段级', '令牌', 'F2A', 'A2F', '平滑']
    weights = [1, 1, 1, 1, 1, 5]
    total_weight = sum(weights)
    percentages = [w/total_weight * 100 for w in weights]
    
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA0DD', '#FECA57']
    
    wedges, texts, autotexts = ax1.pie(percentages, labels=losses, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90,
                                        wedgeprops=dict(width=0.5, edgecolor='white'))
    
    # 美化饼图
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
        autotext.set_fontsize(10)
    
    ax1.set_title('损失组件权重分布', fontsize=14, weight='bold')
    
    # 损失权重条形图
    x_pos = np.arange(len(losses))
    bars = ax2.bar(x_pos, weights, color=colors_pie, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'λ={weight}', ha='center', va='bottom', fontsize=11, weight='bold')
    
    ax2.set_xlabel('损失组件', fontsize=12)
    ax2.set_ylabel('权重值', fontsize=12)
    ax2.set_title('损失组件权重', fontsize=14, weight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(losses)
    ax2.set_ylim(0, 6)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

fig2 = visualize_loss_weights()
plt.show()

def visualize_loss_computation_flow():
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 定义节点位置
    nodes = {
        'input': (6, 9),
        'frame_feat': (3, 7),
        'seg_feat': (6, 7),
        'action_feat': (9, 7),
        'frame_pred': (3, 5),
        'seg_pred': (6, 5),
        'action_pred': (9, 5),
        'frame_loss': (3, 3),
        'seg_loss': (6, 3),
        'token_loss': (9, 3),
        'attention_loss': (6, 4),
        'smooth_loss': (1.5, 3),
        'total_loss': (6, 1)
    }
    
    # 绘制节点
    node_styles = {
        'input': {'color': '#E8E8E8', 'shape': 'box'},
        'feat': {'color': '#B8E6B8', 'shape': 'round'},
        'pred': {'color': '#FFE4B5', 'shape': 'round'},
        'loss': {'color': '#FFB6C1', 'shape': 'box'},
        'total': {'color': '#9370DB', 'shape': 'box'}
    }
    
    # 绘制输入
    ax.add_patch(Rectangle((nodes['input'][0]-1, nodes['input'][1]-0.4), 2, 0.8,
                          facecolor=node_styles['input']['color'],
                          edgecolor='black'))
    ax.text(nodes['input'][0], nodes['input'][1], '输入视频', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # 绘制特征节点
    feat_labels = {'frame_feat': '帧特征', 'seg_feat': '段特征', 'action_feat': '动作特征'}
    for feat in ['frame_feat', 'seg_feat', 'action_feat']:
        x, y = nodes[feat]
        ax.add_patch(FancyBboxPatch((x-1, y-0.4), 2, 0.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor=node_styles['feat']['color'],
                                   edgecolor='black'))
        ax.text(x, y, feat_labels[feat], ha='center', va='center', fontsize=10)
    
    # 绘制预测节点
    pred_labels = {'frame_pred': '帧预测', 'seg_pred': '段预测', 'action_pred': '动作预测'}
    for pred in ['frame_pred', 'seg_pred', 'action_pred']:
        x, y = nodes[pred]
        ax.add_patch(FancyBboxPatch((x-1, y-0.4), 2, 0.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor=node_styles['pred']['color'],
                                   edgecolor='black'))
        ax.text(x, y, pred_labels[pred], ha='center', va='center', fontsize=10)
    
    # 绘制损失节点
    loss_labels = {
        'frame_loss': r'$\mathcal{L}_{frame}$',
        'seg_loss': r'$\mathcal{L}_{seg}$',
        'token_loss': r'$\mathcal{L}_{token}$',
        'attention_loss': r'$\mathcal{L}_{f2a/a2f}$',
        'smooth_loss': r'$\mathcal{L}_{smooth}$'
    }
    
    for loss, label in loss_labels.items():
        x, y = nodes[loss]
        ax.add_patch(Rectangle((x-1, y-0.4), 2, 0.8,
                              facecolor=node_styles['loss']['color'],
                              edgecolor='black'))
        ax.text(x, y, label, ha='center', va='center', fontsize=11)
    
    # 绘制总损失
    x, y = nodes['total_loss']
    ax.add_patch(Rectangle((x-1.5, y-0.4), 3, 0.8,
                          facecolor=node_styles['total']['color'],
                          edgecolor='black', linewidth=2))
    ax.text(x, y, r'$\mathcal{L}_{total}$', ha='center', va='center', 
            fontsize=12, weight='bold', color='white')
    
    # 绘制连接线
    connections = [
        ('input', 'frame_feat'), ('input', 'seg_feat'), ('input', 'action_feat'),
        ('frame_feat', 'frame_pred'), ('seg_feat', 'seg_pred'), ('action_feat', 'action_pred'),
        ('frame_pred', 'frame_loss'), ('seg_pred', 'seg_loss'), ('action_pred', 'token_loss'),
        ('frame_feat', 'smooth_loss'),
        ('seg_feat', 'attention_loss'), ('action_feat', 'attention_loss'),
        ('frame_loss', 'total_loss'), ('seg_loss', 'total_loss'), 
        ('token_loss', 'total_loss'), ('attention_loss', 'total_loss'),
        ('smooth_loss', 'total_loss')
    ]
    
    for start, end in connections:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        ax.arrow(x1, y1-0.4, x2-x1, y2-y1+0.8, 
                head_width=0.15, head_length=0.1, 
                fc='gray', ec='gray', alpha=0.6)
    
    # 添加公式注释
    formulas = [
        (3, 2.3, r'CE($y$, $\hat{y}$)'),
        (6, 2.3, r'CE($y$, $\hat{y}$)'),
        (9, 2.3, '匈牙利算法'),
        (6, 3.3, r'KL($A$, $\hat{A}$)'),
        (1.5, 2.3, r'$||p_t - p_{t-1}||$')
    ]
    
    for x, y, formula in formulas:
        ax.text(x, y, formula, ha='center', va='center', 
                fontsize=9, style='italic',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('FACT模型损失计算流程', fontsize=16, weight='bold')
    
    plt.tight_layout()
    return fig

fig3 = visualize_loss_computation_flow()
plt.show()

def visualize_temporal_losses():
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # 生成示例数据
    time_steps = 50
    x = np.arange(time_steps)
    
    # 帧级预测概率（带噪声）
    true_segments = [0]*10 + [1]*15 + [2]*10 + [3]*15
    frame_probs = np.zeros((time_steps, 4))
    for t, seg in enumerate(true_segments):
        frame_probs[t, seg] = 0.8 + 0.2*np.random.rand()
        noise = 0.1*np.random.rand(4)
        noise[seg] = 0
        frame_probs[t] += noise
        frame_probs[t] /= frame_probs[t].sum()
    
    # 绘制帧级损失
    ax1 = axes[0]
    im1 = ax1.imshow(frame_probs.T, aspect='auto', cmap='RdYlBu_r')
    ax1.set_ylabel('类别', fontsize=11)
    ax1.set_title(r'帧级预测与 $\mathcal{L}_{frame}$', fontsize=12)
    ax1.set_yticks([0, 1, 2, 3])
    
    # 添加真实标签
    for t, seg in enumerate(true_segments):
        ax1.plot(t, seg, 'w*', markersize=3)
    
    # 绘制段级损失
    ax2 = axes[1]
    segment_boundaries = [0, 10, 25, 35, 50]
    segment_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i in range(len(segment_boundaries)-1):
        ax2.axvspan(segment_boundaries[i], segment_boundaries[i+1], 
                   alpha=0.3, color=segment_colors[i])
        ax2.text((segment_boundaries[i] + segment_boundaries[i+1])/2, 0.5,
                f'段 {i}', ha='center', va='center', fontsize=10, weight='bold')
    
    ax2.set_xlim(0, time_steps)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('段', fontsize=11)
    ax2.set_title(r'段级表示与 $\mathcal{L}_{seg}$', fontsize=12)
    ax2.set_yticks([])
    
    # 绘制平滑损失
    ax3 = axes[2]
    # 计算相邻帧差异
    frame_diff = np.sum(np.abs(frame_probs[1:] - frame_probs[:-1]), axis=1)
    ax3.plot(x[1:], frame_diff, 'k-', linewidth=2, label=r'$||p_t - p_{t-1}||_1$')
    ax3.fill_between(x[1:], 0, frame_diff, alpha=0.3, color='orange')
    
    # 标记大的变化点
    peaks = np.where(frame_diff > 0.5)[0]
    ax3.plot(peaks+1, frame_diff[peaks], 'ro', markersize=8, label='较大变化')
    
    ax3.set_xlabel('时间步', fontsize=11)
    ax3.set_ylabel('L1距离', fontsize=11)
    ax3.set_title(r'时序平滑性与 $\mathcal{L}_{smooth}$', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

fig4 = visualize_temporal_losses()
plt.show()

def create_loss_formula_cards():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # 定义每个损失的信息
    loss_info = [
        {
            'title': '帧级损失',
            'formula': r'$\mathcal{L}_{frame} = -\frac{1}{T}\sum_{t=1}^{T}\sum_{c=1}^{C} y_{t,c}\log(\hat{y}_{t,c})$',
            'desc': '帧预测的交叉熵',
            'color': '#FF6B6B'
        },
        {
            'title': '段级损失',
            'formula': r'$\mathcal{L}_{seg} = -\frac{1}{S}\sum_{s=1}^{S}\sum_{c=1}^{C} y_{s,c}\log(\hat{y}_{s,c})$',
            'desc': '段预测的交叉熵',
            'color': '#4ECDC4'
        },
        {
            'title': '令牌损失',
            'formula': r'$\mathcal{L}_{token} = \min_{\sigma}\sum_{i=1}^{N} \text{CE}(y_i, \hat{y}_{\sigma(i)})$',
            'desc': '匈牙利匹配 + 交叉熵',
            'color': '#45B7D1'
        },
        {
            'title': 'F2A注意力损失',
            'formula': r'$\mathcal{L}_{f2a} = D_{KL}(A_{target} || A_{f2a})$',
            'desc': '帧到动作注意力对齐',
            'color': '#96CEB4'
        },
        {
            'title': 'A2F注意力损失',
            'formula': r'$\mathcal{L}_{a2f} = D_{KL}(A_{target} || A_{a2f})$',
            'desc': '动作到帧注意力对齐',
            'color': '#DDA0DD'
        },
        {
            'title': '平滑性损失',
            'formula': r'$\mathcal{L}_{smooth} = \frac{1}{T-1}\sum_{t=2}^{T} ||p_t - p_{t-1}||_1$',
            'desc': '时序一致性约束',
            'color': '#FECA57'
        }
    ]
    
    for i, (ax, info) in enumerate(zip(axes, loss_info)):
        # 创建卡片背景
        card = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                             boxstyle="round,pad=0.05",
                             facecolor=info['color'],
                             alpha=0.3,
                             edgecolor='black',
                             linewidth=2)
        ax.add_patch(card)
        
        # 添加标题
        ax.text(0.5, 0.85, info['title'], 
                ha='center', va='center', fontsize=14, weight='bold',
                transform=ax.transAxes)
        
        # 添加公式
        ax.text(0.5, 0.5, info['formula'],
                ha='center', va='center', fontsize=11,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 添加描述
        ax.text(0.5, 0.15, info['desc'],
                ha='center', va='center', fontsize=9, style='italic',
                transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.suptitle('FACT模型损失函数', fontsize=16, weight='bold')
    plt.tight_layout()
    return fig

fig6 = create_loss_formula_cards()
plt.show()