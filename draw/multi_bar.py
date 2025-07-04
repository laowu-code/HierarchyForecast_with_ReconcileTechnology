import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = 'Arial'
# 假设你的4个sheet数据存储在Excel文件中
file_path = '../data_record/reconcile_metrics_all.xlsx'  # 修改为你的文件路径

# 读取4个sheet数据
sheets = ['A(Tibet_2018)', 'B(Tibet_2019)', 'C(USA_2019)', 'D(Australia_2019)']
data = {sheet: pd.read_excel(file_path, sheet_name=sheet) for sheet in sheets}

# 开始绘图
fig, axes = plt.subplots(4, 1, figsize=(20, 16), sharex=True)  # 2x2布局
axes = axes.flatten()  # 展平axes，方便索引
dataset=['Dataset A','Dataset B','Dataset C','Dataset D']
ksize=20
for idx, sheet_name in enumerate(sheets):
    all_handles = []
    all_labels = []
    df = data[sheet_name]
    models = df.iloc[1:, 0] + "-" + df.iloc[1:, 1]  # 获取模型名称
    models=[x[:-5] if 'Base' in x else x for x in models]
    models=[str(x) for x in models]
    x = np.arange(len(models))  # x轴位置

    # 提取指标
    node_mae = df.iloc[1:, 2]
    node_rmse = df.iloc[1:, 4]
    node_pl = df.iloc[1:, 11]
    node_is = df.iloc[1:, 10]
    level_rmse = df.iloc[1:, 14]
    level_mae=df.iloc[1:, 12]
    level_pl = df.iloc[1:, -1]
    level_is = df.iloc[1:, -2]


    colors_=['#3265FF','#81B380','#E6671A','#FFC432']
    # 绘制柱状图
    ax = axes[idx]
    width = 0.2  # 柱子宽度
    ax2 = ax.twinx()
    ls_s=1.5
    for spine in ax.spines.values():
        spine.set_linewidth(ls_s)

    ################### Node-based averaging metrics #####################################
    # ax.set_ylabel('RMSE Values', color=colors_[0],fontsize=ksize)
    # ax2.set_ylabel('PL Values', color=colors_[1],fontsize=ksize)
    # ax.tick_params(axis='y', colors=colors_[0], labelsize=ksize)
    # ax2.tick_params(axis='y', colors=colors_[1], labelsize=ksize)
    # ax.bar(x - 0.5 * width, node_rmse, width, label='Node-based averaging RMSE', color=colors_[0],
    #        edgecolor='black', linewidth=1.5)
    # ax2.bar(x + 0.5 * width, node_pl, width, label='Node-based averaging PL', color=colors_[1],
    #         edgecolor='black', linewidth=1.5)
    # ax.axhline(y=node_rmse[2], linestyle='--', color=colors_[0], linewidth=1.5)
    # ax2.axhline(y=node_pl[2], linestyle='--', color=colors_[1], linewidth=1.5)
    # ax2.set_yticks(np.arange(0.0, 0.045, 0.01))
    # ax.set_yticks(np.arange(0.15,0.53,0.1))
    # ax.set_ylim(0.15,0.53)

    ###################Level-based averaging metrics#####################################
    ax.set_ylabel('RMSE Values', color=colors_[2],fontsize=ksize)
    ax2.set_ylabel('PL Values', color=colors_[3],fontsize=ksize)
    ax.tick_params(axis='y', colors=colors_[2], labelsize=ksize)
    ax2.tick_params(axis='y', colors=colors_[3], labelsize=ksize)
    ax.bar(x -0.5 * width, level_rmse, width, label='Level-based averaging RMSE', color=colors_[2],
           edgecolor='black', linewidth=ls_s)
    ax2.bar(x + 0.5 * width, level_pl, width, label='Level-based averaging PL', color=colors_[3],
            edgecolor='black', linewidth=ls_s)
    ax.axhline(y=level_rmse[2], linestyle='--', color=colors_[2], linewidth=ls_s)
    ax2.axhline(y=level_pl[2], linestyle='--', color=colors_[3], linewidth=ls_s)
    ax2.set_yticks(np.arange(0.0, 0.045, 0.01))
    ax.set_yticks(np.arange(0.15,0.53,0.1))
    ax.set_ylim(0.15,0.53)



    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles.extend(handles1 + handles2)
    all_labels.extend(labels1 + labels2)
    ax.set_title(f'{dataset[idx]}', fontsize=ksize)
    if idx == 3:  # 只有最后一个子图显示x轴标签
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=ksize)
    else:
        ax.set_xticks([])  # 隐藏其他子图的x轴标签


# handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(all_handles, all_labels, ncol=4, loc='upper center',fontsize=ksize,bbox_to_anchor=(0.5, 0.98))
plt.subplots_adjust(hspace=0.2, wspace=0.3,top=0.91,bottom=0.15,left=0.05,right=0.95)# 调整图例的位置
# plt.suptitle('MAE and PL values (Node & Level)', fontsize=ksize, y=1.02)
plt.savefig('../pic_/metrics_all_level.svg', format='SVG', dpi=800)
# plt.savefig('../pic_/metrics_all_node.svg', format='SVG', dpi=800)
plt.show()
