import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.patches import Rectangle, ConnectionPatch
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = 'Arial'
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
model_names = ['iTransformer', 'Pyraformer', 'Mamba','MFFDM']
# npz里面包含['base','true','BU', 'TD', 'ols', 'hls_add', 'wls', 'mint_sample', 'mint_shrink'](true只在iTransformer中)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()  # 展平axes，方便索引
dataset_name=['Dataset A','Dataset B','Dataset C','Dataset D']
ksize=20
for idx,(site, dataset) in enumerate(zip(['Tibet', 'Tibet', 'USA', 'Australia'], ['2018', '2019', '2019', '2019'])):
    preds_files = [f'../data_record/{site}/{dataset}/Forecasts_hierarchy_{x}_{site}_{dataset}.npz' for x in model_names]
    preds = [np.load(x) for x in preds_files]
    all_preds=[]
    for i,x in enumerate(preds):
        if i==0:
            all_preds.append(x['true'])
            # all_preds.append(x['base'])
            all_preds.append(x['wls'])
        else:
            # all_preds.append(x['base'])
            all_preds.append(x['wls'])
    # 线条样式和颜色配置
    line_styles = ['-',]# '--', '-.', ':'
    colors = ['blue',   'red', 'green','orange','purple', 'cyan', 'magenta']
    markers = ['o', 's', '^', 'D', 'p', '*', 'x']  # 不同形状的点（圆形、方形、三角形等）
    marker_sizes = [5]
    # colors=['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF']
    linewidths = [1]  # 每条线的宽度
    l_min=[150+230,0+240,500+260,750+160]
    l_l=50
    l_max=[450,300,2000,1050]
    labels=['Observed','iTransformer-WLS','Pyraformer-WLS','Mamba-WLS','MFFDM-WLS']
    # 绘制 all_preds 中的每个数组，每个数组对应一条线

    for i, line_data in enumerate(all_preds):
        axes[idx].plot(line_data[l_min[idx]:l_min[idx]+l_l,-3],
                linestyle=line_styles[i % len(line_styles)],
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],  # 使用不同的形状
                markersize=marker_sizes[i % len(marker_sizes)],
                linewidth=linewidths[i % len(linewidths)],
                label=f' {labels[i]}')
    ls_s=1.5
    for spine in axes[idx].spines.values():
        spine.set_linewidth(ls_s)
    # 设置标题
    axes[idx].set_title(f'{dataset_name[idx]}',fontsize=ksize)
    # axes[idx].set_xticks(np.arange(0,2000,150))
    axes[idx].set_xlabel('Time (30min)',fontsize=ksize)
    axes[idx].set_ylabel('Wind speed (m/s)',fontsize=ksize)
    axes[idx].tick_params(axis='x', labelsize=ksize)
    axes[idx].tick_params(axis='y', labelsize=ksize)
    # axes[idx].set_yticks(np.arange(0, 7, 0.5))
    # axes[idx].grid(True)
    ls_s=1
    for spine in axes[idx].spines.values():
        spine.set_linewidth(ls_s)
    # rec_positions=[[20,30,4,5],[20,30,4,5],[20,30,4,5],[20,30,4,5]]
    # [xmin,xmax,ymin,ymax]=rec_positions[idx]
    # xmin, xmax = 2, 6
    # ymin, ymax = -0.5, 0.5
    #
    # # 绘制虚线框
    # rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
    #                      edgecolor='k',
    #                      linestyle='--',  # 定义为虚线
    #                      linewidth=3)  # 自定义粗细为2
    # axes[idx].add_patch(rect)
    # ax=axes[idx]
    # 添加放大图
    # positions_=[[245,253,2.5,6],[254,260,3.3,6.5],[138,143,1.8,5.5],[133,138,4.5,6.8]]
    # positions_ax=[[2/3,0.5,0.1,0.4],[2/3-0.025,0.05,0.1,0.4],[1/4+0.025,0.1,0.1,0.4],[1/4,0.5+0.025,0.1,0.4]]
    # [x1, x2, y1, y2] =positions_[idx]  # 自定义放大区域，这里是例子，你可以调整
    #
    # ax_inset = axes[idx].inset_axes(positions_ax[idx])  # 添加子图（缩小图）
    #
    # # 在放大区域绘制数据
    # for i, line_data in enumerate(all_preds):
    #     ax_inset.plot(line_data[l_min[idx]:l_max[idx], -3],
    #                   linestyle=line_styles[i % len(line_styles)],
    #                   color=colors[i % len(colors)],
    #                   linewidth=linewidths[i % len(linewidths)])
    #
    # # 设置放大区域的坐标范围（确保显示数据）
    # # ax_inset.set_axis_off()
    # ax_inset.set_xlim(x1,x2)
    # ax_inset.set_ylim(y1,y2)
    # ax_inset.set_xticks([])
    # ax_inset.set_yticks([])
    #
    #
    # # 在主图上添加矩形框，表示放大区域，并使用虚线
    # rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=ls_s, edgecolor='k', facecolor='none', linestyle='--')
    # axes[idx].add_patch(rect)
    #
    # # 使用虚线连接主图和放大图
    # con = ConnectionPatch(xyA=(x1, y1), xyB=(x1, y1), coordsA='data', coordsB='data',
    #                       axesA=axes[idx], axesB=ax_inset, color='k', linestyle='--', linewidth=ls_s)
    # fig.add_artist(con)
    #
    # con = ConnectionPatch(xyA=(x2, y1), xyB=(x2, y1), coordsA='data', coordsB='data',
    #                       axesA=axes[idx], axesB=ax_inset, color='k', linestyle='--', linewidth=ls_s)
    # fig.add_artist(con)
    #
    # con = ConnectionPatch(xyA=(x1, y2), xyB=(x1, y2), coordsA='data', coordsB='data',
    #                       axesA=axes[idx], axesB=ax_inset, color='k', linestyle='--', linewidth=ls_s)
    # fig.add_artist(con)
    #
    # con = ConnectionPatch(xyA=(x2, y2), xyB=(x2, y2), coordsA='data', coordsB='data',
    #                       axesA=axes[idx], axesB=ax_inset, color='k', linestyle='--', linewidth=ls_s)
    # fig.add_artist(con)
handles, labels = plt.gca().get_legend_handles_labels()
# 假设 all_handles 是线条或标记的句柄
# for handle in handles:
#     handle.set_linewidth(2)  # 调整线条的宽度
#     handle.set_markeredgewidth(2)  # 调整标记边缘的宽度
#     handle.set_markersize(8)  # 设置标记大小

fig.legend(handles, labels, ncol=7, loc='upper center',fontsize=ksize,bbox_to_anchor=(0.5, 1)
           ,markerscale=2)
plt.subplots_adjust(hspace=0.35, wspace=0.2,top=0.9,bottom=0.08,left=0.05,right=0.95)# 调整图例的位置
# plt.suptitle('MAE and PL values (Node & Level)', fontsize=ksize, y=1.02)
plt.savefig('../pic_/preds_all.svg', format='SVG', dpi=800)
# plt.savefig('../pic_/preds_all.png', dpi=800)
# plt.savefig('../pic_/metrics_all_node.svg', format='SVG', dpi=800)
plt.show()





    # # 自定义线条粗细、点的形状、线条颜色等的设置（示例设置，可根据喜好修改）
    # linewidths = 1  # 6条线的不同粗细
    # markers = ['o', 's', '^', 'v', '*', 'D']  # 6种不同的点形状
    # colors = ['r', 'g', 'b', 'c', 'm', 'y']  # 6种不同的颜色
    #
    # # 创建画布和4个子图，2x2布局
    # fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    # axes = axes.flatten()  # 展平axes，方便索引
    #
    # # 绘制第一组数据（6条曲线）在第一个子图上
    # for i in range(6):
    #     axes[0].plot(np.arange(100), data1[i], marker=markers[i], markersize=5, linestyle='-', linewidth=linewidths,
    #                  color=colors[i], label=f'Line {i + 1}')
    # axes[0].set_title('Data 1')
    # axes[0].set_xlabel('X-axis')
    # axes[0].set_ylabel('Y-axis')
    # axes[0].legend(loc='best')
    #
    # # 绘制第二组数据（6条曲线）在第二个子图上
    # for i in range(6):
    #     axes[1].plot(np.arange(100), data2[i], marker=markers[i], markersize=5, linestyle='--', linewidth=linewidths,
    #                  color=colors[i], label=f'Line {i + 1}')
    # axes[1].set_title('Data 2')
    # axes[1].set_xlabel('X-axis')
    # axes[1].set_ylabel('Y-axis')
    # axes[1].legend(loc='best')
    #
    # # 绘制第三组数据（6条曲线）在第三个子图上
    # for i in range(6):
    #     axes[2].plot(np.arange(100), data3[i], marker=markers[i], markersize=5, linestyle='-.', linewidth=linewidths,
    #                  color=colors[i], label=f'Line {i + 1}')
    # axes[2].set_title('Data 3')
    # axes[2].set_xlabel('X-axis')
    # axes[2].set_ylabel('Y-axis')
    # axes[2].legend(loc='best')
    #
    # # 绘制第四组数据（6条曲线）在第四个子图上
    # for i in range(6):
    #     axes[3].plot(np.arange(100), data4[i], marker=markers[i], markersize=5, linestyle=':', linewidth=linewidths,
    #                  color=colors[i], label=f'Line {i + 1}')
    # axes[3].set_title('Data 4')
    # axes[3].set_xlabel('X-axis')
    # axes[3].set_ylabel('Y-axis')
    # axes[3].legend(loc='best')
    #
    # # 调整子图之间的间距等布局设置
    # plt.tight_layout()
    #
    # # 显示图形
    # plt.show()
