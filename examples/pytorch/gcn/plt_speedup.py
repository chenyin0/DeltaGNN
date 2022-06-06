import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import matplotlib.ticker as mtick


def plot_speedup():
    # GCN
    # CPU GPU HyGCN AWB-GCN I-GCN
    gcn_cora = [1, 4.9, 500, 800, 1500, 1800]
    gcn_citeseer = [1, 6, 40, 450, 1100, 1800]
    gcn_pubmed = [1, 70, 250, 970, 1900, 2000]
    gcn_reddit = [1, 8000, 1800, 15000, 17000, 20000]

    gcn_data = np.array([gcn_cora, gcn_citeseer, gcn_pubmed, gcn_reddit])
    cpu = gcn_data[:, 0]
    gpu = gcn_data[:, 1]
    hygcn = gcn_data[:, 2]
    awb_gcn = gcn_data[:, 3]
    i_gcn = gcn_data[:, 4]
    delta_gcn = gcn_data[:, 5]

    labels = ['Cora', 'Citeseer', 'Pubmed', 'Reddit']
    items = ['CPU-DGL', 'GPU-DGL', 'HyGCN', 'AWB-GCN', 'I-GCN', 'Delta-GCN']

    data = [cpu, gpu, hygcn, awb_gcn, i_gcn, delta_gcn]

    # Group size in each label
    group_size = len(items)

    total_width = 0.35
    label_num = len(labels)
    width = total_width / label_num
    # Bar offset
    offset = -width / 2 * (group_size - 1)
    # Bar interval
    space = 0

    x = np.arange(label_num)  # the label locations
    x = x - ((group_size - 1) / 2) * width

    set_size = 1
    color = sns.cubehelix_palette(start=2.8,
                                  rot=-.1,
                                  n_colors=len(items) / set_size,
                                  dark=0.2,
                                  light=0.65)

    # color_2 = sns.cubehelix_palette(start=1,
    #                                 rot=-.1,
    #                                 n_colors=len(items) / set_size,
    #                                 dark=0.2,
    #                                 light=0.65)

    fig, ax1 = plt.subplots(figsize=(6, 3), dpi=600)
    # ax1 = plt.subplot(1, 3, 1)

    rects = [0 for n in range(len(items))]
    for i in range(len(items)):
        rects[i] = ax1.bar(
            x + i * (width + space) + offset,
            data[i],
            width=width,
            alpha=.99,
            edgecolor='w',
            label=items[i],
            zorder=2,
            #    color=color[i]
        )

    plt.grid(axis='y', zorder=1)

    # Plot图例
    plt.legend(fontsize=8)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(), 1)
            ax1.annotate(
                '{}'.format(height),
                fontsize=10,
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center',
                va='bottom',
                rotation=90)

    # for i in range(len(items)):
    #     autolabel(rects[i])

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, va="center", position=(0, -0.05), fontsize=14)
    # ax1.xaxis.set_visible(False)
    # ax1.xaxis.set_tick_params(direction='in')
    # ax1.axes.get_xaxis().set_visible(False)

    # plt.yticks(size=10)
    # plt.xticks(size = 10)

    ax1.set_ylim([0.1, 3e5])
    ax1.set_yscale('log')

    # my_y_ticks = np.arange(0, 120, 20)
    # plt.yticks(my_y_ticks)

    plt.legend(ncol=3, labelspacing=0, handlelength=1, fontsize=12, loc="best")
    # ax1.get_legend().remove()

    #plt.axhline(y=1, color='k', linestyle='-', linewidth=0.8)
    # fmt = '%.f%%'
    # yticks = mtick.FormatStrFormatter(fmt)
    # ax1.yaxis.set_major_formatter(yticks)

    fontsize = 16
    #plt.xlabel('Unroll number',fontsize = fontsize)
    plt.ylabel('Speedup', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('./figure/speedup.pdf', dpi=600, bbox_inches="tight", pad_inches=0)


plot_speedup()