import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import matplotlib.ticker as mtick
import plt_speedup


def plot_energy_efficiency():
    # GCN
    # Cora Citeseer Pubmed Reddit
    # HyGCN, AWB-GCN, I-GCN, Ours
    cora_dram_access = [5, 62.5, 2.5, 1]
    citeseer_dram_access = [45.5, 47.5, 2.275, 1]
    pubmed_dram_access = [11.7, 70.7, 5.3, 1]
    reddit_dram_access = [26.7, 6.67, 1.67, 1]

    # cora_combination_reduction = [5, 45.5, 11.7, 1]
    # citeseer_combination_reduction = [5, 45.5, 11.7, 1]
    # pubmed_combination_reduction = [5, 45.5, 11.7, 1]
    # reddit_combination_reduction = [5, 45.5, 11.7, 1]

    # Cora Citeseer Pubmed Reddit
    dram_energy_gcn_occupancy = [0.3, 0.3, 0.3, 0.3]
    # combination_energy_gcn_occupancy = []

    energy_gcn_cora = [
        round((cora_dram_access[i] - cora_dram_access[-1]) * dram_energy_gcn_occupancy[0] +
              cora_dram_access[-1], 2) for i in range(len(cora_dram_access))
    ]
    energy_gcn_citeseer = [
        round((citeseer_dram_access[i] - citeseer_dram_access[-1]) * dram_energy_gcn_occupancy[1] +
              citeseer_dram_access[-1], 2) for i in range(len(citeseer_dram_access))
    ]
    energy_gcn_pubmed = [
        round((pubmed_dram_access[i] - pubmed_dram_access[-1]) * dram_energy_gcn_occupancy[2] +
              pubmed_dram_access[-1], 2) for i in range(len(pubmed_dram_access))
    ]
    energy_gcn_reddit = [
        round((reddit_dram_access[i] - reddit_dram_access[-1]) * dram_energy_gcn_occupancy[3] +
              reddit_dram_access[-1], 2) for i in range(len(reddit_dram_access))
    ]

    energy_gcn_cora[-1] = 1
    energy_gcn_citeseer[-1] = 1
    energy_gcn_pubmed[-1] = 1
    energy_gcn_reddit[-1] = 1

    sp_gcn_cora = plt_speedup.gcn_cora.copy()
    sp_gcn_citeseer = plt_speedup.gcn_citeseer.copy()
    sp_gcn_pubmed = plt_speedup.gcn_pubmed.copy()
    sp_gcn_reddit = plt_speedup.gcn_reddit.copy()

    speedup_gcn_cora = sp_gcn_cora[2:5] + [sp_gcn_cora[-1]]
    speedup_gcn_citeseer = sp_gcn_citeseer[2:5] + [sp_gcn_citeseer[-1]]
    speedup_gcn_pubmed = sp_gcn_pubmed[2:5] + [sp_gcn_pubmed[-1]]
    speedup_gcn_reddit = sp_gcn_reddit[2:5] + [sp_gcn_reddit[-1]]

    # Energy efficiency
    ee_gcn_cora = [speedup_gcn_cora[i] / energy_gcn_cora[i] for i in range(len(speedup_gcn_cora))]
    ee_gcn_citeseer = [
        speedup_gcn_citeseer[i] / energy_gcn_citeseer[i] for i in range(len(speedup_gcn_citeseer))
    ]
    ee_gcn_pubmed = [
        speedup_gcn_pubmed[i] / energy_gcn_pubmed[i] for i in range(len(speedup_gcn_pubmed))
    ]
    ee_gcn_reddit = [
        speedup_gcn_reddit[i] / energy_gcn_reddit[i] for i in range(len(speedup_gcn_reddit))
    ]

    # Norm
    ee_gcn_cora_norm = [ee_gcn_cora[i] / ee_gcn_cora[0] for i in range(len(ee_gcn_cora))]
    ee_gcn_citeseer_norm = [
        ee_gcn_citeseer[i] / ee_gcn_citeseer[0] for i in range(len(ee_gcn_citeseer))
    ]
    ee_gcn_pubmed_norm = [ee_gcn_pubmed[i] / ee_gcn_pubmed[0] for i in range(len(ee_gcn_pubmed))]
    ee_gcn_reddit_norm = [ee_gcn_reddit[i] / ee_gcn_reddit[0] for i in range(len(ee_gcn_reddit))]

    gcn_data = np.array(
        [ee_gcn_cora_norm, ee_gcn_citeseer_norm, ee_gcn_pubmed_norm, ee_gcn_reddit_norm])
    hygcn = gcn_data[:, 0]
    awb_gcn = gcn_data[:, 1]
    i_gcn = gcn_data[:, 2]
    delta_gcn = gcn_data[:, 3]

    print('HyGCN: ', hygcn)
    print('AWB-GCN: ', awb_gcn)
    print('I-GCN: ', i_gcn)
    print('Ours: ', delta_gcn)

    labels = ['Cora', 'Citeseer', 'Pubmed', 'Reddit']
    items = ['HyGCN', 'AWB-GCN', 'I-GCN', 'Ours']

    data = [hygcn, awb_gcn, i_gcn, delta_gcn]

    # Group size in each label
    group_size = len(items)

    total_width = 0.6
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

    ax1.set_ylim([0.1, 5e4])
    ax1.set_yscale('log')

    # my_y_ticks = np.arange(0, 120, 20)
    # plt.yticks(my_y_ticks)

    plt.legend(ncol=4, labelspacing=0, handlelength=1, fontsize=12, loc="best")
    # ax1.get_legend().remove()

    #plt.axhline(y=1, color='k', linestyle='-', linewidth=0.8)
    # fmt = '%.f%%'
    # yticks = mtick.FormatStrFormatter(fmt)
    # ax1.yaxis.set_major_formatter(yticks)

    fontsize = 16
    #plt.xlabel('Unroll number',fontsize = fontsize)
    plt.ylabel('Energy Efficiency', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('./figure/energy_efficiency.pdf', dpi=600, bbox_inches="tight", pad_inches=0)


if __name__ == '__main__':
    plot_energy_efficiency()