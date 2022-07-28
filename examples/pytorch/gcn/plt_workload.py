import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def plt_workload_imbalance(g_csr, deg_th):
    indptr = g_csr[0].numpy().tolist()
    indices = g_csr[1].numpy().tolist()
    workload = []
    workload_delta = []
    for node_id in range(len(indptr) - 1):
        ngh_num = indptr[node_id + 1] - indptr[node_id]
        workload.append(ngh_num)
        if ngh_num > deg_th:
            workload_delta.append(ngh_num)
        else:
            workload_delta.append(1)

    items = ['Full-retrain', 'Delta-retrain']

    # workload = [0, 1, 5, 6, 9, 78]
    # workload_delta = [5, 8, 9, 63, 1, 2]

    data = [workload, workload_delta]

    # Group size in each label
    group_size = len(items)

    total_width = len(workload)*1.3
    label_num = len(workload)
    width = total_width / label_num
    # Bar offset
    offset = -width / 2 * (group_size - 1)
    # Bar interval
    space = 0

    x = np.arange(label_num)  # the label locations
    x = x - ((group_size - 1) / 2) * width

    fig, ax1 = plt.subplots(figsize=(8, 3), dpi=600)
    # ax1 = plt.subplot(1, 3, 1)

    rects = [0 for n in range(len(items))]
    for i in range(len(items)):
        rects[i] = ax1.bar(
            x + i * (width + space) + offset,
            data[i],
            width=width,
            alpha=.99,
            # edgecolor='w',
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
    # ax1.set_xticklabels(labels, va="center", position=(0, -0.05), fontsize=14)
    # ax1.xaxis.set_visible(False)
    ax1.xaxis.set_tick_params(direction='in')
    # ax1.xaxis.minorticks_off()
    xmajorLocator = MultipleLocator(500) #将x主刻度标签设置为500的倍数
    ax1.xaxis.set_major_locator(xmajorLocator)

    # plt.yticks(size=10)
    # plt.xticks(size = 10)

    ax1.set_ylim([0, 40])
    # ax1.set_yscale('log')

    # my_y_ticks = np.arange(0, 120, 20)
    # plt.yticks(my_y_ticks)

    plt.legend(ncol=2, labelspacing=0, handlelength=1, fontsize=12, loc="best")
    # ax1.get_legend().remove()

    #plt.axhline(y=1, color='k', linestyle='-', linewidth=0.8)
    # fmt = '%.f%%'
    # yticks = mtick.FormatStrFormatter(fmt)
    # ax1.yaxis.set_major_formatter(yticks)

    fontsize = 16
    plt.xlabel('Node ID',fontsize = fontsize)
    plt.ylabel('Workload size', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('./figure/workload_imbalance.pdf', dpi=600, bbox_inches="tight", pad_inches=0)
