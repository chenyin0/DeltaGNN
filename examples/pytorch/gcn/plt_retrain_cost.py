import matplotlib.pyplot as plt
import numpy as np


# Computation
def comp_per_layer(feat_dim, v_num, e_num):
    """
        X_k+1 = A*X_k*W
        Aggregate: delta = A*X
        Combination: delta*W
    """
    aggr = e_num
    comb = v_num * feat_dim
    comp = aggr + comb
    return comp


def access_per_layer(feat_dim, e_num):
    access = e_num * feat_dim * 32 / (8 * 1024 * 1024)  # MB
    return access


def norm(li, base):
    for i in range(len(li)):
        li[i] = round(li[i] / base, 2)


def accumulate(li):
    sum = 0
    for i in range(len(li)):
        tmp = li[i]
        li[i] += sum
        sum += tmp


# def accumulate_delta(li):
#     sum = 0
#     for i in range(len(li)):
#         li[i] += sum
#         sum += li[i]


def plt_full_retrain():
    r"""
    Plot full and delta (This function is deprecated)
    """
    cora = np.loadtxt('./results/cora_delta_neighbor.txt', delimiter=',')
    citeseer = np.loadtxt('./results/citeseer_delta_neighbor.txt', delimiter=',')
    pubmed = np.loadtxt('./results/pubmed_delta_neighbor.txt', delimiter=',')

    v_cora = cora[:, 0]
    e_cora = cora[:, 1]

    v_citeseer = citeseer[:, 0]
    e_citeseer = citeseer[:, 1]

    v_pubmed = pubmed[:, 0]
    e_pubmed = pubmed[:, 1]

    comp_cora = []
    comp_citeseer = []
    comp_pubmed = []
    mem_cora = []
    mem_citeseer = []
    mem_pubmed = []

    layer_num = 2
    feat_dim = 16

    for i in range(len(cora)):
        comp = comp_per_layer(feat_dim, v_cora[i], e_cora[i])
        comp_cora.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_cora[i])
        mem_cora.append(layer_num * mem)

    for i in range(len(citeseer)):
        comp = comp_per_layer(feat_dim, v_citeseer[i], e_citeseer[i])
        comp_citeseer.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_citeseer[i])
        mem_citeseer.append(layer_num * mem)

    for i in range(len(pubmed)):
        comp = comp_per_layer(feat_dim, v_pubmed[i], e_pubmed[i])
        comp_pubmed.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_pubmed[i])
        mem_pubmed.append(layer_num * mem)

    ##
    """
    Computation and memory access of delta-retraining        
    """
    comp_cora_delta = []
    comp_citeseer_delta = []
    comp_pubmed_delta = []
    mem_cora_delta = []
    mem_citeseer_delta = []
    mem_pubmed_delta = []

    v_base = 0
    e_base = 0
    for i in range(len(cora)):
        comp = comp_per_layer(feat_dim, v_cora[i] - v_base, e_cora[i] - e_base)
        comp_cora_delta.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_cora[i] - e_base)
        mem_cora_delta.append(layer_num * mem)
        v_base = v_cora[i]
        e_base = e_cora[i]

    v_base = 0
    e_base = 0
    for i in range(len(citeseer)):
        comp = comp_per_layer(feat_dim, v_citeseer[i] - v_base, e_citeseer[i] - e_base)
        comp_citeseer_delta.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_citeseer[i] - e_base)
        mem_citeseer_delta.append(layer_num * mem)
        v_base = v_citeseer[i]
        e_base = e_citeseer[i]

    v_base = 0
    e_base = 0
    for i in range(len(pubmed)):
        comp = comp_per_layer(feat_dim, v_pubmed[i] - v_base, e_pubmed[i] - e_base)
        comp_pubmed_delta.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_pubmed[i] - e_base)
        mem_pubmed_delta.append(layer_num * mem)
        v_base = v_pubmed[i]
        e_base = e_pubmed[i]
    ##

    accumulate(comp_cora)
    accumulate(comp_citeseer)
    accumulate(comp_pubmed)
    accumulate(mem_cora)
    accumulate(mem_citeseer)
    accumulate(mem_pubmed)

    # normalize
    norm(comp_cora, comp_cora[0])
    norm(comp_citeseer, comp_citeseer[0])
    norm(comp_pubmed, comp_pubmed[0])
    norm(mem_cora, mem_cora[0])
    norm(mem_citeseer, mem_citeseer[0])
    norm(mem_pubmed, mem_pubmed[0])

    ## delta
    accumulate(comp_cora_delta)
    accumulate(comp_citeseer_delta)
    accumulate(comp_pubmed_delta)
    accumulate(mem_cora_delta)
    accumulate(mem_citeseer_delta)
    accumulate(mem_pubmed_delta)

    # normalize
    norm(comp_cora_delta, comp_cora_delta[0])
    norm(comp_citeseer_delta, comp_citeseer_delta[0])
    norm(comp_pubmed_delta, comp_pubmed_delta[0])
    norm(mem_cora_delta, mem_cora_delta[0])
    norm(mem_citeseer_delta, mem_citeseer_delta[0])
    norm(mem_pubmed_delta, mem_pubmed_delta[0])

    import seaborn as sns
    import matplotlib.ticker as mtick

    labels = [
        'Orig', 'Time 1', 'Time 2', 'Time 3', 'Time 4', 'Time 5', 'Time 6', 'Time 7', 'Time 8'
    ]
    items = ['Cora', 'Citeseer', 'Pubmed', 'Cora_delta', 'Citeseer_delta', 'Pubmed_delta']
    """
    Computation
    """
    # data = [comp_cora, comp_citeseer, comp_pubmed]
    data = [
        comp_cora, comp_citeseer, comp_pubmed, comp_cora_delta, comp_citeseer_delta,
        comp_pubmed_delta
    ]

    # Group size in each label
    group_size = len(items)

    total_width = 1.2
    label_num = len(labels)
    width = total_width / label_num
    # Bar offset
    offset = -width / 2 * (group_size - 1)
    # Bar interval
    space = 0

    x = np.arange(label_num)  # the label locations
    x = x - ((group_size - 1) / 2) * width

    set_size = 2
    color_1 = sns.cubehelix_palette(start=2.8,
                                    rot=-.1,
                                    n_colors=len(items) / set_size,
                                    dark=0.2,
                                    light=0.65)

    color_2 = sns.cubehelix_palette(start=1,
                                    rot=-.1,
                                    n_colors=len(items) / set_size,
                                    dark=0.2,
                                    light=0.65)

    fig, ax1 = plt.subplots(figsize=(11, 3), dpi=600)
    # ax1 = plt.subplot(1, 3, 1)

    rects = [0 for n in range(len(items))]
    for i in range(len(items)):
        if i < len(items) / set_size:
            rects[i] = ax1.bar(x + i * (width + space) + offset,
                               data[i],
                               width=width,
                               alpha=.99,
                               edgecolor='w',
                               label=items[i],
                               zorder=2,
                               color=color_1[i])
        else:
            rects[i] = ax1.bar(x + i * (width + space) + offset,
                               data[i],
                               width=width,
                               alpha=.99,
                               edgecolor='w',
                               label=items[i],
                               zorder=2,
                               color=color_2[i - 3])

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

    for i in range(len(items)):
        autolabel(rects[i])

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, va="center", position=(0, -0.05), fontsize=14)
    # ax1.xaxis.set_visible(False)
    # ax1.xaxis.set_tick_params(direction='in')
    # ax1.axes.get_xaxis().set_visible(False)

    # plt.yticks(size=10)
    # plt.xticks(size = 10)

    # ax1.set_ylim([0, 100])
    # ax1.set_yscale('log')

    # my_y_ticks = np.arange(0, 120, 20)
    # plt.yticks(my_y_ticks)

    plt.legend(ncol=2, labelspacing=0, handlelength=1, fontsize=14, loc="best")
    # ax1.get_legend().remove()

    #plt.axhline(y=1, color='k', linestyle='-', linewidth=0.8)
    # fmt = '%.f%%'
    # yticks = mtick.FormatStrFormatter(fmt)
    # ax1.yaxis.set_major_formatter(yticks)

    fontsize = 16
    #plt.xlabel('Unroll number',fontsize = fontsize)
    plt.ylabel('Computation\n(norm to non-retrain)', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('./figure/retrain_comp.pdf', dpi=600, bbox_inches="tight", pad_inches=0)

    ##
    """
    Mem access
    """

    # data = [mem_cora, mem_citeseer, mem_pubmed]
    data = [
        mem_cora, mem_citeseer, mem_pubmed, mem_cora_delta, mem_citeseer_delta, mem_pubmed_delta
    ]

    # Group size in each label
    group_size = len(items)

    total_width = 1.2
    label_num = len(labels)
    width = total_width / label_num
    # Bar offset
    offset = -width / 2 * (group_size - 1)
    # Bar interval
    space = 0

    x = np.arange(label_num)  # the label locations
    x = x - ((group_size - 1) / 2) * width

    color_1 = sns.cubehelix_palette(start=0,
                                    rot=-.4,
                                    n_colors=len(items) / set_size,
                                    dark=0.2,
                                    light=0.65)

    color_2 = sns.cubehelix_palette(start=1,
                                    rot=-.1,
                                    n_colors=len(items) / set_size,
                                    dark=0.2,
                                    light=0.65)

    fig, ax1 = plt.subplots(figsize=(11, 3), dpi=600)
    # ax1 = plt.subplot(1, 3, 1)

    rects = [0 for n in range(len(items))]
    for i in range(len(items)):
        if i < len(items) / set_size:
            rects[i] = ax1.bar(x + i * (width + space) + offset,
                               data[i],
                               width=width,
                               alpha=.99,
                               edgecolor='w',
                               label=items[i],
                               zorder=2,
                               color=color_1[i])
        else:
            rects[i] = ax1.bar(x + i * (width + space) + offset,
                               data[i],
                               width=width,
                               alpha=.99,
                               edgecolor='w',
                               label=items[i],
                               zorder=2,
                               color=color_2[i - 3])

    plt.grid(axis='y', zorder=1)

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

    for i in range(len(items)):
        autolabel(rects[i])

    # Plot图例
    plt.legend(fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, va="center", position=(0, -0.05), fontsize=14)
    # ax1.xaxis.set_visible(False)
    # ax1.xaxis.set_tick_params(direction='in')
    # ax1.axes.get_xaxis().set_visible(False)

    # plt.yticks(size=10)
    # plt.xticks(size = 10)

    # ax1.set_ylim([0, 100])
    # ax1.set_yscale('log')

    # my_y_ticks = np.arange(0, 120, 20)
    # plt.yticks(my_y_ticks)

    plt.legend(ncol=2, labelspacing=0, handlelength=1, fontsize=14, loc="best")
    # ax1.get_legend().remove()

    #plt.axhline(y=1, color='k', linestyle='-', linewidth=0.8)
    # fmt = '%.f%%'
    # yticks = mtick.FormatStrFormatter(fmt)
    # ax1.yaxis.set_major_formatter(yticks)

    fontsize = 16
    #plt.xlabel('Unroll number',fontsize = fontsize)
    plt.ylabel('Memory access\n(norm to Origin)', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('./figure/retrain_mem.pdf', dpi=600, bbox_inches="tight", pad_inches=0)


def plt_delta_retrain():
    deg_th = 8
    cora_evo = np.loadtxt('./results/node_access/cora_evo.txt', delimiter=',')
    citeseer_evo = np.loadtxt('./results/node_access/citeseer_evo.txt', delimiter=',')
    pubmed_evo = np.loadtxt('./results/node_access/pubmed_evo.txt', delimiter=',')

    cora_evo_delta = np.loadtxt('./results/node_access/cora_evo_delta_deg_' + str(deg_th) + '.txt',
                                delimiter=',')
    citeseer_evo_delta = np.loadtxt('./results/node_access/citeseer_evo_delta_deg_' + str(deg_th) +
                                    '.txt',
                                    delimiter=',')
    pubmed_evo_delta = np.loadtxt('./results/node_access/pubmed_evo_delta_deg_' + str(deg_th) +
                                  '.txt',
                                  delimiter=',')
    # pubmed_delta = np.loadtxt('./results/node_access/amazon_comp_evo_delta_deg_' + str(deg_th) + '.txt',
    #                     delimiter=',')

    v_cora = cora_evo[:, 0]
    e_cora = cora_evo[:, 1]

    v_citeseer = citeseer_evo[:, 0]
    e_citeseer = citeseer_evo[:, 1]

    v_pubmed = pubmed_evo[:, 0]
    e_pubmed = pubmed_evo[:, 1]

    comp_cora = []
    comp_citeseer = []
    comp_pubmed = []
    mem_cora = []
    mem_citeseer = []
    mem_pubmed = []

    layer_num = 2
    feat_dim = 16

    for i in range(len(cora_evo)):
        comp = comp_per_layer(feat_dim, v_cora[i], e_cora[i])
        comp_cora.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_cora[i])
        mem_cora.append(layer_num * mem)

    for i in range(len(citeseer_evo)):
        comp = comp_per_layer(feat_dim, v_citeseer[i], e_citeseer[i])
        comp_citeseer.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_citeseer[i])
        mem_citeseer.append(layer_num * mem)

    for i in range(len(pubmed_evo)):
        comp = comp_per_layer(feat_dim, v_pubmed[i], e_pubmed[i])
        comp_pubmed.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_pubmed[i])
        mem_pubmed.append(layer_num * mem)

    ##
    """
    Computation and memory access of ngh-delta retraining    
    """
    v_cora_ngh_delta = cora_evo_delta[:, 0]
    e_cora_ngh_delta = cora_evo_delta[:, 1]
    v_citeseer_ngh_delta = citeseer_evo_delta[:, 0]
    e_citeseer_ngh_delta = citeseer_evo_delta[:, 1]
    v_pubmed_ngh_delta = pubmed_evo_delta[:, 0]
    e_pubmed_ngh_delta = pubmed_evo_delta[:, 1]

    comp_cora_ngh_delta = []
    comp_citeseer_ngh_delta = []
    comp_pubmed_ngh_delta = []
    mem_cora_ngh_delta = []
    mem_citeseer_ngh_delta = []
    mem_pubmed_ngh_delta = []

    for i in range(len(cora_evo_delta)):
        comp = comp_per_layer(feat_dim, v_cora_ngh_delta[i], e_cora_ngh_delta[i])
        comp_cora_ngh_delta.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_cora_ngh_delta[i])
        mem_cora_ngh_delta.append(layer_num * mem)

    for i in range(len(citeseer_evo_delta)):
        comp = comp_per_layer(feat_dim, v_citeseer_ngh_delta[i], e_citeseer_ngh_delta[i])
        comp_citeseer_ngh_delta.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_citeseer_ngh_delta[i])
        mem_citeseer_ngh_delta.append(layer_num * mem)

    for i in range(len(pubmed_evo_delta)):
        comp = comp_per_layer(feat_dim, v_pubmed_ngh_delta[i], e_pubmed_ngh_delta[i])
        comp_pubmed_ngh_delta.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_pubmed_ngh_delta[i])
        mem_pubmed_ngh_delta.append(layer_num * mem)
    ##

    ##
    """
    Computation and memory access of ngh_all retraining        
    """
    v_cora_ngh_all = cora_evo[:, 2]
    e_cora_ngh_all = cora_evo[:, 3]
    v_citeseer_ngh_all = citeseer_evo[:, 2]
    e_citeseer_ngh_all = citeseer_evo[:, 3]
    v_pubmed_ngh_all = pubmed_evo[:, 2]
    e_pubmed_ngh_all = pubmed_evo[:, 3]

    comp_cora_ngh_all = []
    comp_citeseer_ngh_all = []
    comp_pubmed_ngh_all = []
    mem_cora_ngh_all = []
    mem_citeseer_ngh_all = []
    mem_pubmed_ngh_all = []

    for i in range(len(cora_evo)):
        comp = comp_per_layer(feat_dim, v_cora_ngh_all[i], e_cora_ngh_all[i])
        comp_cora_ngh_all.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_cora_ngh_all[i])
        mem_cora_ngh_all.append(layer_num * mem)

    for i in range(len(citeseer_evo)):
        comp = comp_per_layer(feat_dim, v_citeseer_ngh_all[i], e_citeseer_ngh_all[i])
        comp_citeseer_ngh_all.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_citeseer_ngh_all[i])
        mem_citeseer_ngh_all.append(layer_num * mem)

    for i in range(len(pubmed_evo)):
        comp = comp_per_layer(feat_dim, v_pubmed_ngh_all[i], e_pubmed_ngh_all[i])
        comp_pubmed_ngh_all.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_pubmed_ngh_all[i])
        mem_pubmed_ngh_all.append(layer_num * mem)
    ##

    accumulate(comp_cora)
    accumulate(comp_citeseer)
    accumulate(comp_pubmed)
    accumulate(mem_cora)
    accumulate(mem_citeseer)
    accumulate(mem_pubmed)

    # normalize
    # norm(comp_cora, comp_cora[0])
    # norm(comp_citeseer, comp_citeseer[0])
    # norm(comp_pubmed, comp_pubmed[0])
    # norm(mem_cora, mem_cora[0])
    # norm(mem_citeseer, mem_citeseer[0])
    # norm(mem_pubmed, mem_pubmed[0])
    norm(comp_cora, comp_cora_ngh_delta[0])
    norm(comp_citeseer, comp_citeseer_ngh_delta[0])
    norm(comp_pubmed, comp_pubmed_ngh_delta[0])
    norm(mem_cora, mem_cora_ngh_delta[0])
    norm(mem_citeseer, mem_citeseer_ngh_delta[0])
    norm(mem_pubmed, mem_pubmed_ngh_delta[0])

    ## ngh_all
    accumulate(comp_cora_ngh_all)
    accumulate(comp_citeseer_ngh_all)
    accumulate(comp_pubmed_ngh_all)
    accumulate(mem_cora_ngh_all)
    accumulate(mem_citeseer_ngh_all)
    accumulate(mem_pubmed_ngh_all)

    # normalize
    # norm(comp_cora_ngh_all, comp_cora_ngh_all[0])
    # norm(comp_citeseer_ngh_all, comp_citeseer_ngh_all[0])
    # norm(comp_pubmed_ngh_all, comp_pubmed_ngh_all[0])
    # norm(mem_cora_ngh_all, mem_cora_ngh_all[0])
    # norm(mem_citeseer_ngh_all, mem_citeseer_ngh_all[0])
    # norm(mem_pubmed_ngh_all, mem_pubmed_ngh_all[0])
    norm(comp_cora_ngh_all, comp_cora_ngh_delta[0])
    norm(comp_citeseer_ngh_all, comp_citeseer_ngh_delta[0])
    norm(comp_pubmed_ngh_all, comp_pubmed_ngh_delta[0])
    norm(mem_cora_ngh_all, mem_cora_ngh_delta[0])
    norm(mem_citeseer_ngh_all, mem_citeseer_ngh_delta[0])
    norm(mem_pubmed_ngh_all, mem_pubmed_ngh_delta[0])

    ## ngh_delta
    accumulate(comp_cora_ngh_delta)
    accumulate(comp_citeseer_ngh_delta)
    accumulate(comp_pubmed_ngh_delta)
    accumulate(mem_cora_ngh_delta)
    accumulate(mem_citeseer_ngh_delta)
    accumulate(mem_pubmed_ngh_delta)

    # normalize
    norm(comp_cora_ngh_delta, comp_cora_ngh_delta[0])
    norm(comp_citeseer_ngh_delta, comp_citeseer_ngh_delta[0])
    norm(comp_pubmed_ngh_delta, comp_pubmed_ngh_delta[0])
    norm(mem_cora_ngh_delta, mem_cora_ngh_delta[0])
    norm(mem_citeseer_ngh_delta, mem_citeseer_ngh_delta[0])
    norm(mem_pubmed_ngh_delta, mem_pubmed_ngh_delta[0])

    import seaborn as sns
    import matplotlib.ticker as mtick

    labels = [
        'Inital', 'Snapshot 1', 'Snapshot 2', 'Snapshot 3', 'Snapshot 4', 'Snapshot 5',
        'Snapshot 6', 'Snapshot 7'
    ]
    items = [
        'Cora_full', 'Citeseer_full', 'Pubmed_full', 'Cora_ngh_all', 'Citeseer_ngh_all',
        'Pubmed_ngh_all', 'Cora_ngh_delta', 'Citeseer_ngh_delta', 'Pubmed_ngh_delta'
    ]
    """
    Computation
    """
    data = [
        comp_cora, comp_citeseer, comp_pubmed, comp_cora_ngh_all, comp_citeseer_ngh_all,
        comp_pubmed_ngh_all, comp_cora_ngh_delta, comp_citeseer_ngh_delta, comp_pubmed_ngh_delta
    ]

    # Only count seven shotsnaps
    for i in range(len(data)):
        data[i] = data[i][:8]

    # # Print combination(computation) reduction of each dataset
    # comb_reduct_cora = sum(comp_cora) / sum(comp_cora_ngh_delta)
    # comb_reduct_citeseer = sum(comp_citeseer) / sum(comp_citeseer_ngh_delta)
    # comb_reduct_pubmed = sum(comp_pubmed) / sum(comp_pubmed_ngh_delta)
    # print('\nCombination reduction: \ncora: {:.2%}, citeseer: {:.2%}, pubmed: {:.2%}'.format(
    #     comb_reduct_cora, comb_reduct_citeseer, comb_reduct_pubmed))

    # Group size in each label
    group_size = len(items)

    total_width = 0.65
    label_num = len(labels)
    width = total_width / label_num
    # Bar offset
    offset = -width / 2 * (group_size - 1)
    # Bar interval
    space = 0

    x = np.arange(label_num)  # the label locations
    x = x - ((group_size - 1) / 2) * width

    span = 3
    color_1 = sns.cubehelix_palette(start=2.8, rot=-.1, n_colors=span, dark=0.2, light=0.65)

    color_2 = sns.cubehelix_palette(start=1, rot=-.1, n_colors=span, dark=0.2, light=0.65)

    color_3 = sns.cubehelix_palette(start=1.8, rot=-.1, n_colors=span, dark=0.2, light=0.65)

    fig, ax1 = plt.subplots(figsize=(11, 3), dpi=600)
    # ax1 = plt.subplot(1, 3, 1)

    rects = [0 for n in range(len(items))]
    for i in range(len(items)):
        if i < span:
            rects[i] = ax1.bar(x + i * (width + space) + offset,
                               data[i],
                               width=width,
                               alpha=.99,
                               edgecolor='w',
                               label=items[i],
                               zorder=2,
                               color=color_1[i])
        elif i >= span and i < span * 2:
            rects[i] = ax1.bar(x + i * (width + space) + offset,
                               data[i],
                               width=width,
                               alpha=.99,
                               edgecolor='w',
                               label=items[i],
                               zorder=2,
                               color=color_2[i - span])
        else:
            rects[i] = ax1.bar(x + i * (width + space) + offset,
                               data[i],
                               width=width,
                               alpha=.99,
                               edgecolor='w',
                               label=items[i],
                               zorder=2,
                               color=color_3[i - span * 2])

    plt.grid(axis='y', zorder=1)

    # Plot图例
    plt.legend(fontsize=8)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(), 1)
            ax1.annotate(
                '{}'.format(height),
                fontsize=8,
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center',
                va='bottom',
                rotation=90)

    for i in range(len(items)):
        autolabel(rects[i])

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, va="center", position=(0, -0.05), fontsize=14)
    # ax1.xaxis.set_visible(False)
    # ax1.xaxis.set_tick_params(direction='in')
    # ax1.axes.get_xaxis().set_visible(False)

    # plt.yticks(size=10)
    # plt.xticks(size = 10)

    # ax1.set_ylim([0, 10000])
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
    plt.ylabel('Total combination\n(norm to Origin)', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('./figure/retrain_comp_delta.pdf', dpi=600, bbox_inches="tight", pad_inches=0)
    """
    Mem access
    """

    # data = [mem_cora, mem_citeseer, mem_pubmed]
    data = [
        mem_cora, mem_citeseer, mem_pubmed, mem_cora_ngh_all, mem_citeseer_ngh_all,
        mem_pubmed_ngh_all, mem_cora_ngh_delta, mem_citeseer_ngh_delta, mem_pubmed_ngh_delta
    ]

    # Only count seven shotsnaps
    for i in range(len(data)):
        data[i] = data[i][:8]

    # # Print aggregation (mem access) reduction of each dataset
    # aggr_reduct_cora = sum(mem_cora) / sum(mem_cora_ngh_delta)
    # aggr_reduct_citeseer = sum(mem_citeseer) / sum(mem_citeseer_ngh_delta)
    # aggr_reduct_pubmed = sum(mem_pubmed) / sum(mem_pubmed_ngh_delta)
    # print('\nAggregation reduction: \ncora: {:.2%}, citeseer: {:.2%}, pubmed: {:.2%}'.format(
    #     aggr_reduct_cora, aggr_reduct_citeseer, aggr_reduct_pubmed))

    # Group size in each label
    group_size = len(items)

    total_width = 0.65
    label_num = len(labels)
    width = total_width / label_num
    # Bar offset
    offset = -width / 2 * (group_size - 1)
    # Bar interval
    space = 0

    x = np.arange(label_num)  # the label locations
    x = x - ((group_size - 1) / 2) * width

    span = 3
    color_1 = sns.cubehelix_palette(start=2.8, rot=-.1, n_colors=span, dark=0.2, light=0.65)

    color_2 = sns.cubehelix_palette(start=1, rot=-.1, n_colors=span, dark=0.2, light=0.65)

    color_3 = sns.cubehelix_palette(start=1.8, rot=-.1, n_colors=span, dark=0.2, light=0.65)

    fig, ax1 = plt.subplots(figsize=(11, 3), dpi=600)
    # ax1 = plt.subplot(1, 3, 1)

    rects = [0 for n in range(len(items))]
    for i in range(len(items)):
        if i < span:
            rects[i] = ax1.bar(x + i * (width + space) + offset,
                               data[i],
                               width=width,
                               alpha=.99,
                               edgecolor='w',
                               label=items[i],
                               zorder=2,
                               color=color_1[i])
        elif i >= span and i < span * 2:
            rects[i] = ax1.bar(x + i * (width + space) + offset,
                               data[i],
                               width=width,
                               alpha=.99,
                               edgecolor='w',
                               label=items[i],
                               zorder=2,
                               color=color_2[i - span])
        else:
            rects[i] = ax1.bar(x + i * (width + space) + offset,
                               data[i],
                               width=width,
                               alpha=.99,
                               edgecolor='w',
                               label=items[i],
                               zorder=2,
                               color=color_3[i - span * 2])

    plt.grid(axis='y', zorder=1)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(), 1)
            ax1.annotate(
                '{}'.format(height),
                fontsize=8,
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center',
                va='bottom',
                rotation=90)

    for i in range(len(items)):
        autolabel(rects[i])

    # Plot图例
    plt.legend(fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, va="center", position=(0, -0.05), fontsize=14)
    # ax1.xaxis.set_visible(False)
    # ax1.xaxis.set_tick_params(direction='in')
    # ax1.axes.get_xaxis().set_visible(False)

    # plt.yticks(size=10)
    # plt.xticks(size = 10)

    # ax1.set_ylim([0, 100])
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
    plt.ylabel('Total aggregation\n(norm to Origin)', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('./figure/retrain_mem_delta.pdf', dpi=600, bbox_inches="tight", pad_inches=0)


def plt_delta_retrain_cora():
    deg_th = 8
    cora_evo = np.loadtxt('./results/node_access/cora_evo.txt', delimiter=',')
    citeseer_evo = np.loadtxt('./results/node_access/citeseer_evo.txt', delimiter=',')
    pubmed_evo = np.loadtxt('./results/node_access/pubmed_evo.txt', delimiter=',')

    cora_evo_delta = np.loadtxt('./results/node_access/cora_evo_delta_deg_' + str(deg_th) + '.txt',
                                delimiter=',')
    citeseer_evo_delta = np.loadtxt('./results/node_access/citeseer_evo_delta_deg_' + str(deg_th) +
                                    '.txt',
                                    delimiter=',')
    pubmed_evo_delta = np.loadtxt('./results/node_access/pubmed_evo_delta_deg_' + str(deg_th) +
                                  '.txt',
                                  delimiter=',')
    # pubmed_delta = np.loadtxt('./results/node_access/amazon_comp_evo_delta_deg_' + str(deg_th) + '.txt',
    #                     delimiter=',')

    v_cora = cora_evo[:, 0]
    e_cora = cora_evo[:, 1]

    v_citeseer = citeseer_evo[:, 0]
    e_citeseer = citeseer_evo[:, 1]

    v_pubmed = pubmed_evo[:, 0]
    e_pubmed = pubmed_evo[:, 1]

    comp_cora = []
    comp_citeseer = []
    comp_pubmed = []
    mem_cora = []
    mem_citeseer = []
    mem_pubmed = []

    layer_num = 2
    feat_dim = 16

    for i in range(len(cora_evo)):
        comp = comp_per_layer(feat_dim, v_cora[i], e_cora[i])
        comp_cora.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_cora[i])
        mem_cora.append(layer_num * mem)

    for i in range(len(citeseer_evo)):
        comp = comp_per_layer(feat_dim, v_citeseer[i], e_citeseer[i])
        comp_citeseer.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_citeseer[i])
        mem_citeseer.append(layer_num * mem)

    for i in range(len(pubmed_evo)):
        comp = comp_per_layer(feat_dim, v_pubmed[i], e_pubmed[i])
        comp_pubmed.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_pubmed[i])
        mem_pubmed.append(layer_num * mem)

    ##
    """
    Computation and memory access of ngh-delta retraining    
    """
    v_cora_ngh_delta = cora_evo_delta[:, 0]
    e_cora_ngh_delta = cora_evo_delta[:, 1]
    v_citeseer_ngh_delta = citeseer_evo_delta[:, 0]
    e_citeseer_ngh_delta = citeseer_evo_delta[:, 1]
    v_pubmed_ngh_delta = pubmed_evo_delta[:, 0]
    e_pubmed_ngh_delta = pubmed_evo_delta[:, 1]

    comp_cora_ngh_delta = []
    comp_citeseer_ngh_delta = []
    comp_pubmed_ngh_delta = []
    mem_cora_ngh_delta = []
    mem_citeseer_ngh_delta = []
    mem_pubmed_ngh_delta = []

    for i in range(len(cora_evo_delta)):
        comp = comp_per_layer(feat_dim, v_cora_ngh_delta[i], e_cora_ngh_delta[i])
        comp_cora_ngh_delta.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_cora_ngh_delta[i])
        mem_cora_ngh_delta.append(layer_num * mem)

    for i in range(len(citeseer_evo_delta)):
        comp = comp_per_layer(feat_dim, v_citeseer_ngh_delta[i], e_citeseer_ngh_delta[i])
        comp_citeseer_ngh_delta.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_citeseer_ngh_delta[i])
        mem_citeseer_ngh_delta.append(layer_num * mem)

    for i in range(len(pubmed_evo_delta)):
        comp = comp_per_layer(feat_dim, v_pubmed_ngh_delta[i], e_pubmed_ngh_delta[i])
        comp_pubmed_ngh_delta.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_pubmed_ngh_delta[i])
        mem_pubmed_ngh_delta.append(layer_num * mem)
    ##

    ##
    """
    Computation and memory access of ngh_all retraining        
    """
    v_cora_ngh_all = cora_evo[:, 2]
    e_cora_ngh_all = cora_evo[:, 3]
    v_citeseer_ngh_all = citeseer_evo[:, 2]
    e_citeseer_ngh_all = citeseer_evo[:, 3]
    v_pubmed_ngh_all = pubmed_evo[:, 2]
    e_pubmed_ngh_all = pubmed_evo[:, 3]

    comp_cora_ngh_all = []
    comp_citeseer_ngh_all = []
    comp_pubmed_ngh_all = []
    mem_cora_ngh_all = []
    mem_citeseer_ngh_all = []
    mem_pubmed_ngh_all = []

    for i in range(len(cora_evo)):
        comp = comp_per_layer(feat_dim, v_cora_ngh_all[i], e_cora_ngh_all[i])
        comp_cora_ngh_all.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_cora_ngh_all[i])
        mem_cora_ngh_all.append(layer_num * mem)

    for i in range(len(citeseer_evo)):
        comp = comp_per_layer(feat_dim, v_citeseer_ngh_all[i], e_citeseer_ngh_all[i])
        comp_citeseer_ngh_all.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_citeseer_ngh_all[i])
        mem_citeseer_ngh_all.append(layer_num * mem)

    for i in range(len(pubmed_evo)):
        comp = comp_per_layer(feat_dim, v_pubmed_ngh_all[i], e_pubmed_ngh_all[i])
        comp_pubmed_ngh_all.append(layer_num * comp)
        mem = access_per_layer(feat_dim, e_pubmed_ngh_all[i])
        mem_pubmed_ngh_all.append(layer_num * mem)
    ##

    accumulate(comp_cora)
    accumulate(comp_citeseer)
    accumulate(comp_pubmed)
    accumulate(mem_cora)
    accumulate(mem_citeseer)
    accumulate(mem_pubmed)

    # normalize
    # norm(comp_cora, comp_cora[0])
    # norm(comp_citeseer, comp_citeseer[0])
    # norm(comp_pubmed, comp_pubmed[0])
    # norm(mem_cora, mem_cora[0])
    # norm(mem_citeseer, mem_citeseer[0])
    # norm(mem_pubmed, mem_pubmed[0])
    norm(comp_cora, comp_cora_ngh_delta[0])
    norm(comp_citeseer, comp_citeseer_ngh_delta[0])
    norm(comp_pubmed, comp_pubmed_ngh_delta[0])
    norm(mem_cora, mem_cora_ngh_delta[0])
    norm(mem_citeseer, mem_citeseer_ngh_delta[0])
    norm(mem_pubmed, mem_pubmed_ngh_delta[0])

    ## ngh_all
    accumulate(comp_cora_ngh_all)
    accumulate(comp_citeseer_ngh_all)
    accumulate(comp_pubmed_ngh_all)
    accumulate(mem_cora_ngh_all)
    accumulate(mem_citeseer_ngh_all)
    accumulate(mem_pubmed_ngh_all)

    # normalize
    # norm(comp_cora_ngh_all, comp_cora_ngh_all[0])
    # norm(comp_citeseer_ngh_all, comp_citeseer_ngh_all[0])
    # norm(comp_pubmed_ngh_all, comp_pubmed_ngh_all[0])
    # norm(mem_cora_ngh_all, mem_cora_ngh_all[0])
    # norm(mem_citeseer_ngh_all, mem_citeseer_ngh_all[0])
    # norm(mem_pubmed_ngh_all, mem_pubmed_ngh_all[0])
    norm(comp_cora_ngh_all, comp_cora_ngh_delta[0])
    norm(comp_citeseer_ngh_all, comp_citeseer_ngh_delta[0])
    norm(comp_pubmed_ngh_all, comp_pubmed_ngh_delta[0])
    norm(mem_cora_ngh_all, mem_cora_ngh_delta[0])
    norm(mem_citeseer_ngh_all, mem_citeseer_ngh_delta[0])
    norm(mem_pubmed_ngh_all, mem_pubmed_ngh_delta[0])

    ## ngh_delta
    accumulate(comp_cora_ngh_delta)
    accumulate(comp_citeseer_ngh_delta)
    accumulate(comp_pubmed_ngh_delta)
    accumulate(mem_cora_ngh_delta)
    accumulate(mem_citeseer_ngh_delta)
    accumulate(mem_pubmed_ngh_delta)

    # normalize
    norm(comp_cora_ngh_delta, comp_cora_ngh_delta[0])
    norm(comp_citeseer_ngh_delta, comp_citeseer_ngh_delta[0])
    norm(comp_pubmed_ngh_delta, comp_pubmed_ngh_delta[0])
    norm(mem_cora_ngh_delta, mem_cora_ngh_delta[0])
    norm(mem_citeseer_ngh_delta, mem_citeseer_ngh_delta[0])
    norm(mem_pubmed_ngh_delta, mem_pubmed_ngh_delta[0])

    import seaborn as sns
    import matplotlib.ticker as mtick

    labels = ['Inital', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
    items = ['Full-graph updating', 'Full-nghs updating', 'Delta updating']
    """
    Computation
    """
    # data = [
    #     comp_cora, comp_citeseer, comp_pubmed, comp_cora_ngh_all, comp_citeseer_ngh_all,
    #     comp_pubmed_ngh_all, comp_cora_ngh_delta, comp_citeseer_ngh_delta, comp_pubmed_ngh_delta
    # ]

    data = [comp_cora, comp_cora_ngh_all, comp_cora_ngh_delta]

    # Only count seven shotsnaps
    for i in range(len(data)):
        data[i] = data[i][:8]

    # # Print combination(computation) reduction of each dataset
    # comb_reduct_cora = sum(comp_cora) / sum(comp_cora_ngh_delta)
    # comb_reduct_citeseer = sum(comp_citeseer) / sum(comp_citeseer_ngh_delta)
    # comb_reduct_pubmed = sum(comp_pubmed) / sum(comp_pubmed_ngh_delta)
    # print('\nCombination reduction: \ncora: {:.2%}, citeseer: {:.2%}, pubmed: {:.2%}'.format(
    #     comb_reduct_cora, comb_reduct_citeseer, comb_reduct_pubmed))

    # Group size in each label
    group_size = len(items)

    total_width = 1.5
    label_num = len(labels)
    width = total_width / label_num
    # Bar offset
    offset = -width / 2 * (group_size - 1)
    # Bar interval
    space = 0

    x = np.arange(label_num)  # the label locations
    x = x - ((group_size - 1) / 2) * width

    span = 1
    color_1 = sns.cubehelix_palette(start=2.8, rot=-.1, n_colors=span, dark=0.2, light=0.65)

    color_2 = sns.cubehelix_palette(start=1, rot=-.1, n_colors=span, dark=0.2, light=0.65)

    color_3 = sns.cubehelix_palette(start=1.8, rot=-.1, n_colors=span, dark=0.2, light=0.65)

    fig, ax1 = plt.subplots(figsize=(5, 3), dpi=600)
    # ax1 = plt.subplot(1, 3, 1)

    rects = [0 for n in range(len(items))]
    for i in range(len(items)):
        if i < span:
            rects[i] = ax1.bar(x + i * (width + space) + offset,
                               data[i],
                               width=width,
                               alpha=.99,
                               edgecolor='w',
                               label=items[i],
                               zorder=2,
                               color=color_1[i])
        elif i >= span and i < span * 2:
            rects[i] = ax1.bar(x + i * (width + space) + offset,
                               data[i],
                               width=width,
                               alpha=.99,
                               edgecolor='w',
                               label=items[i],
                               zorder=2,
                               color=color_2[i - span])
        else:
            rects[i] = ax1.bar(x + i * (width + space) + offset,
                               data[i],
                               width=width,
                               alpha=.99,
                               edgecolor='w',
                               label=items[i],
                               zorder=2,
                               color=color_3[i - span * 2])

    plt.grid(axis='y', zorder=1)

    # Plot图例
    plt.legend(fontsize=8)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(), 1)
            ax1.annotate(
                '{}'.format(height),
                fontsize=8,
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

    ax1.set_ylim([6e-1, 1.5e3])
    ax1.set_yscale('log')

    # my_y_ticks = np.arange(0, 120, 20)
    # plt.yticks(my_y_ticks)

    plt.legend(labelspacing=0, handlelength=1, fontsize=13, loc="best")
    # ax1.get_legend().remove()

    #plt.axhline(y=1, color='k', linestyle='-', linewidth=0.8)
    # fmt = '%.f%%'
    # yticks = mtick.FormatStrFormatter(fmt)
    # ax1.yaxis.set_major_formatter(yticks)

    fontsize = 15
    #plt.xlabel('Unroll number',fontsize = fontsize)
    plt.ylabel('Accumulated computing', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('./figure/retrain_comp_delta.pdf', dpi=600, bbox_inches="tight", pad_inches=0)
    """
    Mem access
    """

    # data = [mem_cora, mem_citeseer, mem_pubmed]
    data = [mem_cora, mem_cora_ngh_all, mem_cora_ngh_delta]

    # Only count seven shotsnaps
    for i in range(len(data)):
        data[i] = data[i][:8]

    # # Print aggregation (mem access) reduction of each dataset
    # aggr_reduct_cora = sum(mem_cora) / sum(mem_cora_ngh_delta)
    # aggr_reduct_citeseer = sum(mem_citeseer) / sum(mem_citeseer_ngh_delta)
    # aggr_reduct_pubmed = sum(mem_pubmed) / sum(mem_pubmed_ngh_delta)
    # print('\nAggregation reduction: \ncora: {:.2%}, citeseer: {:.2%}, pubmed: {:.2%}'.format(
    #     aggr_reduct_cora, aggr_reduct_citeseer, aggr_reduct_pubmed))

    # Group size in each label
    group_size = len(items)

    total_width = 1.5
    label_num = len(labels)
    width = total_width / label_num
    # Bar offset
    offset = -width / 2 * (group_size - 1)
    # Bar interval
    space = 0

    x = np.arange(label_num)  # the label locations
    x = x - ((group_size - 1) / 2) * width

    span = 1
    color_1 = sns.cubehelix_palette(start=2.8, rot=-.1, n_colors=span, dark=0.2, light=0.65)

    color_2 = sns.cubehelix_palette(start=1, rot=-.1, n_colors=span, dark=0.2, light=0.65)

    color_3 = sns.cubehelix_palette(start=1.8, rot=-.1, n_colors=span, dark=0.2, light=0.65)

    fig, ax1 = plt.subplots(figsize=(5, 3), dpi=600)
    # ax1 = plt.subplot(1, 3, 1)

    rects = [0 for n in range(len(items))]
    for i in range(len(items)):
        if i < span:
            rects[i] = ax1.bar(x + i * (width + space) + offset,
                               data[i],
                               width=width,
                               alpha=.99,
                               edgecolor='w',
                               label=items[i],
                               zorder=2,
                               color=color_1[i])
        elif i >= span and i < span * 2:
            rects[i] = ax1.bar(x + i * (width + space) + offset,
                               data[i],
                               width=width,
                               alpha=.99,
                               edgecolor='w',
                               label=items[i],
                               zorder=2,
                               color=color_2[i - span])
        else:
            rects[i] = ax1.bar(x + i * (width + space) + offset,
                               data[i],
                               width=width,
                               alpha=.99,
                               edgecolor='w',
                               label=items[i],
                               zorder=2,
                               color=color_3[i - span * 2])

    plt.grid(axis='y', zorder=1)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(), 1)
            ax1.annotate(
                '{}'.format(height),
                fontsize=8,
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center',
                va='bottom',
                rotation=90)

    # for i in range(len(items)):
    #     autolabel(rects[i])

    # Plot图例
    plt.legend(fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, va="center", position=(0, -0.05), fontsize=14)
    # ax1.xaxis.set_visible(False)
    # ax1.xaxis.set_tick_params(direction='in')
    # ax1.axes.get_xaxis().set_visible(False)

    # plt.yticks(size=10)
    # plt.xticks(size = 10)

    ax1.set_ylim([6e-1, 1.5e3])
    ax1.set_yscale('log')

    # my_y_ticks = np.arange(0, 120, 20)
    # plt.yticks(my_y_ticks)

    plt.legend(labelspacing=0, handlelength=1, fontsize=13, loc="best")
    # ax1.get_legend().remove()

    #plt.axhline(y=1, color='k', linestyle='-', linewidth=0.8)
    # fmt = '%.f%%'
    # yticks = mtick.FormatStrFormatter(fmt)
    # ax1.yaxis.set_major_formatter(yticks)

    fontsize = 15
    #plt.xlabel('Unroll number',fontsize = fontsize)
    plt.ylabel('Accumulated accessing', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('./figure/retrain_mem_delta.pdf', dpi=600, bbox_inches="tight", pad_inches=0)


# plt_full_retrain()
# plt_delta_retrain()
plt_delta_retrain_cora()