import matplotlib.pyplot as plt
import numpy as np


def plt_edge_epoch():
    cora = np.loadtxt('./results/cora_add_edge_deg_30.txt', delimiter=',')
    citeseer = np.loadtxt('./results/citeseer_add_edge_deg_30.txt', delimiter=',')
    pubmed = np.loadtxt('./results/pubmed_add_edge_deg_30.txt', delimiter=',')

    epoch_num_cora = cora.shape[0] - 1
    epoch_num_citeseer = citeseer.shape[0] - 1
    epoch_num_pubmed = pubmed.shape[0] - 1

    # Drop the last epoch for no inserted node
    epoch_num_cora -= 1
    epoch_num_citeseer -= 1
    epoch_num_pubmed -= 1

    x1 = range(epoch_num_cora)
    x2 = range(epoch_num_citeseer)
    x3 = range(epoch_num_pubmed)

    fig = plt.figure(figsize=(10, 3), dpi=100)

    ax1 = plt.subplot(1, 3, 1)
    # plt.plot(x1, cora[:, -2], 'v-', color='g', label="Delta")
    plt.plot(x1, cora[:epoch_num_cora, -1], 'o-', color='b', label="Delta retrain")
    plt.plot(x1, cora[:epoch_num_cora, -3], 'o-', color='r', label="Full graph retrain")
    plt.plot(x1, cora[:epoch_num_cora, -4], 'o-', color='grey', label="W/O retrain")
    # plt.legend(prop={'size': 10})
    ax1.set_ylim([0, 100])
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.title('Cora', fontsize=16)

    ax2 = plt.subplot(1, 3, 2)
    # plt.plot(x2, citeseer[:, -2], 'v-', color='g', label="Delta")
    plt.plot(x2, citeseer[:epoch_num_citeseer, -1], 'v-', color='b', label="Delta retrain")
    plt.plot(x2, citeseer[:epoch_num_citeseer, -3], 'o-', color='r', label="Full graph retrain")
    plt.plot(x2, citeseer[:epoch_num_citeseer, -4], 'o-', color='grey', label="W/O retrain")
    # plt.legend(prop={'size': 10})
    ax2.set_ylim([0, 100])
    # plt.ylabel('Accuracy (%)', fontsize=16)
    plt.title('Citeseer', fontsize=16)

    ax3 = plt.subplot(1, 3, 3)
    # plt.plot(x3, pubmed[:, -2], 'v-', color='g', label="Delta")
    plt.plot(x3, pubmed[:epoch_num_pubmed, -1], 'v-', color='b', label="Delta retrain")
    plt.plot(x3, pubmed[:epoch_num_pubmed, -3], 'o-', color='r', label="Full graph retrain")
    plt.plot(x3, pubmed[:epoch_num_pubmed, -4], 'o-', color='grey', label="W/O retrain")
    plt.legend(prop={'size': 12})
    ax3.set_ylim([0, 100])
    # plt.ylabel('Accuracy (%)', fontsize=16)
    plt.title('Pubmed', fontsize=16)

    plt.suptitle('Accuracy with graph evolving', fontsize=18, y=1.07)

    # # Plot legend for all subplot
    # lines, labels = fig.axes[-1].get_legend_handles_labels()
    # fig.legend(lines, labels, ncol=4, loc='upper center', prop={'size': 10})

    # ax1.set_ylim([0, 100])
    # ax2.set_ylim([0, 100])
    # ax3.set_ylim([40, 100])

    # plt.show()

    plt.tight_layout()
    plt.savefig('./figure/acc_degrad.pdf', dpi=600, bbox_inches="tight", pad_inches=0)


plt_edge_epoch()