import matplotlib.pyplot as plt
import numpy as np


def plt_edge_epoch():
    cora_evo = np.loadtxt('./results/cora_accuracy_evo.txt', delimiter=',')
    citeseer_evo = np.loadtxt('./results/citeseer_accuracy_evo.txt', delimiter=',')
    pubmed_evo = np.loadtxt('./results/pubmed_accuracy_evo.txt', delimiter=',')

    cora_evo_delta = np.loadtxt('./results/cora_accuracy_evo_delta_30.txt', delimiter=',')
    citeseer_evo_delta = np.loadtxt('./results/citeseer_accuracy_evo_delta_30.txt', delimiter=',')
    pubmed_evo_delta = np.loadtxt('./results/pubmed_accuracy_evo_delta_30.txt', delimiter=',')

    epoch_num_cora = cora_evo.shape[0] - 1
    epoch_num_citeseer = citeseer_evo.shape[0] - 1
    epoch_num_pubmed = pubmed_evo.shape[0] - 1

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
    plt.plot(x1, cora_evo_delta[:epoch_num_cora, 2], 'o-', color='b', label="Delta retrain")
    plt.plot(x1, cora_evo[:epoch_num_cora, 3], 'o-', color='r', label="Full graph retrain")
    plt.plot(x1, cora_evo[:epoch_num_cora, 2], 'o-', color='grey', label="W/O retrain")
    # plt.legend(prop={'size': 10})
    ax1.set_ylim([0, 100])
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.title('Cora', fontsize=16)

    ax2 = plt.subplot(1, 3, 2)
    # plt.plot(x2, citeseer[:, -2], 'v-', color='g', label="Delta")
    plt.plot(x2, citeseer_evo_delta[:epoch_num_citeseer, 2], 'v-', color='b', label="Delta retrain")
    plt.plot(x2, citeseer_evo[:epoch_num_citeseer, 3], 'o-', color='r', label="Full graph retrain")
    plt.plot(x2, citeseer_evo[:epoch_num_citeseer, 2], 'o-', color='grey', label="W/O retrain")
    # plt.legend(prop={'size': 10})
    ax2.set_ylim([0, 100])
    # plt.ylabel('Accuracy (%)', fontsize=16)
    plt.title('Citeseer', fontsize=16)

    ax3 = plt.subplot(1, 3, 3)
    # plt.plot(x3, pubmed[:, -2], 'v-', color='g', label="Delta")
    plt.plot(x3, pubmed_evo_delta[:epoch_num_pubmed, 2], 'v-', color='b', label="Delta retrain")
    plt.plot(x3, pubmed_evo[:epoch_num_pubmed, 3], 'o-', color='r', label="Full graph retrain")
    plt.plot(x3, pubmed_evo[:epoch_num_pubmed, 2], 'o-', color='grey', label="W/O retrain")
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