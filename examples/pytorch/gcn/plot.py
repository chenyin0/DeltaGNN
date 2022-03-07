import matplotlib.pyplot as plt
import numpy as np


def plt_edge_epoch():
    cora = np.loadtxt('./results/cora_add_edge.txt', delimiter=',')
    citeseer = np.loadtxt('./results/citeseer_add_edge.txt', delimiter=',')
    pubmed = np.loadtxt('./results/pubmed_add_edge.txt', delimiter=',')

    x = cora[:, 0]
    plt.plot(x, cora[:, 1], 'o-', label="Cora")
    plt.plot(x, citeseer[:, 1], 's-', label="Citeseer")
    plt.plot(x, pubmed[:, 1], 'v-', label="Pubmed")

    plt.plot(x, cora[:, 2], 'o-', label="Cora_retrain")
    plt.plot(x, citeseer[:, 2], 's-', label="Citeseer_retrain")
    plt.plot(x, pubmed[:, 2], 'v-', label="Pubmed_retrain")

    plt.legend(prop={'size': 16})
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('New edge num', fontsize=16)
    plt.show()


# plt_edge_epoch()