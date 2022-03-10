import matplotlib.pyplot as plt
import numpy as np


def plt_edge_epoch():
    cora = np.loadtxt('./results/cora_add_edge.txt', delimiter=',')
    citeseer = np.loadtxt('./results/citeseer_add_edge.txt', delimiter=',')
    pubmed = np.loadtxt('./results/pubmed_add_edge.txt', delimiter=',')

    x1 = cora[:, 0]
    x2 = citeseer[:, 0]
    x3 = pubmed[:, 0]
    plt.plot(x1, cora[:, 1], 'o-', color='r', label="Cora")
    plt.plot(x2, citeseer[:, 1], 's-', color='b', label="Citeseer")
    plt.plot(x3, pubmed[:, 1], 'v-', color='g', label="Pubmed")

    plt.plot(x1, cora[:, 2], 'o-', color='r', label="Cora_retrain")
    plt.plot(x2, citeseer[:, 2], 's-', color='b', label="Citeseer_retrain")
    plt.plot(x3, pubmed[:, 2], 'v-', color='g', label="Pubmed_retrain")

    plt.legend(prop={'size': 16})
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('New edge num', fontsize=16)
    plt.show()


plt_edge_epoch()