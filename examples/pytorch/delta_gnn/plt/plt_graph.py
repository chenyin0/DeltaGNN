import matplotlib.pyplot as plt
import numpy as np
# import igraph as ig
from scipy.sparse import csr_matrix
import time
from brokenaxes import brokenaxes
from matplotlib import ticker
from matplotlib.pyplot import MultipleLocator

# def graph_visualize(indptr, indices, data):
#     print(">>> Graph visualize")
#     begin = time.time()
#     if (data is None):
#         data = np.ones(len(indices))

#     mat_size = len(indptr) - 1
#     adj = csr_matrix((data, indices, indptr), shape=(mat_size, mat_size))

#     # g = igraph.Graph()
#     # print(adj.toarray())
#     g = ig.Graph.Adjacency(adj.toarray(), mode="undirected")
#     # print(g)
#     # g.Adjacency(adj, mode="directed")
#     # g._graph_from_sparse_matrix(adj, mode="directed")

#     ig.Graph.community_fastgreedy(g)
#     layout = g.layout("drl")

#     import matplotlib.pyplot as plt
#     print(">>> Plot fig")
#     fig, ax = plt.subplots()
#     ig.plot(g, target=ax, layout=layout)
#     # plt.savefig("test.svg")
#     end = time.time()
#     print("Time cost: ", end - begin)
#     plt.show()

#     # img.save("img.pdf")

#     # fig, ax = plt.subplots()
#     # ig.plot(g, layout=layout, target=ax)
#     # ig.plot(g, layout=layout)

#     # igraph.plot(g, layout)
#     # fig.show()

# def plot_degree_distribution(degree_list):
#     node_totol_num = 0
#     degree_list = degree_list[1:]
#     for i in degree_list:
#         node_totol_num += i

#     x = [i for i in range(len(degree_list))]
#     y = [round(degree_list[i] * 100 / node_totol_num) for i in range(len(degree_list))]
#     # y = [degree_list[i] for i in range(len(degree_list))]

#     fig = plt.figure(figsize=(3, 3.3))
#     bax = brokenaxes(xlims=((0, 22), (165, 170)), hspace=1, despine=False)
#     bax.bar(x, y)

#     for ax in bax.axs:
#         # ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
#         ax.xaxis.set_tick_params(labelsize=13)
#         ax.yaxis.set_tick_params(labelsize=13)

#         # ax.xaxis.set_tick_params(bottom=False)
#         # ax.yaxis.set_tick_params(bottom=False)

#         ax.xaxis.set_major_locator(MultipleLocator(5))

#     bax.set_xlabel('Vertex degree', labelpad=20, fontsize=16)
#     bax.set_ylabel('Vertex Distribution (%)', labelpad=25, fontsize=16)

#     plt.savefig('./figure/degree_distribution/deg_dist' + dataset_name + '.pdf',
#                 dpi=600,
#                 bbox_inches="tight",
#                 pad_inches=0)


def plot_degree_distribution(vertex_num, degree_list, dataset_name):
    node_totol_num = vertex_num

    x = [i for i in range(len(degree_list))]
    y = [round(len(degree_list[i]) * 100 / node_totol_num, 2) for i in range(len(degree_list))]
    # y = [degree_list[i] for i in range(len(degree_list))]

    # Skip deg = 0
    x = x[1:]
    y = y[1:]

    deg_broken = 0
    for i, deg_rate in enumerate(y):
        if deg_rate < 1:
            deg_broken = i
            break
    deg_broken += 10
    deg_max = len(degree_list)
    # deg_interval = round(deg_max / 6)

    fig = plt.figure(figsize=(3, 3.3))
    bax = brokenaxes(xlims=((0, deg_broken), (deg_max, deg_max + 5)), hspace=1, despine=False)
    # bax = brokenaxes(xlims=((0, 50), (3320, 3400)), hspace=1, despine=False)
    bax.bar(x, y)

    for ax in bax.axs:
        # ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        ax.xaxis.set_tick_params(labelsize=13)
        ax.yaxis.set_tick_params(labelsize=13)

        # ax.xaxis.set_tick_params(bottom=False)
        # ax.yaxis.set_tick_params(bottom=False)

        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.tick_params('y', labelsize=10)
        ax.tick_params('x', labelrotation=30, labelsize=10)

    bax.set_xlabel('Vertex degree', labelpad=20, fontsize=14)
    bax.set_ylabel('Vertex Distribution (%)', labelpad=25, fontsize=14)

    plt.savefig('./figure/degree_distribution/deg_dist_' + dataset_name + '.pdf',
                dpi=600,
                bbox_inches="tight",
                pad_inches=0)


def plot_node_degree(degree_list, dataset_name):
    r"""
    Plot degree distribution
    """

    for i in range(len(degree_list)):
        degree_list[i] = min(degree_list[i], 1000)

    plt.bar(range(len(degree_list)), degree_list)
    plt.show()

    plt.savefig('./figure/node_degree_' + dataset_name + '.pdf',
                dpi=600,
                bbox_inches="tight",
                pad_inches=0)
