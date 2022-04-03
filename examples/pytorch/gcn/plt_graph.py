import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
from scipy.sparse import csr_matrix
import time


def graph_visualize(indptr, indices, data):
    print(">>> Graph visualize")
    begin = time.time()
    if (data is None):
        data = np.ones(len(indices))

    mat_size = len(indptr) - 1
    adj = csr_matrix((data, indices, indptr), shape=(mat_size, mat_size))

    # g = igraph.Graph()
    # print(adj.toarray())
    g = ig.Graph.Adjacency(adj.toarray(), mode="undirected")
    # print(g)
    # g.Adjacency(adj, mode="directed")
    # g._graph_from_sparse_matrix(adj, mode="directed")

    ig.Graph.community_fastgreedy(g)
    layout = g.layout("drl")

    import matplotlib.pyplot as plt
    print(">>> Plot fig")
    fig, ax = plt.subplots()
    ig.plot(g, target=ax, layout=layout)
    # plt.savefig("test.svg")
    end = time.time()
    print("Time cost: ", end - begin)
    plt.show()

    # img.save("img.pdf")

    # fig, ax = plt.subplots()
    # ig.plot(g, layout=layout, target=ax)
    # ig.plot(g, layout=layout)

    # igraph.plot(g, layout)
    # fig.show()