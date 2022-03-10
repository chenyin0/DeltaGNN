from collections import Counter
from numpy import int64
# from msilib import sequence
import torch as th
import dgl
from dgl.data.utils import generate_mask_tensor
from dgl.data.citation_graph import _sample_mask
import math
import random


def gen_root_node_queue(g):
    """
    Gen root node according to in-degree
    """
    src_edges = g.edges()[0].numpy().tolist()
    nodes = Counter(src_edges).most_common(len(src_edges))
    nodes = [x[0] for x in nodes]
    return nodes


def bfs_traverse(g_csr, root_node_q):
    """
    Gen node evolve sequence
    """

    # g_csr = g.adj_sparse('csr')
    indptr = g_csr[0]
    indices = g_csr[1]

    queue = []
    seen = set()
    sequence = list()
    while len(root_node_q) > 0:
        root = root_node_q.pop(0)
        queue.append(root)
        seen.add(root)
        sequence.append(root)
        while len(queue) > 0:
            vetex = queue.pop(0)
            begin = indptr[root]
            end = indptr[root + 1]
            nodes = indices[begin:end].tolist()
            # print(nodes)
            for w in nodes:
                # print(w)
                if w not in seen:
                    queue.append(w)
                    seen.add(w)
                    sequence.append(w)
                # print(len(seen), len(queue))

        # Pop nodes in root_node_q which have been visited
        while len(root_node_q) > 0 and root_node_q[0] in seen:
            root_node_q.pop(0)
        # print('>> seen size: ', len(seen), 'seq size: ', len(sequence))

    return sequence


def graph_evolve(new_nodes,
                 g_orig_csr,
                 g_orig,
                 node_map_orig2evo,
                 node_map_evo2orig,
                 g_evo=None):
    """
    Construct evolve graph from an orginal static graph
    """

    indptr = g_orig_csr[0]
    indices = g_orig_csr[1]

    edge_src_nodes = list()
    edge_dst_nodes = list()
    for node in new_nodes:
        # e_dest_nodes = indices[indptr[node]:indptr[node + 1]]
        # e_src_nodes = th.linspace(node, node, edge_dest_nodes.size()[0])
        e_src_nodes = indices[indptr[node]:indptr[node + 1]]
        e_dst_nodes = th.linspace(node, node, len(e_src_nodes))
        edge_dst_nodes.extend(e_src_nodes.numpy().tolist())
        edge_src_nodes.extend(e_dst_nodes.numpy().tolist())

    # Remapping node_id from g_orig -> g_evo, and record mapping from g_evo -> g_orig
    edge_src_nodes_reindex = []
    edge_dst_nodes_reindex = []
    for node in edge_src_nodes:
        node_id_evo = node_reindex(node_map_orig2evo, node)
        edge_src_nodes_reindex.append(node_id_evo)
        if node_id_evo not in node_map_evo2orig:
            node_map_evo2orig[node_id_evo] = node

    for node in edge_dst_nodes:
        node_id_evo = node_reindex(node_map_orig2evo, node)
        edge_dst_nodes_reindex.append(node_id_evo)
        if node_id_evo not in node_map_evo2orig:
            node_map_evo2orig[node_id_evo] = node

    edge_src_nodes_reindex = th.tensor(edge_src_nodes_reindex, dtype=th.int64)
    edge_dst_nodes_reindex = th.tensor(edge_dst_nodes_reindex, dtype=th.int64)

    if g_evo is None:
        # Construct a new graph
        g_evo = dgl.graph((edge_src_nodes_reindex, edge_dst_nodes_reindex))
    else:
        g_evo.add_edges(th.tensor(edge_src_nodes_reindex),
                        th.tensor(edge_dst_nodes_reindex))

    # print('\n>> g_evo', g_evo)

    features_evo, labels_evo, train_mask_evo, val_mask_evo, test_mask_evo = update_g_evo(
        g_evo, g_orig, node_map_evo2orig)

    return g_evo, features_evo, labels_evo, train_mask_evo, val_mask_evo, test_mask_evo


def node_reindex(node_map, node_id_old):
    """
    node_id(g_orig) -> node_id(g_evo)
    """
    if node_id_old in node_map:
        node_id_new = node_map[node_id_old]
    else:
        node_id_new = len(node_map)
        node_map[node_id_old] = node_id_new

    return node_id_new


def update_g_evo(g_evo, g_orig, node_map):
    """
    Update feature and train/eval/test mask
    """
    nodes_orig_index = []
    for node in g_evo.nodes().tolist():
        nodes_orig_index.append(node_map[node])

    features = g_orig.ndata['feat'][nodes_orig_index, :]
    # if 'feat' in g_evo.ndata:
    #     feat = g_evo.ndata['feat']
    #     features = th.cat((feat, feat_new), 0)
    # else:
    #     features = feat_new
    g_evo.ndata['feat'] = features

    labels = g_orig.ndata['label'][nodes_orig_index]
    # if 'label' in g_evo.ndata:
    #     label = g_evo.ndata['label']
    #     labels = th.cat((label, label_new), 0)
    # else:
    #     labels = label_new
    g_evo.ndata['label'] = labels

    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    idx_train = range(math.floor(labels.size()[0] * train_ratio))
    train_mask = generate_mask_tensor(_sample_mask(idx_train, labels.shape[0]))
    g_evo.ndata['train_mask'] = train_mask

    idx_val = range(len(idx_train),
                    len(idx_train) + math.floor(labels.size()[0] * val_ratio))
    val_mask = generate_mask_tensor(_sample_mask(idx_val, labels.shape[0]))
    g_evo.ndata['val_mask'] = val_mask

    # idx_test = range(len(idx_val),
    #                  len(idx_val) + math.floor(labels.size()[0] * test_ratio))
    # test_mask = generate_mask_tensor(_sample_mask(idx_test, labels.shape[0]))
    # g_evo.ndata['test_mask'] = test_mask

    loc_list = range(labels.size()[0])
    idx_test = random.sample(loc_list, math.floor(labels.size()[0] * test_ratio))
    idx_test = idx_test.sort()
    test_mask = generate_mask_tensor(_sample_mask(idx_test, labels.shape[0]))
    g_evo.ndata['test_mask'] = test_mask

    return features, labels, train_mask, val_mask, test_mask
