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


def update_g_struct(new_nodes,
                    g_orig_csr,
                    node_map_orig2evo,
                    node_map_evo2orig,
                    g_evo=None):

    indptr = g_orig_csr[0]
    indices = g_orig_csr[1]

    edge_src_nodes = list()
    edge_dst_nodes = list()
    for node in new_nodes:
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

    # Remove redundant edges
    edge_src_nodes_tmp = []
    edge_dst_nodes_tmp = []
    if g_evo is not None:
        for i in range(len(edge_src_nodes_reindex)):
            src_v_id = edge_src_nodes_reindex[i]
            des_v_id = edge_dst_nodes_reindex[i]
            if not (g_evo.has_nodes(src_v_id) and g_evo.has_nodes(des_v_id)
                    and g_evo.has_edges_between(src_v_id, des_v_id)):
                edge_src_nodes_tmp.append(edge_src_nodes_reindex[i])
                edge_dst_nodes_tmp.append(edge_dst_nodes_reindex[i])

        edge_src_nodes_reindex = edge_src_nodes_tmp
        edge_dst_nodes_reindex = edge_dst_nodes_tmp

    # if g_evo is not None:
    #     for i, item in enumerate(edge_src_nodes_reindex[:]):
    #         src_v_id = edge_src_nodes_reindex[i]
    #         des_v_id = edge_dst_nodes_reindex[i]
    #         if g_evo.has_nodes(src_v_id) and g_evo.has_nodes(
    #                 des_v_id) and g_evo.has_edges_between(src_v_id, des_v_id):
    #             edge_src_nodes_tmp.append(edge_src_nodes_reindex[i])
    #             edge_dst_nodes_tmp.append(edge_dst_nodes_reindex[i])

    #     edge_src_nodes_reindex = edge_src_nodes_tmp
    #     edge_dst_nodes_reindex = edge_dst_nodes_tmp

    edge_src_nodes_reindex = th.tensor(edge_src_nodes_reindex, dtype=th.int64)
    edge_dst_nodes_reindex = th.tensor(edge_dst_nodes_reindex, dtype=th.int64)

    if g_evo is None:
        # Construct a new graph
        g_evo = dgl.graph((edge_src_nodes_reindex, edge_dst_nodes_reindex))
    else:
        g_evo.add_edges(th.tensor(edge_src_nodes_reindex),
                        th.tensor(edge_dst_nodes_reindex))

    return g_evo


def graph_evolve(new_nodes,
                 g_orig_csr,
                 g_orig,
                 node_map_orig2evo,
                 node_map_evo2orig,
                 g_evo=None):
    """
    Construct evolve graph from an orginal static graph
    """

    # indptr = g_orig_csr[0]
    # indices = g_orig_csr[1]

    # edge_src_nodes = list()
    # edge_dst_nodes = list()
    # for node in new_nodes:
    #     e_src_nodes = indices[indptr[node]:indptr[node + 1]]
    #     e_dst_nodes = th.linspace(node, node, len(e_src_nodes))
    #     edge_dst_nodes.extend(e_src_nodes.numpy().tolist())
    #     edge_src_nodes.extend(e_dst_nodes.numpy().tolist())

    # # Remapping node_id from g_orig -> g_evo, and record mapping from g_evo -> g_orig
    # edge_src_nodes_reindex = []
    # edge_dst_nodes_reindex = []
    # for node in edge_src_nodes:
    #     node_id_evo = node_reindex(node_map_orig2evo, node)
    #     edge_src_nodes_reindex.append(node_id_evo)
    #     if node_id_evo not in node_map_evo2orig:
    #         node_map_evo2orig[node_id_evo] = node

    # for node in edge_dst_nodes:
    #     node_id_evo = node_reindex(node_map_orig2evo, node)
    #     edge_dst_nodes_reindex.append(node_id_evo)
    #     if node_id_evo not in node_map_evo2orig:
    #         node_map_evo2orig[node_id_evo] = node

    # edge_src_nodes_reindex = th.tensor(edge_src_nodes_reindex, dtype=th.int64)
    # edge_dst_nodes_reindex = th.tensor(edge_dst_nodes_reindex, dtype=th.int64)

    # if g_evo is None:
    #     # Construct a new graph
    #     g_evo = dgl.graph((edge_src_nodes_reindex, edge_dst_nodes_reindex))
    # else:
    #     g_evo.add_edges(th.tensor(edge_src_nodes_reindex),
    #                     th.tensor(edge_dst_nodes_reindex))

    if g_evo is None:
        # Construct a new graph
        g_evo = update_g_struct(new_nodes, g_orig_csr, node_map_orig2evo,
                                node_map_evo2orig)
    else:
        g_evo = update_g_struct(new_nodes, g_orig_csr, node_map_orig2evo,
                                node_map_evo2orig, g_evo)

    # print('\n>> g_evo', g_evo)

    update_g_attribute(g_evo, g_orig, node_map_evo2orig)
    return g_evo


def graph_evolve_delta(new_nodes,
                       g_orig_csr,
                       g_orig,
                       node_map_orig2evo,
                       node_map_evo2orig,
                       g_evo=None):
    """
    Construct evolve graph from an orginal static graph
    """

    if g_evo is None:
        # Construct a new graph
        g_evo = update_g_struct(new_nodes, g_orig_csr, node_map_orig2evo,
                                node_map_evo2orig)
    else:
        g_evo = update_g_struct(new_nodes, g_orig_csr, node_map_orig2evo,
                                node_map_evo2orig, g_evo)

    # print('\n>> g_evo', g_evo)

    update_g_attribute_delta(new_nodes, g_evo, g_orig, node_map_evo2orig,
                             node_map_orig2evo)
    return g_evo


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


def update_g_attribute(g_evo, g_orig, node_map_evo2orig):
    """
    Update feature and train/eval/test mask
    """
    # Get orig_node_index of evo_node_index
    nodes_orig_index = []
    for node in g_evo.nodes().tolist():
        nodes_orig_index.append(node_map_evo2orig[node])

    features = g_orig.ndata['feat'][nodes_orig_index, :]
    g_evo.ndata['feat'] = features

    labels = g_orig.ndata['label'][nodes_orig_index]
    g_evo.ndata['label'] = labels

    train_ratio = 0.06
    val_ratio = 0.15
    test_ratio = 0.3

    # idx_train = range(math.floor(labels.size()[0] * train_ratio))
    # train_mask = generate_mask_tensor(_sample_mask(idx_train, labels.shape[0]))
    # g_evo.ndata['train_mask'] = train_mask

    loc_list = range(labels.size()[0])
    idx_train = random.sample(loc_list,
                              math.floor(labels.size()[0] * train_ratio))
    idx_train.sort()
    train_mask = generate_mask_tensor(_sample_mask(idx_train, labels.shape[0]))
    g_evo.ndata['train_mask'] = train_mask

    # idx_val = range(len(idx_train),
    #                 len(idx_train) + math.floor(labels.size()[0] * val_ratio))
    # val_mask = generate_mask_tensor(_sample_mask(idx_val, labels.shape[0]))
    # g_evo.ndata['val_mask'] = val_mask

    loc_list = range(labels.size()[0])
    idx_val = random.sample(loc_list, math.floor(labels.size()[0] * val_ratio))
    idx_val.sort()
    val_mask = generate_mask_tensor(_sample_mask(idx_val, labels.shape[0]))
    g_evo.ndata['val_mask'] = val_mask

    # idx_test = range(len(idx_val),
    #                  len(idx_val) + math.floor(labels.size()[0] * test_ratio))
    # test_mask = generate_mask_tensor(_sample_mask(idx_test, labels.shape[0]))
    # g_evo.ndata['test_mask'] = test_mask

    loc_list = range(labels.size()[0])
    idx_test = random.sample(loc_list,
                             math.floor(labels.size()[0] * test_ratio))
    idx_test.sort()
    test_mask = generate_mask_tensor(_sample_mask(idx_test, labels.shape[0]))
    g_evo.ndata['test_mask'] = test_mask


def update_g_attribute_delta(new_nodes, g_evo, g_orig, node_map_evo2orig,
                             node_map_orig2evo):
    """
    Update feature and eval/test mask, but set train_mask with the new inserted vertices for delta updating
    """
    # Get orig_node_index of evo_node_index
    nodes_orig_index = []
    for node in g_evo.nodes().tolist():
        nodes_orig_index.append(node_map_evo2orig[node])

    features = g_orig.ndata['feat'][nodes_orig_index, :]
    g_evo.ndata['feat'] = features

    labels = g_orig.ndata['label'][nodes_orig_index]
    g_evo.ndata['label'] = labels

    # train_ratio = 0.06
    val_ratio = 0.15
    test_ratio = 0.3

    # loc_list = range(labels.size()[0])
    # idx_train = random.sample(loc_list,
    #                           math.floor(labels.size()[0] * train_ratio))
    # idx_train.sort()
    # train_mask = generate_mask_tensor(_sample_mask(idx_train, labels.shape[0]))
    # g_evo.ndata['train_mask'] = train_mask
    """
    Training mask are set to new inserted vertices and its neighbor vi 
    """
    nodes_index_evo = []
    for node in new_nodes:
        nodes_index_evo.append(node_map_orig2evo[node])

    nodes_index_evo.sort()
    idx_train = nodes_index_evo
    train_mask = generate_mask_tensor(_sample_mask(idx_train, labels.shape[0]))
    g_evo.ndata['train_mask'] = train_mask

    loc_list = range(labels.size()[0])
    idx_val = random.sample(loc_list, math.floor(labels.size()[0] * val_ratio))
    idx_val.sort()
    val_mask = generate_mask_tensor(_sample_mask(idx_val, labels.shape[0]))
    g_evo.ndata['val_mask'] = val_mask

    loc_list = range(labels.size()[0])
    idx_test = random.sample(loc_list,
                             math.floor(labels.size()[0] * test_ratio))
    idx_test.sort()
    test_mask = generate_mask_tensor(_sample_mask(idx_test, labels.shape[0]))
    g_evo.ndata['test_mask'] = test_mask


def count_neighbor(nodes, g_csr, node_map_orig2evo, layer_num):
    """ 
    Count neighbor edges and vertices of specific node set
    """
    edge_set = set()
    node_set = set(nodes)
    indptr = g_csr[0].numpy().tolist()
    indices = g_csr[1].numpy().tolist()
    node_queue = nodes
    for layer_id in range(layer_num):
        node_num = len(node_queue)
        for i in range(node_num):
            node_id = node_queue[i]
            begin = indptr[node_id]
            end = indptr[node_id + 1]
            for edge_id in range(begin, end):
                node = indices[edge_id]
                if node in node_map_orig2evo:
                    node_queue.append(node)
                    node_set.add(node)
                    edge_set.add(edge_id)

            # s_tmp = set(indices[begin:end])
            # node_set.update(s_tmp)

        # Pop visited node
        node_queue = node_queue[node_num:]

    node_sum = len(node_set)
    edge_sum = len(edge_set)

    return node_sum, edge_sum