from collections import Counter
from numpy import int64
# from msilib import sequence
import torch as th
import dgl
from dgl.data.utils import generate_mask_tensor
from dgl.data.citation_graph import _sample_mask
import math
import random
import numpy as np


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


def get_ngh(g_csr, root_nodes):
    '''
    deg_th: node degree threshold, disable value = -1
    '''
    indptr = g_csr[0]
    indices = g_csr[1]
    ngh = []
    for root_node in root_nodes:
        begin = indptr[root_node]
        end = indptr[root_node + 1]
        ngh.extend(indices[begin:end].tolist())

    return ngh


def get_ngh_with_deg_th(g, root_nodes, deg_th):
    '''
    deg_th: node degree threshold
    '''
    g_csr = g.adj_sparse('csr')
    indptr = g_csr[0]
    indices = g_csr[1]
    # key = 'the ngh node', value = 'all neighbors of this ngh node'
    ngh_low_deg = dict()
    # key = 'the ngh node', value = 'the ngh node's neighbors which are inserted nodes'
    ngh_high_deg = dict()
    if deg_th < 0:
        raise ValueError("Degree threshold should be larger than 0")
    else:
        for root_node in root_nodes:
            begin = indptr[root_node]
            end = indptr[root_node + 1]
            ngh_nodes = indices[begin:end].tolist()
            for ngh_node in ngh_nodes:
                ngh_begin = indptr[ngh_node]
                ngh_end = indptr[ngh_node + 1]
                ngh_node_deg = ngh_end - ngh_begin
                if ngh_node_deg >= deg_th:
                    # Add all neighbors of the node with high degree
                    ngh_high_deg.setdefault(ngh_node,
                                            set()).update(indices[ngh_begin:ngh_end].tolist())
                else:
                    # Add neighbors of the node with low degree
                    # Ensure edge: ngh_node->root_node exists
                    if g.has_edges_between(ngh_node, root_node):
                        ngh_low_deg.setdefault(ngh_node, set()).add(root_node)

    return ngh_high_deg, ngh_low_deg


def gen_edge_mask(g, ngh_dict):
    src_nodes = []
    dst_nodes = []
    for root_node, ngh in ngh_dict.items():
        src_nodes.extend([root_node for i in range(len(ngh))])
        dst_nodes.extend(ngh)

    # edge_ids = g.edge_ids(th.Tensor(src_nodes).long(), th.Tensor(dst_nodes).long())
    edge_ids = g.edge_ids(th.tensor(src_nodes, dtype=th.long), th.tensor(dst_nodes, dtype=th.long))

    edge_ids = edge_ids.tolist()
    edge_mask = [0 for i in range(g.number_of_edges())]
    for id in edge_ids:
        edge_mask[id] = 1

    g.edata['edge_mask'] = th.Tensor(edge_mask)


def update_g_struct(new_nodes, g_orig_csr, node_map_orig2evo, node_map_evo2orig, g_evo=None):

    indptr = g_orig_csr[0].numpy().tolist()
    indices = g_orig_csr[1].numpy().tolist()

    edge_src_nodes = list()
    edge_dst_nodes = list()
    for node in new_nodes:
        e_src_nodes = indices[indptr[node]:indptr[node + 1]]
        # e_dst_nodes = th.linspace(node, node, len(e_src_nodes))
        e_dst_nodes = [node] * len(e_src_nodes)
        # edge_dst_nodes.extend(e_src_nodes.numpy().tolist())
        # edge_src_nodes.extend(e_dst_nodes.numpy().tolist())
        edge_dst_nodes.extend(e_src_nodes)
        edge_src_nodes.extend(e_dst_nodes)

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
        g_evo.add_edges(edge_src_nodes_reindex, edge_dst_nodes_reindex)

    return g_evo


def graph_evolve(new_nodes, g_orig_csr, g_orig, node_map_orig2evo, node_map_evo2orig, g_evo=None):
    """
    Construct evolve graph from an orginal static graph
    """
    if g_evo is None:
        # Construct a new graph
        g_evo = update_g_struct(new_nodes, g_orig_csr, node_map_orig2evo, node_map_evo2orig)
    else:
        g_evo = update_g_struct(new_nodes, g_orig_csr, node_map_orig2evo, node_map_evo2orig, g_evo)

    # print('\n>> g_evo', g_evo)

    update_g_attr(new_nodes, g_evo, g_orig, node_map_orig2evo, node_map_evo2orig)
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
        g_evo = update_g_struct(new_nodes, g_orig_csr, node_map_orig2evo, node_map_evo2orig)
    else:
        g_evo = update_g_struct(new_nodes, g_orig_csr, node_map_orig2evo, node_map_evo2orig, g_evo)

    # print('\n>> g_evo', g_evo)

    update_g_attr_delta(new_nodes, g_evo, g_orig, node_map_evo2orig, node_map_orig2evo)
    return g_evo


def graph_evolve_delta_all_ngh(new_nodes,
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
        g_evo = update_g_struct(new_nodes, g_orig_csr, node_map_orig2evo, node_map_evo2orig)
    else:
        g_evo = update_g_struct(new_nodes, g_orig_csr, node_map_orig2evo, node_map_evo2orig, g_evo)

    # print('\n>> g_evo', g_evo)

    update_g_attr_all_ngh(new_nodes, g_evo, g_orig, node_map_evo2orig, node_map_orig2evo)
    return g_evo


def node_reindex(node_map, node_id_prev):
    """
    node_id(g_orig) -> node_id(g_evo)
    """
    if node_id_prev in node_map:
        node_id_new = node_map[node_id_prev]
    else:
        node_id_new = len(node_map)
        node_map[node_id_prev] = node_id_new

    return node_id_new


def nodes_reindex(node_map, nodes_id_prev):
    nodes_id_new = []
    for node_id in nodes_id_prev:
        nodes_id_new.append(node_reindex(node_map, node_id))
    return nodes_id_new


def update_g_attr(new_nodes, g_evo, g_orig, node_map_orig2evo, node_map_evo2orig):
    """
    Update graph attribution (feature and train/eval/test mask)
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
    idx_train = random.sample(loc_list, math.floor(labels.size()[0] * train_ratio))
    print('Train_set size: ', len(idx_train))
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
    idx_test = random.sample(loc_list, math.floor(labels.size()[0] * test_ratio))

    # Add new nodes and its neighbors in test set
    new_nodes_evo = nodes_reindex(node_map_orig2evo, new_nodes)
    idx_test.extend(new_nodes_evo)
    idx_test.extend(get_ngh(g_evo.adj_sparse('csr'), new_nodes_evo))
    idx_test = list(set(idx_test))

    idx_test.sort()
    test_mask = generate_mask_tensor(_sample_mask(idx_test, labels.shape[0]))
    g_evo.ndata['test_mask'] = test_mask


def update_g_attr_delta(new_nodes, g_evo, g_orig, node_map_evo2orig, node_map_orig2evo):
    """
    Update feature and eval/test mask
    Set train_mask only with the new inserted vertices
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

    # loc_list = range(labels.size()[0])
    # idx_train = random.sample(loc_list,
    #                           math.floor(labels.size()[0] * train_ratio))
    # idx_train.sort()
    # train_mask = generate_mask_tensor(_sample_mask(idx_train, labels.shape[0]))
    # g_evo.ndata['train_mask'] = train_mask
    """
    Training mask are set to new inserted vertices 
    """
    # nodes_index_evo = []
    # for node in new_nodes:
    #     nodes_index_evo.append(node_map_orig2evo[node])
    new_nodes_evo = nodes_reindex(node_map_orig2evo, new_nodes)
    nodes_index_evo = new_nodes_evo
    # Restrict neighbor size less than training ratio
    ngh_limit = math.floor(labels.size()[0] * train_ratio)
    if len(nodes_index_evo) > ngh_limit:
        nodes_index_evo = random.sample(nodes_index_evo, ngh_limit)

    # # Add inserted nodes into training set
    # nodes_index_evo.extend(new_nodes_evo)
    # nodes_index_evo = list(set(nodes_index_evo))

    print('Train_set size: ', len(nodes_index_evo))
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
    idx_test = random.sample(loc_list, math.floor(labels.size()[0] * test_ratio))

    # Add new nodes and its neighbors in test set
    idx_test.extend(new_nodes_evo)
    idx_test.extend(get_ngh(g_evo.adj_sparse('csr'), new_nodes_evo))
    idx_test = list(set(idx_test))

    idx_test.sort()
    test_mask = generate_mask_tensor(_sample_mask(idx_test, labels.shape[0]))
    g_evo.ndata['test_mask'] = test_mask


def update_g_attr_all_ngh(new_nodes, g_evo, g_orig, node_map_evo2orig, node_map_orig2evo):
    """
    Update feature and eval/test mask
    Set train_mask with the new inserted vertices and all of its neighbors
    """
    # Get orig_node_index of evo_node_index
    nodes_orig_index = []
    for node_id in g_evo.nodes().tolist():
        nodes_orig_index.append(node_map_evo2orig[node_id])

    features = g_orig.ndata['feat'][nodes_orig_index, :]
    g_evo.ndata['feat'] = features

    labels = g_orig.ndata['label'][nodes_orig_index]
    g_evo.ndata['label'] = labels

    train_ratio = 0.06
    val_ratio = 0.15
    test_ratio = 0.1

    # loc_list = range(labels.size()[0])
    # idx_train = random.sample(loc_list,
    #                           math.floor(labels.size()[0] * train_ratio))
    # idx_train.sort()
    # train_mask = generate_mask_tensor(_sample_mask(idx_train, labels.shape[0]))
    # g_evo.ndata['train_mask'] = train_mask
    """
    Training mask are set to new inserted vertices and its neighbors
    """
    nodes_index_evo = []
    new_nodes_evo = nodes_reindex(node_map_orig2evo, new_nodes)
    nodes_index_evo.extend(new_nodes_evo)
    nodes_index_evo.extend(get_ngh(g_evo.adj_sparse('csr'), new_nodes_evo))

    # nodes_index_evo.extend(ngh_queue)
    nodes_index_evo = list(set(nodes_index_evo))
    # Restrict neighbor size less than training ratio
    ngh_limit = math.floor(labels.size()[0] * train_ratio)
    if len(nodes_index_evo) > ngh_limit:
        nodes_index_evo = random.sample(nodes_index_evo, ngh_limit)

    print('Train_set size: ', len(nodes_index_evo))
    nodes_index_evo.sort()
    idx_train = nodes_index_evo
    train_mask = generate_mask_tensor(_sample_mask(idx_train, labels.shape[0]))
    g_evo.ndata['train_mask'] = train_mask

    loc_list = range(labels.size()[0])
    idx_val = random.sample(loc_list, math.floor(labels.size()[0] * val_ratio))
    idx_val.sort()
    val_mask = generate_mask_tensor(_sample_mask(idx_val, labels.shape[0]))
    g_evo.ndata['val_mask'] = val_mask

    # loc_list = range(labels.size()[0])
    # idx_test = random.sample(loc_list, math.floor(labels.size()[0] * test_ratio))
    idx_test = []
    # Add new nodes and its neighbors in test set
    idx_test.extend(new_nodes_evo)
    idx_test.extend(get_ngh(g_evo.adj_sparse('csr'), new_nodes_evo))
    idx_test = list(set(idx_test))
    if not idx_test:
        loc_list = range(labels.size()[0])
        idx_test = random.sample(loc_list, math.floor(labels.size()[0] * test_ratio))

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
    # node_queue = nodes
    node_queue = nodes
    for layer_id in range(layer_num):
        node_queue_seen = set(node_queue)
        node_num = len(node_queue)
        for i in range(node_num):
            # print(i, node_num, len(node_queue))
            node = node_queue[i]
            begin = indptr[node]
            end = indptr[node + 1]
            for edge in range(begin, end):
                ngh_node = indices[edge]
                if ngh_node in node_map_orig2evo:
                    if ngh_node not in node_queue_seen:
                        node_queue.append(ngh_node)
                        node_queue_seen.add(ngh_node)
                        node_set.add(ngh_node)
                        edge_set.add(edge)

        # Pop visited node
        node_queue = node_queue[node_num:]

    node_sum = len(node_set)
    edge_sum = len(edge_set)

    return node_sum, edge_sum


def count_neighbor_delta(nodes, g_csr, node_map_orig2evo, layer_num, deg_th=0):
    """ 
    Count accesses of the new nodes and edges in GNN-delta under degree threshold
    """
    node_access_num = 0
    edge_access_num = 0
    indptr = g_csr[0].numpy().tolist()
    indices = g_csr[1].numpy().tolist()
    node_queue = nodes
    # ngh_queue = []
    ngh_queue = set()

    for node in node_queue:
        begin = indptr[node]
        end = indptr[node + 1]
        for edge in range(begin, end):
            node_ngh = indices[edge]
            if node_ngh in node_map_orig2evo:
                begin_ngh = indptr[node_ngh]
                end_ngh = indptr[node_ngh + 1]
                # Count all ngh access for high degree nodes
                if end_ngh - begin_ngh >= deg_th:
                    # ngh_queue.append(node_ngh)
                    ngh_queue.add(node_ngh)
                # Only count delta access for low degree nodes
                else:
                    node_access_num += 1
                    edge_access_num += 1

    node_ngh_access_num, edge_ngh_access_num = count_neighbor(list(ngh_queue), g_csr, node_map_orig2evo,
                                                              layer_num)

    node_access_num += node_ngh_access_num
    edge_access_num += edge_ngh_access_num

    return node_access_num, edge_access_num


def save_graph_csr(g, dataset):
    g_csr = g.adj_sparse('csr')
    indptr = g_csr[0].numpy().tolist()
    indices = g_csr[1].numpy().tolist()

    np.savetxt('./dataset/csr/' + dataset + '_indptr.txt', indptr, fmt='%d')
    np.savetxt('./dataset/csr/' + dataset + '_indices.txt', indices, fmt='%d')
