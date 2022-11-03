from collections import Counter
from numpy import int64
# from msilib import sequence
import torch
import torch as th
import dgl
from dgl.data.utils import generate_mask_tensor
from dgl.data.citation_graph import _sample_mask
import math
import random
import numpy as np
import copy as cp
import time


def gen_root_node_queue(g):
    """
    Gen root node according to in-degree
    """
    src_edges = g.edges()[0].cpu().numpy().tolist()
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
            begin = indptr[vetex]
            end = indptr[vetex + 1]
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


def get_nghs(g_csr, root_nodes):
    indptr = g_csr[0].cpu().numpy().tolist()
    indices = g_csr[1].cpu().numpy().tolist()
    ngh = []
    for root_node in root_nodes:
        # if root_node < indptr.shape[0] - 1:
        begin = indptr[root_node]
        end = indptr[root_node + 1]
        ngh.extend(indices[begin:end])

    return ngh


def get_nghs_multi_layers(g_csr, root_nodes, layer_num):
    indptr = g_csr[0].cpu().numpy().tolist()
    indices = g_csr[1].cpu().numpy().tolist()
    nodes_q = cp.deepcopy(root_nodes)
    ngh = []
    for i in range(layer_num):
        ngh_nodes = get_nghs(g_csr, nodes_q)
        ngh.extend(ngh_nodes)
        nodes_q = ngh_nodes

    ngh = list(set(ngh))

    return ngh


def get_dst_nghs_multi_layers(g, root_nodes, layer_num):
    nodes_q = list(cp.deepcopy(root_nodes))
    nghs_total = []
    ngh_per_layer = []
    for i in range(layer_num):
        for node in nodes_q:
            nghs = g.successors(node).cpu().numpy().tolist()
            ngh_per_layer.extend(nghs)
            nghs_total.extend(nghs)
        # print('>> ngh_per_layer_size: ', len(ngh_per_layer))
        # print('>> ngh_size: ', len(nghs_total))
        nodes_q.clear()
        nodes_q.extend(ngh_per_layer)
        ngh_per_layer.clear()

    # nghs_total = list(set(nghs_total))

    return nghs_total


def get_dst_nghs_multi_layers_with_mapping(g, root_nodes, layer_num):
    """
    Return the nghs and a mapping dict (Dict format, k: ngh_v, v: root_v)
    """
    nodes_q = list(cp.deepcopy(root_nodes))
    nghs_total = []
    ngh_per_layer = []
    dict_v = dict()
    for i in range(layer_num):
        for node in nodes_q:
            nghs = g.successors(node).cpu().numpy().tolist()
            ngh_per_layer.extend(nghs)
            for ngh in nghs:
                nghs_total.append(ngh)
                if ngh not in dict_v:
                    dict_v[ngh] = node
        nodes_q.clear()
        nodes_q.extend(ngh_per_layer)
        ngh_per_layer.clear()

    return nghs_total, dict_v


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
            ngh_high_deg.setdefault(root_node, set()).update(
                ngh_nodes
            )  # Regard inserted nodes as high-degree nodes to execute all-ngh retraining
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

    # # Add previous vertices in train set into ngh_high_deg
    # train_mask = g.ndata['train_mask']
    # train_nodes_id = train_mask.nonzero().squeeze()
    # train_nodes_id = train_nodes_id.cpu().numpy().tolist()
    # prev_nodes_id = train_nodes_id - ngh_low_deg.keys() - ngh_high_deg.keys()
    # for node in prev_nodes_id:
    #     ngh_high_deg.setdefault(node, set()).update(get_ngh(g_csr, [node]))

    return ngh_high_deg, ngh_low_deg


# def gen_nodes_with_deg_th(g, nodes, inserted_nodes, deg_th):
#     '''
#     Select out nodes for delta-updating (generate edge mask)
#     Principle:
#         1) If the nodes is newly inserted nodes, regard to high-degree nodes
#         2) Else, determined by the node degree

#     nodes: nodes which need to be classified according to deg_th
#     inserted_nodes: nodes which are newly inserted in current graph snapshot
#     deg_th: node degree threshold
#     '''

#     g_csr = g.adj_sparse('csr')
#     nodes_high_deg = dict()
#     nodes_low_deg = dict()
#     nodes_high_deg_t = [i for i in nodes if i in inserted_nodes]
#     # Insert inserted nodes as high degree nodes
#     for node in nodes_high_deg_t:
#         nodes_high_deg.setdefault(node, set()).update(get_ngh(g_csr, [node]))

#     nodes_t = [i for i in nodes if i not in nodes_high_deg_t]
#     for node in nodes_t:
#         deg = count_node_degree(g_csr, node)
#         if deg >= deg_th:
#             nodes_high_deg.setdefault(node, set()).update(get_ngh(g_csr, [node]))
#         else:
#             nodes_low_deg.setdefault(node, set()).add(node)

#     return nodes_high_deg, nodes_low_deg

# def gen_nodes_with_deg_th(g, nodes, inserted_nodes, deg_th):
#     '''
#     Select out nodes for delta-updating (generate edge mask)
#     Principle:
#         1) If the nodes is newly inserted nodes, regard to high-degree nodes
#         2) Else, determined by the node degree

#     nodes: nodes which need to be classified according to deg_th
#     inserted_nodes: nodes which are newly inserted in current graph snapshot
#     deg_th: node degree threshold
#     '''

#     g_csr = g.adj_sparse('csr')
#     nodes_high_deg = []
#     nodes_low_deg = []
#     nodes_high_deg_t = [i for i in nodes if i in inserted_nodes]
#     # Regard inserted nodes to high degree nodes
#     nodes_high_deg.extend(nodes_high_deg_t)

#     nodes_t = [i for i in nodes if i not in inserted_nodes]
#     for node in nodes_t:
#         deg = count_node_degree(g_csr, node)
#         if deg >= deg_th:
#             nodes_high_deg.append(node)
#         else:
#             nodes_low_deg.append(node)

#     return nodes_high_deg, nodes_low_deg

# def gen_nodes_with_deg_th(g, nodes, inserted_nodes, deg_th):
#     '''
#     Select out nodes for delta-updating (generate edge mask)
#     Principle:
#         1) If the nodes is newly inserted nodes, regard to high-degree nodes
#         2) Else, determined by the node degree

#     nodes: nodes which need to be classified according to deg_th
#     inserted_nodes: nodes which are newly inserted in current graph snapshot
#     deg_th: node degree threshold
#     '''

#     g_csr = g.adj_sparse('csr')
#     nodes_high_deg = dict()
#     nodes_low_deg = dict()
#     nodes_high_deg_t = [i for i in nodes if i in inserted_nodes]
#     # Insert inserted nodes as high degree nodes
#     for node in nodes_high_deg_t:
#         nodes_high_deg.setdefault(node, set()).update(get_nghs(g_csr, [node]))

#     nodes_t = [i for i in nodes if i not in nodes_high_deg_t]
#     for node in nodes_t:
#         deg = count_node_degree(g_csr, node)
#         if deg >= deg_th:
#             nodes_high_deg.setdefault(node, set()).update(get_nghs(g_csr, [node]))
#         else:
#             nodes_low_deg.setdefault(node, set()).update(node)

#     return nodes_high_deg, nodes_low_deg


def gen_nodes_with_deg_th(g, nodes, inserted_nodes, deg_th):
    '''
    Select out nodes for delta-updating (generate edge mask)
    Principle:
        1) If the nodes is newly inserted nodes, regard to high-degree nodes
        2) Else, determined by the node degree

    nodes: nodes which need to be classified according to deg_th
    inserted_nodes: nodes which are newly inserted in current graph snapshot    
    deg_th: node degree threshold
    '''

    g_csr = g.adj_sparse('csr')
    nodes_high_deg = dict()
    nodes_low_deg = dict()
    nodes_high_deg_t = [i for i in nodes if i in inserted_nodes]
    # Insert inserted nodes as high degree nodes
    for node in nodes_high_deg_t:
        nodes_high_deg.setdefault(node, set()).update(get_nghs(g_csr, [node]))

    nodes_t = [i for i in nodes if i not in nodes_high_deg_t]
    for node in nodes_t:
        deg = count_node_degree(g_csr, node)
        if deg >= deg_th:
            nodes_high_deg.setdefault(node, set()).update(get_nghs(g_csr, [node]))
        else:
            nodes_low_deg.setdefault(node, set()).update(node)

    return nodes_high_deg, nodes_low_deg


def gen_train_nodes_with_deg_th(g, inserted_nodes, deg_th):
    '''
    Select out train nodes for delta-updating (generate edge mask)
    Principle:
        1) If the train_nodes is newly inserted nodes, regard to high-degree nodes
        2) Else, determined by the node degree

    deg_th: node degree threshold
    '''

    train_nodes = th.nonzero(g.ndata['train_mask'])
    train_nodes = train_nodes.squeeze().cpu().numpy().tolist()
    train_nodes_high_deg, train_nodes_low_deg = gen_nodes_with_deg_th(g, train_nodes,
                                                                      inserted_nodes, deg_th)

    return train_nodes_high_deg, train_nodes_low_deg


def gen_test_nodes_with_deg_th(g, inserted_nodes, deg_th):
    '''
    Select out test nodes for delta-updating (generate edge mask)
    Principle:
        1) If the test_nodes is newly inserted nodes, regard to high-degree nodes
        2) Else, determined by the node degree

    deg_th: node degree threshold
    '''

    test_nodes = th.nonzero(g.ndata['test_mask'])
    test_nodes = test_nodes.squeeze().cpu().numpy().tolist()
    test_nodes_high_deg, test_nodes_low_deg = gen_nodes_with_deg_th(g, test_nodes, inserted_nodes,
                                                                    deg_th)

    return test_nodes_high_deg, test_nodes_low_deg


# def gen_edge_mask(g, ngh_dict):
#     src_nodes = []
#     dst_nodes = []
#     for root_node, ngh in ngh_dict.items():
#         src_nodes.extend([root_node for i in range(len(ngh))])
#         dst_nodes.extend(ngh)

#     # a = th.tensor(src_nodes, dtype=th.long)
#     # b = th.tensor(dst_nodes, dtype=th.long)
#     # print(a, b)
#     # print(g.edges())
#     # k = g.edge_ids(th.tensor([0, 0]), th.tensor([1, 3]))

#     # edge_ids = g.edge_ids(th.tensor(src_nodes, dtype=th.long), th.tensor(dst_nodes, dtype=th.long))
#     src_nodes = th.tensor(src_nodes, dtype=th.long).to(g.device)
#     dst_nodes = th.tensor(dst_nodes, dtype=th.long).to(g.device)
#     edge_ids = g.edge_ids(src_nodes, dst_nodes)

#     edge_ids = edge_ids.tolist()
#     edge_mask = [0 for i in range(g.number_of_edges())]
#     for id in edge_ids:
#         edge_mask[id] = 1

#     g.edata['edge_mask'] = th.Tensor(edge_mask).to(g.device)

# def gen_edge_mask(g, ngh_dict):
#     src_nodes = []
#     dst_nodes = []
#     for root_node, ngh in ngh_dict.items():
#         src_nodes.extend([root_node for i in range(len(ngh))])
#         dst_nodes.extend(ngh)

#     # a = th.tensor(src_nodes, dtype=th.long)
#     # b = th.tensor(dst_nodes, dtype=th.long)
#     # print(a, b)
#     # print(g.edges())
#     # k = g.edge_ids(th.tensor([0, 0]), th.tensor([1, 3]))

#     # edge_ids = g.edge_ids(th.tensor(src_nodes, dtype=th.long), th.tensor(dst_nodes, dtype=th.long))
#     src_nodes = th.tensor(src_nodes, dtype=th.long).to(g.device)
#     dst_nodes = th.tensor(dst_nodes, dtype=th.long).to(g.device)
#     edge_ids = g.edge_ids(src_nodes, dst_nodes)

#     edge_ids = edge_ids.tolist()
#     edge_mask = [0 for i in range(g.number_of_edges())]
#     for id in edge_ids:
#         edge_mask[id] = 1

#     return th.Tensor(edge_mask).to(g.device)

# def gen_edge_mask(g, inserted_nodes, deg_th, layer_num):
# # g_csr = g.adj_sparse('csr')
# src_nodes = []
# dst_nodes = []
# node_total = list(nodes_high_deg.keys()) + list(nodes_low_deg.keys())

# # For layer = 1
# for node in nodes_high_deg.keys():
#     src_node = g.predecessors(node).cpu().numpy().tolist()
#     src_nodes.extend(src_node)
#     dst_nodes.extend([node for i in range(len(src_node))])

# # for root_node, nghs in nodes_high_deg.items():
# #     nghs = list(nghs)
# #     src_nodes.extend(nghs)
# #     dst_nodes.extend([root_node for i in range(len(nghs))])

# #     ## Debug_yin
# #     for v in nghs:
# #         src_v = g.predecessors(v).cpu().numpy().tolist()
# #         src_nodes.extend(src_v)
# #         dst_nodes.extend([v for i in range(len(src_v))])

#     # for i in range(len(nghs)):
#     #     if g.has_edges_between(nghs[i], root_node):
#     #         src_nodes.append(nghs[i])
#     #         dst_nodes.append(root_node)
#     # src_nodes.extend(nghs)
#     # dst_nodes.extend([root_node for i in range(len(nghs))])

# for root_node, nghs in nodes_low_deg.items():
#     nghs = list(nghs)
#     src_nodes.extend(nghs)
#     dst_nodes.extend([root_node for i in range(len(nghs))])

#     # ## Debug_yin
#     # for v in nghs:
#     #     src_v = g.predecessors(v).cpu().numpy().tolist()
#     #     src_nodes.extend([v for i in range(len(src_v))])
#     #     dst_nodes.extend(src_v)

#     # for i in range(len(nghs)):
#     #     if g.has_edges_between(nghs[i], root_node):
#     #         src_nodes.append(nghs[i])
#     #         dst_nodes.append(root_node)
#     # src_nodes.extend(nghs)
#     # dst_nodes.extend([root_node for i in range(len(nghs))])

# # For layer >= 2 (Only for high_degree nodes)
# ngh_q = list(nodes_high_deg.keys())
# for i in range(layer_num -1):
#     for i in range(len(ngh_q)):
#         node = ngh_q.pop()
#         # nghs = get_nghs(g_scr, [node])
#         nghs = g.predecessors(node).cpu().numpy().tolist()
#         # Only process on unseen nodes
#         for ngh in nghs:
#             if ngh not in node_total:
#                 src_nodes.append(ngh)
#                 dst_nodes.append(node)
#                 ngh_q.append(ngh)

#                 # ## Debug_yin
#                 # if g.has_edges_between(ngh, node):
#                 #     src_nodes.append(ngh)
#                 #     dst_nodes.append(node)
#     # src_nodes.extend(nghs)
#     # dst_nodes.extend([root_node for i in range(len(nghs))])

# src_nodes = th.tensor(src_nodes, dtype=th.long).to(g.device)
# dst_nodes = th.tensor(dst_nodes, dtype=th.long).to(g.device)
# edge_ids = g.edge_ids(src_nodes, dst_nodes)

# edge_ids = edge_ids.tolist()
# edge_mask = [0 for i in range(g.number_of_edges())]
# for id in edge_ids:
#     edge_mask[id] = 1

# return th.Tensor(edge_mask).to(g.device)

# def gen_edge_mask(g, inserted_nodes, deg_th, layer_num):
#     """
#     Only shield the low degree nodes
#     """
#     nodes_low_deg = []
#     src_nodes = []
#     dst_nodes = []
#     nodes_q = cp.deepcopy(inserted_nodes)
#     ngh_per_layer = []
#     for i in range(layer_num):
#         for node in nodes_q:
#             nghs = g.successors(node).cpu().numpy().tolist()
#             ngh_per_layer.extend(nghs)
#             for ngh in nghs:
#                 deg = g.successors(ngh).shape[0]
#                 if deg < deg_th[i]:
#                     src_nodes.append(node)
#                     dst_nodes.append(ngh)
#                     nodes_low_deg.append(ngh)
#         nodes_q = ngh_per_layer
#         ngh_per_layer = []

#     src_nodes = th.tensor(src_nodes, dtype=th.long).to(g.device)
#     dst_nodes = th.tensor(dst_nodes, dtype=th.long).to(g.device)
#     edge_ids = g.edge_ids(src_nodes, dst_nodes)

#     edge_ids = edge_ids.tolist()

#     edge_low_deg_num = len(list(set(edge_ids)))
#     edge_num = g.number_of_edges()
#     print(edge_num, edge_low_deg_num, 'percentage:', round(edge_low_deg_num / edge_num, 2))

#     edge_mask = [1 for i in range(g.number_of_edges())]
#     for id in edge_ids:
#         edge_mask[id] = 0

#     return nodes_low_deg, th.Tensor(edge_mask).to(g.device)


def get_predecessor_nghs(g, root_nodes, layer_num):
    """
    Traverse multi-layer nghs, and return the [src_nodes] -> [dst_nodes]. 
    Note that predecessor nodes of the root_nodes are defined as src_nodes
    """
    src_nodes = []
    dst_nodes = []
    ngh_per_layer = []
    node_q = cp.deepcopy(root_nodes)
    for i in range(layer_num):
        for node in node_q:
            nghs = g.predecessors(node).cpu().numpy().tolist()
            ngh_per_layer.extend(nghs)
            src_nodes.extend(nghs)
            dst_nodes.extend([node for i in range(len(nghs))])
        node_q = cp.deepcopy(ngh_per_layer)
        ngh_per_layer.clear()

    return src_nodes, dst_nodes


# def gen_edge_mask(g, inserted_nodes, deg_th, layer_num):
#     """
#     Edge_mask: select out processing nodes with high (all-ngh) and low (delta-update) degree
#     """
#     nodes_low_deg = []
#     nodes_high_deg = []
#     src_nodes = []
#     dst_nodes = []

#     # Regard inserted nodes as high-deg nodes
#     nodes_high_deg.extend(inserted_nodes)
#     for node in inserted_nodes:
#         pred_nghs = g.predecessors(node).cpu().numpy().tolist()
#         src_nodes.extend(pred_nghs)
#         dst_nodes.extend([node for i in range(len(pred_nghs))])

#     nodes_q = cp.deepcopy(inserted_nodes)
#     ngh_per_layer = []
#     # Traverse L-hops nghs of inserted nodes
#     for i in range(layer_num):
#         for node in nodes_q:
#             nghs = g.successors(node).cpu().numpy().tolist()
#             ngh_per_layer.extend(nghs)
#             for ngh in nghs:
#                 deg = g.out_degrees(ngh)
#                 # For high deg nodes
#                 if deg >= deg_th[i]:
#                     nodes_high_deg.append(ngh)
#                     # Traverse L-hop predecessor nghs of high deg nodes
#                     src_nodes_tmp, dst_nodes_tmp = get_predecessor_nghs(g, [ngh], layer_num)
#                     src_nodes.extend(src_nodes_tmp)
#                     dst_nodes.extend(dst_nodes_tmp)
#                 # For low deg nodes
#                 else:
#                     nodes_low_deg.append(ngh)
#                     src_nodes.append(node)
#                     dst_nodes.append(ngh)

#         nodes_q = cp.deepcopy(ngh_per_layer)
#         ngh_per_layer.clear()

#     src_nodes = th.tensor(src_nodes, dtype=th.long).to(g.device)
#     dst_nodes = th.tensor(dst_nodes, dtype=th.long).to(g.device)
#     edge_ids = g.edge_ids(src_nodes, dst_nodes)

#     edge_ids = edge_ids.tolist()
#     edge_masked_num = len(list(set(edge_ids)))
#     edge_num_total = g.number_of_edges()
#     print(edge_num_total, edge_masked_num, 'percentage:', round(edge_masked_num / edge_num_total,
#                                                                 2))

#     edge_mask = [0 for i in range(g.number_of_edges())]
#     for id in edge_ids:
#         edge_mask[id] = 1

#     return th.Tensor(edge_mask).to(g.device), nodes_high_deg, nodes_low_deg


def gen_edge_mask(g, inserted_nodes, deg_th, layer_num):
    """
    Edge_mask: select out processing nodes with high (all-ngh) and low (delta-update) degree 
    """
    print('>> Start to gen edge mask')
    time_start = time.perf_counter()

    device = g.device
    g = g.cpu()

    # g = dgl.remove_self_loop(g)
    nodes_low_deg = []
    nodes_high_deg = []
    # Record edges which need be set to zero
    src_nodes = []
    dst_nodes = []
    # Reserve shield edges for delta updating
    src_nodes_shield = []
    dst_nodes_shield = []

    # Regard inserted nodes as high-deg nodes
    nodes_high_deg.extend(inserted_nodes)
    # for node in inserted_nodes:
    #     pred_nghs = g.predecessors(node).cpu().numpy().tolist()
    # src_nodes.extend(pred_nghs)
    # dst_nodes.extend([node for i in range(len(pred_nghs))])

    nodes_q = cp.deepcopy(inserted_nodes)
    ngh_per_layer = []
    # Traverse L-hops nghs of inserted nodes
    for i in range(layer_num):
        for node in nodes_q:
            nghs = g.successors(node).cpu().numpy().tolist()
            ngh_per_layer.extend(nghs)
            for ngh in nghs:
                deg = g.out_degrees(ngh)
                # For high deg nodes
                if deg >= deg_th[i]:
                    nodes_high_deg.append(ngh)
                    # # Traverse L-hop predecessor nghs of high deg nodes
                    # src_nodes_tmp, dst_nodes_tmp = get_predecessor_nghs(g, [ngh], layer_num)
                    # src_nodes.extend(src_nodes_tmp)
                    # dst_nodes.extend(dst_nodes_tmp)
                # For low deg nodes
                else:
                    nodes_low_deg.append(ngh)
                    pred_nghs = g.predecessors(ngh).cpu().numpy().tolist()
                    src_nodes.extend(pred_nghs)
                    dst_nodes.extend(ngh for i in range(len(pred_nghs)))
                    src_nodes_shield.append(node)
                    dst_nodes_shield.append(ngh)

        nodes_q = cp.deepcopy(ngh_per_layer)
        ngh_per_layer.clear()

    src_nodes = th.tensor(src_nodes, dtype=th.long)
    dst_nodes = th.tensor(dst_nodes, dtype=th.long)
    edge_ids = g.edge_ids(src_nodes, dst_nodes)

    src_nodes_shield = th.tensor(src_nodes_shield, dtype=th.long)
    dst_nodes_shield = th.tensor(dst_nodes_shield, dtype=th.long)
    edge_ids_shield = g.edge_ids(src_nodes_shield, dst_nodes_shield)

    edge_ids = edge_ids.tolist()
    edge_ids_shield = edge_ids_shield.tolist()
    # print(list(set(edge_ids)))
    # print(list(set(edge_ids_shield)))
    edge_masked_num = len(list(set(edge_ids))) - len(list(set(edge_ids_shield)))
    edge_num_total = g.number_of_edges()
    print('Edge_total: {:d}, Edge_mask_num: {:d}, ratio: {:.2%}'.format(
        edge_num_total, edge_masked_num, edge_masked_num / edge_num_total))

    edge_mask = [1 for i in range(g.number_of_edges())]
    for id in edge_ids:
        edge_mask[id] = 0
    for id in edge_ids_shield:
        edge_mask[id] = 1

    nodes_high_deg = list(set(nodes_high_deg))
    nodes_low_deg = list(set(nodes_low_deg))
    print()
    print('N_high_deg: {:d}, N_low_deg: {:d}, low_deg_ratio: {:.2%}'.format(
        len(nodes_high_deg), len(nodes_low_deg),
        len(nodes_low_deg) / len(nodes_high_deg)))

    g = g.to(device)

    print('>> Finish gen edge mask ({})'.format(time_format(time.perf_counter() - time_start)))

    return th.Tensor(edge_mask).to(g.device), nodes_high_deg, nodes_low_deg


def count_neighbor(nodes, g_csr, node_map_orig2evo, layer_num, mem_access_q=None):
    """
    Count neighbor edges and vertices of specific node set

    node_access_q: used for generating mem trace
    """
    # edge_set = set()
    # node_set = set(nodes)
    node_access_num = 0
    edge_access_num = 0
    indptr = g_csr[0].cpu().numpy().tolist()
    indices = g_csr[1].cpu().numpy().tolist()
    # node_queue = nodes
    # node_queue_seen = set()
    node_queue = cp.copy(nodes)
    mem_access_q.extend(nodes)
    for layer_id in range(layer_num):
        node_num = len(node_queue)
        for i in range(node_num):
            node = node_queue[i]
            begin = indptr[node]
            end = indptr[node + 1]
            node_access_num += end - begin
            edge_access_num += end - begin
            for edge in range(begin, end):
                ngh_node = indices[edge]
                # if ngh_node not in node_queue_seen:
                node_queue.append(ngh_node)
                mem_access_q.append(ngh_node)

        # Pop visited node
        node_queue = node_queue[node_num:]

    # node_sum = len(node_set)
    # edge_sum = len(edge_set)

    # return node_sum, edge_sum
    return node_access_num, edge_access_num


def count_neighbor_delta(nodes, g_csr, node_map_orig2evo, layer_num, deg_th=0, mem_access_q=None):
    """ 
    Count accesses of the new nodes and edges in GNN-delta under degree threshold
    """
    node_access_num = 0
    edge_access_num = 0
    indptr = g_csr[0].cpu().numpy().tolist()
    indices = g_csr[1].cpu().numpy().tolist()
    print(">> indptr_len: ", len(indptr))
    node_queue = cp.copy(nodes)
    # ngh_queue = []
    ngh_queue = set()
    # node_queue_seen=set()
    mem_access_q.extend(nodes)

    for node in node_queue:
        begin = indptr[node]
        end = indptr[node + 1]
        for edge in range(begin, end):
            node_ngh = indices[edge]
            # if node_ngh in node_map_orig2evo:
            begin_ngh = indptr[node_ngh]
            end_ngh = indptr[node_ngh + 1]
            # Count all ngh access for high degree nodes
            # if node_ngh not in node_queue_seen:
            if end_ngh - begin_ngh >= deg_th:
                # ngh_queue.append(node_ngh)
                ngh_queue.add(node_ngh)
            # Only count delta access for low degree nodes
            else:
                node_access_num += 2  # For accessing this node's new feature and its neighbor's previous feature
                edge_access_num += 2  # For accessing this node's new feature and its neighbor's previous feature
                mem_access_q.append(node_ngh)
                # node_queue_seen.add(node_ngh)

    node_ngh_access_num, edge_ngh_access_num = count_neighbor(list(ngh_queue), g_csr,
                                                              node_map_orig2evo, layer_num,
                                                              mem_access_q)
    print(len(nodes), len(list(ngh_queue)))
    print('>> delta ngh', node_access_num, edge_access_num, node_ngh_access_num,
          edge_ngh_access_num)

    node_access_num += node_ngh_access_num
    edge_access_num += edge_ngh_access_num

    return node_access_num, edge_access_num


# def count_neighbor_full(nodes, g_csr, layer_num):
#     """
#     Count neighbor edges and vertices of specific node set
#     """
#     edge_set = set()
#     node_set = set(nodes)
#     indptr = g_csr[0].numpy().tolist()
#     indices = g_csr[1].numpy().tolist()
#     # node_queue = nodes
#     node_queue = nodes
#     for layer_id in range(layer_num):
#         node_queue_seen = set(node_queue)
#         node_num = len(node_queue)
#         for i in range(node_num):
#             # print(i, node_num, len(node_queue))
#             node = node_queue[i]
#             begin = indptr[node]
#             end = indptr[node + 1]
#             for edge in range(begin, end):
#                 ngh_node = indices[edge]
#                 if ngh_node not in node_queue_seen:
#                     node_queue.append(ngh_node)
#                     node_queue_seen.add(ngh_node)
#                     node_set.add(ngh_node)
#                     edge_set.add(edge)

#         # Pop visited node
#         node_queue = node_queue[node_num:]

#     node_sum = len(node_set)
#     edge_sum = len(edge_set)

#     return node_sum, edge_sum


def save_graph_csr(g, dataset):
    g_csr = g.adj_sparse('csr')
    indptr = g_csr[0].numpy().tolist()
    indices = g_csr[1].numpy().tolist()

    np.savetxt('./dataset/csr/' + dataset + '_indptr.txt', indptr, fmt='%d')
    np.savetxt('./dataset/csr/' + dataset + '_indices.txt', indices, fmt='%d')


def dump_mem_trace(queue, file_path):
    with open(file_path, 'a') as f:
        for item in queue:
            f.write(str(item))
            f.write('\n')


def gen_trace_sorted_by_node_deg(g_csr, node_q):
    """
    Gen node traverse trace sorted by node degree (low degree -> high degree)
    """
    indptr = g_csr[0].numpy().tolist()
    indices = g_csr[1].numpy().tolist()
    deg_nodes = dict()
    degs = set()
    node_trace = list()
    for node in node_q:
        ngh_begin = indptr[node]
        ngh_end = indptr[node + 1]
        deg = ngh_end - ngh_begin
        degs.add(deg)
        deg_nodes.setdefault(deg, list()).extend(indices[ngh_begin:ngh_end])

    deg_list = list(degs)
    deg_list.sort()
    for deg in deg_list:
        """
        Node trace format [deg, node_id, node_id, ....]
        """
        li = [deg]
        li.extend(deg_nodes[deg])
        node_trace.append(li)

    return node_trace


def get_index_of_minvalue(list):
    min_val = min(list)
    min_index = list.index(min_val)
    return min_index


def gen_trace_without_sorted(g_csr, node_q):
    # """
    # Gen node traverse trace sorted by node degree (low degree -> high degree)
    # """
    # indptr = g_csr[0].numpy().tolist()
    # indices = g_csr[1].numpy().tolist()
    # deg_nodes = []
    # for node in node_q:
    #     ngh_begin = indptr[node]
    #     ngh_end = indptr[node + 1]
    #     deg = ngh_end - ngh_begin
    #     deg_nodes.append([deg] + indices[ngh_begin:ngh_end])

    # return deg_nodes

    ##
    """
    Emulate interleave node access among different degs (model cache thrash)
    """
    indptr = g_csr[0].numpy().tolist()
    indices = g_csr[1].numpy().tolist()
    deg_nodes = []
    for node in node_q:
        ngh_begin = indptr[node]
        ngh_end = indptr[node + 1]
        deg = ngh_end - ngh_begin
        deg_nodes.append([deg] + indices[ngh_begin:ngh_end])

    parallel = 8
    nodes_size_bin = [0 for i in range(parallel)]
    nodes_bin = [[] for i in range(parallel)]
    deg_bin = [[] for i in range(parallel)]

    for row in deg_nodes:
        index = get_index_of_minvalue(nodes_size_bin)  # Select the bin which has the min workload
        nodes_size_bin[index] += len(row) - 1  # Update workload size
        nodes_bin[index].extend(row[1:])  # Push workload
        deg_bin[index].extend([row[0]
                               for i in range(len(row) - 1)])  # Record degree of this workload

    node_trace = []
    min_workload_size = min(nodes_size_bin)
    for j in range(min_workload_size):
        for bin_id in range(len(nodes_bin)):
            node_trace.append([deg_bin[bin_id][j], nodes_bin[bin_id][j]])

    return node_trace


def rm_repeat_data(node_trace):
    for row_id in range(len(node_trace)):
        row = node_trace[row_id]
        # for row in node_trace:
        tmp_row = [row[0], row[1]]  # row[0]: deg, row[1]: first element
        if len(row) > 2:
            for i in range(len(row) - 2):
                if row[i + 2] != row[i + 1]:
                    tmp_row.append(row[i + 2])
        # row = tmp_row
        node_trace[row_id] = tmp_row


def eval_node_locality(node_trace):
    """
    Evaluate node access locality. The miss rate is emulated under a direct-mapping cache.
    """
    cacheline_size = 64  # Byte
    cacheline_bias = int(math.log2(cacheline_size))
    data_size = 4  # Byte
    hit_miss_degs = dict()
    hit_rate_degs = []
    degs = set()
    cacheline_id_prev = 0

    for row in node_trace:
        deg = row[0]
        degs.add(deg)
        item = {'hit': 0, 'miss': 0}
        if deg not in hit_miss_degs:
            hit_miss_degs[deg] = item
        for i in range(len(row) - 1):
            node_id = row[i + 1]
            node_addr = node_id * data_size
            cacheline_id = node_addr >> cacheline_bias
            if cacheline_id != cacheline_id_prev:
                hit_miss_degs[deg]['miss'] += 1
                cacheline_id_prev = cacheline_id
            else:
                hit_miss_degs[deg]['hit'] += 1

    degs = list(degs)
    degs.sort()

    for deg in degs:
        hit = hit_miss_degs[deg]['hit']
        miss = hit_miss_degs[deg]['miss']
        hit_rate = round(hit / (hit + miss) * 100, 2)
        hit_rate_degs.append(hit_rate)

    return hit_rate_degs


def save_txt_2d(path, data):
    with open(path, 'w') as f:
        for i in data:
            for j in i:
                f.write(str(j))
                f.write(' ')
            f.write('\n')


def gen_degree_distribution(g_csr):
    """
    Gen node distribution with degree
    """
    indptr = g_csr[0].numpy().tolist()

    deg_list = []
    max_deg = 0
    for i in range(len(indptr) - 1):
        begin = indptr[i]
        end = indptr[i + 1]
        deg = end - begin
        deg_list.append(deg)
        max_deg = deg if deg > max_deg else max_deg

    deg_distribution = [0 for i in range(max_deg + 1)]
    for i in deg_list:
        deg_distribution[i] += 1

    return deg_distribution


def sort_node_by_timestamp(file_path):
    import csv

    timestamp = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for r in reader:
            timestamp.append(int(r[0]))

    time_min = min(timestamp)
    time_max = max(timestamp)

    timestamp_bin = [[] for i in range(time_min, time_max + 1)]
    for i in range(len(timestamp)):
        index = timestamp[i] - time_min
        timestamp_bin[index].append(i)

    # Static the number of accumulated nodes by year
    total = 0
    accumulation = []

    node_q_sort_by_time = []
    for i in timestamp_bin:
        node_q_sort_by_time.extend(i)
        total += len(i)
        accumulation.append(total)

    np.savetxt('./results/graph_struct_evo/graph_structure_evo.txt', accumulation, fmt='%d')

    return node_q_sort_by_time


def count_node_degree(g_csr, node):
    indptr = g_csr[0].cpu().numpy().tolist()
    degree = indptr[node + 1] - indptr[node]
    return degree


def gen_snapshot(init_ratio, snapshot_num, total_node_num):
    node_num = total_node_num
    interval = snapshot_num
    scale_ratio = pow((1 / init_ratio), 1 / interval)
    node_seq = [round(node_num * init_ratio * scale_ratio**i) for i in range(interval)]
    node_seq[-1] = node_num

    return node_seq


def time_format(sec):
    if sec > 3600:
        hour, tmp = divmod(sec, 3600)
        min, s = divmod(tmp, 60)
        time = str(int(hour)) + 'h' + str(int(min)) + 'm' + str(int(s)) + 's'
    elif sec > 60:
        min, s = divmod(sec, 60)
        time = str(int(min)) + 'm' + str(int(s)) + 's'
    else:
        s = round(sec, 2)
        time = str(s) + 's'

    return time
