import torch as th
import math
import random
import dgl
from dgl.data.utils import generate_mask_tensor
from dgl.data.citation_graph import _sample_mask
import util
import time
import pathlib


def gen_node_reindex(node_map, node_id_prev):
    """
    node_id(g_orig) -> node_id(g_evo)
    """
    if node_id_prev in node_map:
        node_id_new = node_map[node_id_prev]
    else:
        node_id_new = len(node_map)
        node_map[node_id_prev] = node_id_new

    return node_id_new


def gen_nodes_reindex(node_map, nodes_id_prev):
    nodes_id_new = []
    for node_id in nodes_id_prev:
        nodes_id_new.append(gen_node_reindex(node_map, node_id))
    return nodes_id_new


def get_node_reindex(node_map, node_id_prev):
    """
    Find the node reindex; If not exist, return -1
    """
    if node_id_prev in node_map:
        node_id_new = node_map[node_id_prev]
    else:
        node_id_new = -1

    return node_id_new


def get_nodes_reindex(node_map, nodes_id_prev):
    nodes_id_new = []
    for node_id in nodes_id_prev:
        node_reindex = get_node_reindex(node_map, node_id)
        if node_reindex != -1:  # If node exist
            nodes_id_new.append(node_reindex)
    return nodes_id_new


def update_g_struct_init(args, init_ratio, init_nodes, g_orig, node_map_orig2evo,
                         node_map_evo2orig):
    """
    Load the init g_struct_update (src_nodes & dst_nodes) from files to save time
    """

    print('>> Start to initialize graph struct')
    time_start = time.perf_counter()

    # Read edge_nodes
    file_edge_nodes = pathlib.Path('./dataset/edge_src_dst_nodes/' + args.dataset +
                                   '_g_struct_init_' + str(init_ratio) + '.txt')
    file_map_orig2evo = pathlib.Path('./dataset/edge_src_dst_nodes/' + args.dataset +
                                     '_map_orig2evo_' + str(init_ratio) + '.txt')
    file_map_evo2orig = pathlib.Path('./dataset/edge_src_dst_nodes/' + args.dataset +
                                     '_map_evo2orig_' + str(init_ratio) + '.txt')

    if file_edge_nodes.exists() and file_map_orig2evo.exists() and file_map_evo2orig.exists():
        # Read edge nodes
        edge_src_nodes = []
        edge_dst_nodes = []
        f = open(file_edge_nodes, "r")
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')  # Delete '\n'
            tmp = line.split(' ')
            edge_src_nodes.append(int(tmp[0]))
            edge_dst_nodes.append(int(tmp[1]))

        # Read node_map_orig2evo
        dict_orig2evo_tmp = {}
        f = open(file_map_orig2evo, "r")
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')  # Delete '\n'
            tmp = line.split(' ')
            k = int(tmp[0])
            v = int(tmp[1])
            dict_orig2evo_tmp[k] = v
        node_map_orig2evo.update(dict_orig2evo_tmp)

        # Read node_map_evo2orig
        dict_evo2orig_tmp = {}
        f = open(file_map_evo2orig, "r")
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')  # Delete '\n'
            tmp = line.split(' ')
            k = int(tmp[0])
            v = int(tmp[1])
            dict_evo2orig_tmp[k] = v
        node_map_evo2orig.update(dict_evo2orig_tmp)

    else:
        edge_src_nodes, edge_dst_nodes = gen_edge_src_edge_nodes(init_nodes, g_orig,
                                                                 node_map_orig2evo,
                                                                 node_map_evo2orig)
        # Write edge
        with open(file_edge_nodes, 'w') as f:
            for i in range(len(edge_src_nodes)):
                edge_src_n = edge_src_nodes[i]
                edge_dst_n = edge_dst_nodes[i]
                f.write(str(edge_src_n) + ' ' + str(edge_dst_n) + '\n')

        # Write node_map_orig2evo
        with open(file_map_orig2evo, 'w') as f:
            for k, v in node_map_orig2evo.items():
                f.write(str(k) + ' ' + str(v) + '\n')

        # Write node_map_evo2orig
        with open(file_map_evo2orig, 'w') as f:
            for k, v in node_map_evo2orig.items():
                f.write(str(k) + ' ' + str(v) + '\n')

    edge_src_nodes = th.tensor(edge_src_nodes, dtype=th.int64)
    edge_dst_nodes = th.tensor(edge_dst_nodes, dtype=th.int64)
    g_evo = dgl.graph((edge_src_nodes, edge_dst_nodes))

    # Remove parallel edges
    device = g_evo.device
    g_evo = dgl.to_simple(g_evo.cpu(), return_counts='cnt', copy_ndata=True, copy_edata=True)
    g_evo = g_evo.to(device)

    print('>> Finish initialize graph structure ({:.2f}s)'.format(time.perf_counter() - time_start))

    return g_evo


def gen_edge_src_edge_nodes(nodes, g_orig, node_map_orig2evo, node_map_evo2orig):
    edge_src_nodes = []
    edge_dst_nodes = []
    for node in nodes:
        pred_nghs = g_orig.predecessors(node).cpu().numpy().tolist()
        for pred_v in pred_nghs:
            if pred_v in nodes or pred_v in node_map_orig2evo:
                edge_src_nodes.append(pred_v)
                edge_dst_nodes.append(node)

        succ_nghs = g_orig.successors(node).cpu().numpy().tolist()
        for succ_v in succ_nghs:
            if succ_v in nodes or succ_v in node_map_orig2evo:
                edge_src_nodes.append(node)
                edge_dst_nodes.append(succ_v)

    # Remapping node_id from g_orig -> g_evo, and record mapping from g_evo -> g_orig
    edge_src_nodes_reindex = []
    edge_dst_nodes_reindex = []
    for node in edge_src_nodes:
        node_id_evo = gen_node_reindex(node_map_orig2evo, node)
        edge_src_nodes_reindex.append(node_id_evo)
        if node_id_evo not in node_map_evo2orig:
            node_map_evo2orig[node_id_evo] = node

    for node in edge_dst_nodes:
        node_id_evo = gen_node_reindex(node_map_orig2evo, node)
        edge_dst_nodes_reindex.append(node_id_evo)
        if node_id_evo not in node_map_evo2orig:
            node_map_evo2orig[node_id_evo] = node

    return edge_src_nodes_reindex, edge_dst_nodes_reindex


def update_g_struct_evo(new_nodes, g_orig, node_map_orig2evo, node_map_evo2orig, g_evo):
    print('>> Start update graph structure')
    time_start = time.perf_counter()

    edge_src_nodes, edge_dst_nodes = gen_edge_src_edge_nodes(new_nodes, g_orig, node_map_orig2evo,
                                                             node_map_evo2orig)

    edge_src_nodes = th.tensor(edge_src_nodes, dtype=th.int64)
    edge_dst_nodes = th.tensor(edge_dst_nodes, dtype=th.int64)

    # if g_evo is None:
    #     # Construct a new graph
    #     g_evo = dgl.graph((edge_src_nodes, edge_dst_nodes))
    # else:
    #     # device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    #     edge_src_nodes = edge_src_nodes.to(g_evo.device)
    #     edge_dst_nodes = edge_dst_nodes.to(g_evo.device)
    #     g_evo.add_edges(edge_src_nodes, edge_dst_nodes)

    edge_src_nodes = edge_src_nodes.to(g_evo.device)
    edge_dst_nodes = edge_dst_nodes.to(g_evo.device)
    g_evo.add_edges(edge_src_nodes, edge_dst_nodes)

    # Remove parallel edges
    device = g_evo.device
    g_evo = dgl.to_simple(g_evo.cpu(), return_counts='cnt', copy_ndata=True, copy_edata=True)
    g_evo = g_evo.to(device)

    print('>> Finish update graph structure ({:.2f}s)'.format(time.perf_counter() - time_start))

    return g_evo


# def update_g_struct(new_nodes, g_orig, node_map_orig2evo, node_map_evo2orig, g_evo=None):
#     print('>> Start update graph structure')
#     time_start = time.perf_counter()

#     edge_src_nodes = []
#     edge_dst_nodes = []
#     for node in new_nodes:
#         pred_nghs = g_orig.predecessors(node).cpu().numpy().tolist()
#         for pred_v in pred_nghs:
#             if pred_v in new_nodes or pred_v in node_map_orig2evo:
#                 edge_src_nodes.append(pred_v)
#                 edge_dst_nodes.append(node)

#         succ_nghs = g_orig.successors(node).cpu().numpy().tolist()
#         for succ_v in succ_nghs:
#             if succ_v in new_nodes or succ_v in node_map_orig2evo:
#                 edge_src_nodes.append(node)
#                 edge_dst_nodes.append(succ_v)

#     # Remapping node_id from g_orig -> g_evo, and record mapping from g_evo -> g_orig
#     edge_src_nodes_reindex = []
#     edge_dst_nodes_reindex = []
#     for node in edge_src_nodes:
#         node_id_evo = gen_node_reindex(node_map_orig2evo, node)
#         edge_src_nodes_reindex.append(node_id_evo)
#         if node_id_evo not in node_map_evo2orig:
#             node_map_evo2orig[node_id_evo] = node

#     for node in edge_dst_nodes:
#         node_id_evo = gen_node_reindex(node_map_orig2evo, node)
#         edge_dst_nodes_reindex.append(node_id_evo)
#         if node_id_evo not in node_map_evo2orig:
#             node_map_evo2orig[node_id_evo] = node

#     edge_src_nodes_reindex = th.tensor(edge_src_nodes_reindex, dtype=th.int64)
#     edge_dst_nodes_reindex = th.tensor(edge_dst_nodes_reindex, dtype=th.int64)

#     if g_evo is None:
#         # Construct a new graph
#         g_evo = dgl.graph((edge_src_nodes_reindex, edge_dst_nodes_reindex))
#     else:
#         # device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
#         edge_src_nodes_reindex = edge_src_nodes_reindex.to(g_evo.device)
#         edge_dst_nodes_reindex = edge_dst_nodes_reindex.to(g_evo.device)
#         g_evo.add_edges(edge_src_nodes_reindex, edge_dst_nodes_reindex)

#     # Remove parallel edges
#     device = g_evo.device
#     g_evo = dgl.to_simple(g_evo.cpu(), return_counts='cnt', copy_ndata=True, copy_edata=True)
#     g_evo = g_evo.to(device)

#     print('>> Finish update graph structure ({:.2f}s)'.format(time.perf_counter() - time_start))

#     return g_evo, edge_src_nodes_reindex, edge_dst_nodes_reindex


def graph_struct_init(args, init_ratio, new_nodes, g_orig, node_map_orig2evo, node_map_evo2orig):
    """
    Initialize graph
    """
    g_evo = update_g_struct_init(args, init_ratio, new_nodes, g_orig, node_map_orig2evo,
                                 node_map_evo2orig)
    update_g_attr_init(g_evo, g_orig, node_map_evo2orig)

    return g_evo


def graph_evolve(new_nodes,
                 g_orig_csr,
                 g_orig,
                 node_map_orig2evo,
                 node_map_evo2orig,
                 layer_num,
                 g_evo=None):
    """
    Construct evolve graph from an orginal static graph
    """
    if g_evo is None:
        # Construct a new graph
        g_evo = update_g_struct_evo(new_nodes, g_orig, node_map_orig2evo, node_map_evo2orig)
        new_graph = True
    else:
        g_evo = update_g_struct_evo(new_nodes, g_orig, node_map_orig2evo, node_map_evo2orig, g_evo)
        new_graph = False

    if new_graph:
        update_g_attr_init(g_evo, g_orig, node_map_evo2orig)
    else:
        update_g_attr_evo(new_nodes, g_evo, g_orig, node_map_orig2evo, node_map_evo2orig, layer_num)

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
        g_evo = update_g_struct_evo(new_nodes, g_orig, node_map_orig2evo, node_map_evo2orig)
        new_graph = True
    else:
        g_evo = update_g_struct_evo(new_nodes, g_orig, node_map_orig2evo, node_map_evo2orig, g_evo)
        new_graph = False

    if new_graph:
        update_g_attr_init(g_evo, g_orig, node_map_evo2orig)
    else:
        update_g_attr_delta(new_nodes, g_evo, g_orig, node_map_orig2evo, node_map_evo2orig)

    return g_evo


def graph_evolve_delta_all_ngh(new_nodes,
                               g_orig_csr,
                               g_orig,
                               node_map_orig2evo,
                               node_map_evo2orig,
                               layer_num,
                               g_evo=None):
    """
    Construct evolve graph from an orginal static graph
    """

    if g_evo is None:
        # Construct a new graph
        g_evo = update_g_struct_evo(new_nodes, g_orig, node_map_orig2evo, node_map_evo2orig)
        new_graph = True
    else:
        g_evo = update_g_struct_evo(new_nodes, g_orig, node_map_orig2evo, node_map_evo2orig, g_evo)
        new_graph = False

    if new_graph:
        update_g_attr_init(g_evo, g_orig, node_map_evo2orig)
    else:
        update_g_attr_evo(new_nodes, g_evo, g_orig, node_map_orig2evo, node_map_evo2orig, layer_num)

    return g_evo


def update_g_attr_init(g_evo, g_orig, node_map_evo2orig):
    """
    Update graph attribution (feature and train/eval/test mask)
    """
    # Get orig_node_index of evo_node_index
    nodes_orig_index = []
    for node in g_evo.nodes().tolist():
        nodes_orig_index.append(node_map_evo2orig[node])

    features = g_orig.ndata['feat'][nodes_orig_index, :].to(g_evo.device)
    g_evo.ndata['feat'] = features

    labels = g_orig.ndata['label'][nodes_orig_index].to(g_evo.device)
    g_evo.ndata['label'] = labels

    train_ratio = 0.06
    val_ratio = 0.15
    test_ratio = 0.3

    g_node_num = g_evo.number_of_nodes()
    train_num = round(g_node_num * train_ratio)
    val_num = round(g_node_num * val_ratio)
    test_num = round(g_node_num * test_ratio)

    # idx_train = range(math.floor(labels.size()[0] * train_ratio))
    idx_train = range(train_num)
    train_mask = generate_mask_tensor(_sample_mask(idx_train, labels.shape[0])).to(g_evo.device)
    g_evo.ndata['train_mask'] = train_mask

    idx_val = range(train_num, train_num + val_num)
    val_mask = generate_mask_tensor(_sample_mask(idx_val, labels.shape[0])).to(g_evo.device)
    g_evo.ndata['val_mask'] = val_mask

    idx_test = range(g_node_num - test_num, g_node_num)
    test_mask = generate_mask_tensor(_sample_mask(idx_test, labels.shape[0])).to(g_evo.device)
    g_evo.ndata['test_mask'] = test_mask


def update_g_attr_evo(new_nodes, g_evo, g_orig, node_map_orig2evo, node_map_evo2orig, layer_num):
    """
    Update feature and eval/test mask
    Set train_mask with the new inserted vertices and all of its n-layer neighbors
    """
    # Get orig_node_index of evo_node_index
    nodes_orig_index = []
    for node_id in g_evo.nodes().tolist():
        nodes_orig_index.append(node_map_evo2orig[node_id])

    features = g_orig.ndata['feat'][nodes_orig_index, :].to(g_evo.device)
    g_evo.ndata['feat'] = features

    labels = g_orig.ndata['label'][nodes_orig_index].to(g_evo.device)
    g_evo.ndata['label'] = labels

    train_ratio = 0.06  # 0.06
    val_ratio = 0.15
    test_ratio = 0.3
    """
    Training mask are set to new inserted vertices and its neighbors, and supplement with previous vertices
    """
    nodes_index_evo = []
    new_nodes_evo = get_nodes_reindex(node_map_orig2evo, new_nodes)
    nodes_index_evo.extend(new_nodes_evo)
    new_nodes_evo_nghs = util.get_dst_nghs_multi_layers(g_evo, new_nodes_evo, layer_num)
    nodes_index_evo.extend(new_nodes_evo_nghs)
    random.shuffle(nodes_index_evo)

    node_num = len(nodes_index_evo)
    train_num = round(node_num * train_ratio)
    idx_train_inserted = nodes_index_evo[:train_num]
    # idx_train_inserted = new_nodes_evo_nghs[:train_num]

    # Supplement train set
    loc_list = range(labels.size()[0])
    loc_list = [i for i in loc_list if i not in nodes_index_evo]
    idx_train_orig = loc_list
    # idx_train_orig = (random.sample(loc_list, math.floor(labels.size()[0] * train_ratio)))
    # idx_train_orig = loc_list[:round(labels.size()[0] * train_ratio)]

    # 1:1 mixed train set
    train_num_mixed = min(len(idx_train_inserted), len(idx_train_orig))
    idx_train_inserted = random.sample(idx_train_inserted, round(train_num_mixed))
    idx_train_orig = random.sample(idx_train_orig, round(train_num_mixed))

    idx_train = idx_train_inserted + idx_train_orig
    idx_train = list(set(idx_train))

    print('Train_set size: ', len(nodes_index_evo))
    idx_train.sort()
    train_mask = generate_mask_tensor(_sample_mask(idx_train, labels.shape[0])).to(g_evo.device)
    g_evo.ndata['train_mask'] = train_mask

    loc_list = range(labels.size()[0])
    idx_val = random.sample(loc_list, math.floor(labels.size()[0] * val_ratio))
    idx_val.sort()
    val_mask = generate_mask_tensor(_sample_mask(idx_val, labels.shape[0])).to(g_evo.device)
    g_evo.ndata['val_mask'] = val_mask

    # val_num = round(node_num * val_ratio)
    # idx_val = nodes_index_evo[train_num:val_num]
    # idx_val.sort()
    # val_mask = generate_mask_tensor(_sample_mask(idx_val, labels.shape[0])).to(g_evo.device)
    # g_evo.ndata['val_mask'] = val_mask

    node_num = g_evo.number_of_nodes()
    test_num = round(node_num * test_ratio)
    # idx_test_inserted = nodes_index_evo[-test_num:-1]
    # random.shuffle(new_nodes_evo_nghs)
    idx_test_inserted = random.sample(new_nodes_evo_nghs,
                                      round(len(new_nodes_evo_nghs) * test_ratio))

    loc_list = range(labels.size()[0])
    loc_list = [i for i in loc_list if i not in nodes_index_evo]
    idx_test_orig = loc_list
    # idx_test_orig = (random.sample(loc_list, math.floor(labels.size()[0] * test_ratio)))
    # idx_test_orig = loc_list[round(labels.size()[0] * test_ratio):-1]

    # 1:1 mixed train set
    test_num_mixed = min(len(idx_test_inserted), len(idx_test_orig))
    idx_test_inserted = random.sample(idx_test_inserted, test_num_mixed)
    idx_test_orig = random.sample(idx_test_orig, round(test_num_mixed))

    # idx_test = idx_test_orig
    idx_test = idx_test_inserted + idx_test_orig
    idx_test = list(set(idx_test))

    # # Add new nodes and its neighbors in test set
    # idx_test = []
    # idx_test.extend(new_nodes_evo)
    # idx_test.extend(get_nghs(g_evo.adj_sparse('csr'), new_nodes_evo))
    # idx_test = list(set(idx_test))

    # if not idx_test:
    #     loc_list = range(labels.size()[0])
    #     idx_test = random.sample(loc_list, math.floor(labels.size()[0] * test_ratio))

    idx_test.sort()
    test_mask = generate_mask_tensor(_sample_mask(idx_test, labels.shape[0])).to(g_evo.device)
    # test_mask = train_mask
    g_evo.ndata['test_mask'] = test_mask


def update_g_attr_delta(new_nodes, g_evo, g_orig, node_map_orig2evo, node_map_evo2orig):
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
    new_nodes_evo = get_nodes_reindex(node_map_orig2evo, new_nodes)
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
    train_mask = generate_mask_tensor(_sample_mask(idx_train, labels.shape[0])).to(g_evo.device)
    g_evo.ndata['train_mask'] = train_mask

    loc_list = range(labels.size()[0])
    idx_val = random.sample(loc_list, math.floor(labels.size()[0] * val_ratio))
    idx_val.sort()
    val_mask = generate_mask_tensor(_sample_mask(idx_val, labels.shape[0])).to(g_evo.device)
    g_evo.ndata['val_mask'] = val_mask

    loc_list = range(labels.size()[0])
    idx_test = random.sample(loc_list, math.floor(labels.size()[0] * test_ratio))

    # Add new nodes and its neighbors in test set
    idx_test.extend(new_nodes_evo)
    idx_test.extend(util.get_nghs(g_evo.adj_sparse('csr'), new_nodes_evo))
    idx_test = list(set(idx_test))

    idx_test.sort()
    test_mask = generate_mask_tensor(_sample_mask(idx_test, labels.shape[0])).to(g_evo.device)
    g_evo.ndata['test_mask'] = test_mask