import torch
import gc
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
# from propagation import InstantGNN
import pdb
import struct
from torch_sparse import SparseTensor
import copy as cp
from collections import Counter
import time
import math


def load_aminer_init(datastr, rmax, alpha):
    if datastr == "1984_author_dense":
        m = 3787605
        n = 1252095
    elif datastr == "2013_author_dense":
        m = 9237799
        n = 1252095

    print("Load %s!" % datastr)
    labels = np.load("./data/aminer/" + datastr + "_labels.npy")

    # py_alg = InstantGNN()

    features = np.load('./data/aminer/aminer_dense_feat.npy')
    split = np.load('./data/aminer/aminer_dense_idx_split.npz')
    train_idx, val_idx, test_idx = split['train'], split['valid'], split['test']

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    train_labels = torch.LongTensor(labels[train_idx])
    val_labels = torch.LongTensor(labels[val_idx])
    test_labels = torch.LongTensor(labels[test_idx])

    train_labels = train_labels.reshape(train_labels.size(0), 1)
    val_labels = val_labels.reshape(val_labels.size(0), 1)
    test_labels = test_labels.reshape(test_labels.size(0), 1)

    return features, train_labels, val_labels, test_labels, train_idx, val_idx, test_idx


def load_dataset_init(datastr):
    print("Load %s!" % datastr)

    # py_alg = InstantGNN()
    features = np.load('./data/' + datastr + '/' + datastr + '_feat.npy')

    data = np.load('./data/' + datastr + '/' + datastr + '_labels.npz')
    train_idx = torch.LongTensor(data['train_idx'])
    val_idx = torch.LongTensor(data['val_idx'])
    test_idx = torch.LongTensor(data['test_idx'])
    labels = torch.LongTensor(data['labels'])
    n_classes = data['n_classes']

    # src_edges = read_packed_edges('./data/' + datastr + '/' + datastr + '_init_src_edge.txt', 'I')
    # dst_edges = read_packed_edges('./data/' + datastr + '/' + datastr + '_init_dst_edge.txt', 'I')

    src_edges, dst_edges = read_packed_edges(
        './data/' + datastr + '/' + datastr + '_init_edges.txt', 'I')

    src_edges = torch.LongTensor(src_edges)
    dst_edges = torch.LongTensor(dst_edges)

    edge_index = torch.stack(
        [torch.cat([src_edges, dst_edges], dim=0),
         torch.cat([dst_edges, src_edges], dim=0)], dim=0)

    return features, labels, train_idx, val_idx, test_idx, n_classes, edge_index


def load_updated_edges(datastr, snapshot_id):
    edges = np.loadtxt('./data/' + datastr + '/' + datastr + '_Edgeupdate_snap' + str(snapshot_id) +
                       '.txt')
    edges = torch.LongTensor(edges)
    # edge_index = edges.reshape(2, edges.shape[0], dim=0)
    edge_src = edges[:, 0]
    edge_dst = edges[:, 1]
    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    return edge_index


def gen_edge_dict(edge_index):
    """
    Key, val = edge_src, edge_dst
    """
    edge_dict = dict()
    print(edge_index.shape[-1])
    print(range(edge_index.shape[-1]))
    for i in range(edge_index.shape[-1]):
        edge_src = edge_index[0][i].item()
        edge_dst = edge_index[1][i].item()
        edge_dict.setdefault(edge_src, {edge_dst}).add(edge_dst)

    return edge_dict


def insert_edge_dict(edge_dict, edge_index_inserted):
    for i in range(edge_index_inserted.shape[-1]):
        edge_src = edge_index_inserted[0][i].item()
        edge_dst = edge_index_inserted[1][i].item()
        edge_dict.setdefault(edge_src, {edge_dst}).add(edge_dst)
    return edge_dict


# def insert_edges_delta(edge_dict, edge_index_inserted, threshold, layer_num):
#     """
#     edge_dict: key,val = edge_src, edge_dst
#     edge_index_inserted: new inserted edges, format edge_src, edge_dst
#     edge_index_delta: to represent delta-msg propagating
#     """
#     v_sen = set()
#     v_insen = set()
#     edge_index_delta = []
#     edge_src_insen = []
#     edge_dst_insen = []
#     edge_src_sen = []
#     edge_dst_sen = []
#     for i in range(edge_index_inserted.shape[-1]):
#         edge_src = edge_index_inserted[0][i].item()
#         edge_dst = edge_index_inserted[1][i].item()
#         if edge_dst in edge_dict:
#             deg_v = len(edge_dict[edge_dst])
#             if deg_v > threshold:
#                 v_sen.add(edge_dst)
#             else:
#                 v_insen.add(edge_dst)
#                 edge_src_insen.append(edge_src)
#                 edge_dst_insen.append(edge_dst)

#     # Update edge_dict
#     edge_dict = insert_edge_dict(edge_dict, edge_index_inserted)
#     # Gen edge_index_delta
#     nodes_q = cp.deepcopy(v_sen)
#     ngh_per_layer = set()
#     # Traverse L-hops nghs of inserted nodes
#     for i in range(layer_num):
#         for node in nodes_q:
#             nghs = edge_dict[node]  # For graph is undirected, we can obtain src_v by dst_v
#             ngh_per_layer.update(nghs)
#             src_v = list(nghs)
#             dst_v = [node for i in range(len(nghs))]
#             edge_src_sen.extend(src_v)
#             edge_dst_sen.extend(dst_v)
#         nodes_q = cp.deepcopy(ngh_per_layer)
#         ngh_per_layer.clear()

#     edge_index_sen = torch.LongTensor([edge_src_sen, edge_dst_sen])
#     edge_index_insen = torch.LongTensor([edge_src_insen, edge_dst_insen])
#     edge_index_delta = insert_edges(edge_index_sen, edge_index_insen)

#     return edge_dict, edge_index_delta, v_sen, v_insen


def insert_edges(edge_index_orig, edge_index_inserted):
    """
    orig_edge_index: format (src, dst)
    inserted _edge_index: format (src, dst)
    """
    return torch.cat([edge_index_orig, edge_index_inserted], dim=-1)


def insert_edges_evo(edge_index, edge_dict, edge_index_inserted, threshold, layer_num):

    edge_dict, edge_index, *tmp = insert_edges_delta(edge_index, edge_dict, edge_index_inserted,
                                                     threshold, layer_num)
    return edge_dict, edge_index


def insert_edges_delta(edge_index_evo_delta, edge_dict, edge_index_inserted, threshold, layer_num):
    """
    edge_dict: key,val = edge_src, edge_dst
    edge_index_inserted: new inserted edges, format edge_src, edge_dst
    edge_index_delta: to represent delta-msg propagating
    """
    v_sen = set()
    v_insen = set()
    v_total = set()
    edge_index_delta = []
    edge_src_insen = []
    edge_dst_insen = []
    edge_src_sen = []
    edge_dst_sen = []

    # Aggregation reduction
    e_num_total = 0

    # Combination reduction
    v_deg_total = 0
    v_deg_delta = 0

    # Access reduction
    access_total = set()
    access_delta = set()

    # Computation reduction
    comp_total = 0
    comp_delta = 0
    has_visited = set()  # Record vertices has been counted

    for i in range(edge_index_inserted.shape[-1]):
        edge_src = edge_index_inserted[0][i].item()
        edge_dst = edge_index_inserted[1][i].item()
        if edge_dst in edge_dict:
            deg_v = len(edge_dict[edge_dst])
            # v_deg_total += deg_v
            v_total.add(edge_dst)
            if deg_v > threshold:
                v_sen.add(edge_dst)
                v_deg_delta += deg_v
                if edge_dst not in has_visited:
                    # comp_delta += deg_v
                    # access_delta.update(edge_dict[edge_dst])
                    has_visited.add(edge_dst)
            else:
                v_insen.add(edge_dst)
                edge_src_insen.append(edge_src)
                edge_dst_insen.append(edge_dst)
                v_deg_delta += 1
                if edge_dst not in has_visited:
                    comp_delta += 1
                    access_delta.add(edge_dst)
                    has_visited.add(edge_dst)

    # Update edge_dict
    edge_dict = insert_edge_dict(edge_dict, edge_index_inserted)
    # Gen edge_index_delta
    nodes_q = cp.deepcopy(v_sen)
    ngh_per_layer = set()
    # Traverse L-hops nghs of sensitive nodes
    for i in range(layer_num):
        for node in nodes_q:
            nghs = edge_dict[node]  # For graph is undirected, we can obtain src_v by dst_v
            for ngh in nghs:
                deg_v = len(edge_dict[ngh])
                if deg_v > threshold:
                    ngh_per_layer.add(ngh)
            # ngh_per_layer.update(nghs)
            src_v = list(nghs)
            dst_v = [node for i in range(len(nghs))]
            edge_src_sen.extend(src_v)
            edge_dst_sen.extend(dst_v)
            comp_delta += len(nghs)
            access_delta.update(nghs)
        nodes_q = cp.deepcopy(ngh_per_layer)
        ngh_per_layer.clear()

    # Gen edge_index_delta
    nodes_q = cp.deepcopy(v_total)
    ngh_per_layer = set()
    # Traverse L-hops nghs of inserted nodes
    for i in range(layer_num):
        for node in nodes_q:
            nghs = edge_dict[node]  # For graph is undirected, we can obtain src_v by dst_v
            ngh_per_layer.update(nghs)
            e_num_total += len(nghs)
            v_deg_total += len(nghs)
            comp_total += len(nghs)
            access_total.update(nghs)
        nodes_q = cp.deepcopy(ngh_per_layer)
        ngh_per_layer.clear()

    edge_index_sen = torch.LongTensor([edge_src_sen, edge_dst_sen])
    edge_index_insen = torch.LongTensor([edge_src_insen, edge_dst_insen])
    edge_index_delta = insert_edges(edge_index_sen, edge_index_insen)
    edge_index_evo_delta = insert_edges(edge_index_evo_delta, edge_index_delta)

    e_num_delta = edge_index_delta.shape[-1]

    # return edge_dict, edge_index_evo_delta, v_sen, v_insen, v_deg_total, v_deg_delta, e_num_total, e_num_delta
    return edge_dict, edge_index_evo_delta, v_sen, v_insen, comp_total, comp_delta, len(
        access_total), len(access_delta)


def load_sbm_init(datastr, rmax, alpha):
    if datastr == "SBM-50000-50-20+1":
        m = 1412466
        n = 50000
    elif datastr == "SBM-500000-50-20+1":
        m = 14141662
        n = 500000
    elif datastr == "SBM-10000000-100-20+1":
        m = 282938572
        n = 10000000
    elif datastr == "SBM-1000000-50-20+1":
        m = 28293138
        n = 1000000

    print("Load %s!" % datastr)

    labels = np.loadtxt('./data/' + datastr + '/' + datastr + '_label.txt')

    # py_alg = InstantGNN()

    if datastr == "SBM-1000000-50-20+1" or datastr == "SBM-500000-50-20+1":
        encode_len = 256
    else:
        encode_len = 1024

    split = np.load('./data/' + datastr + '/' + datastr + '_idx_split.npz')
    train_idx, val_idx, test_idx = split['train'], split['valid'], split['test']
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    features = np.load('./data/' + datastr + '/' + datastr + '_encode_' + str(encode_len) +
                       '_feat.npy')
    # memory_dataset = py_alg.initial_operation('./data/' + datastr + '/adjs/', datastr + '_init', m,
    #                                           n, rmax, alpha, features)

    train_labels = torch.LongTensor(labels[train_idx])
    val_labels = torch.LongTensor(labels[val_idx])
    test_labels = torch.LongTensor(labels[test_idx])

    train_labels = train_labels.reshape(train_labels.size(0), 1)
    val_labels = val_labels.reshape(val_labels.size(0), 1)
    test_labels = test_labels.reshape(test_labels.size(0), 1)

    # return features, train_labels, val_labels, test_labels, train_idx, val_idx, test_idx, memory_dataset, py_alg
    return features, train_labels, val_labels, test_labels, train_idx, val_idx, test_idx


def muticlass_f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    macro = f1_score(labels, preds, average='macro')
    return macro


def com_accuracy(y_pred, y):
    pred = y_pred.data.max(1)[1]
    pred = pred.reshape(pred.size(0), 1)
    correct = pred.eq(y.data).cpu().sum()
    accuracy = correct.to(dtype=torch.long) * 100. / len(y)
    return accuracy


# class SimpleDataset(Dataset):

#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#         assert self.x.size(0) == self.y.size(0)

#     def __len__(self):
#         return self.x.size(0)

#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]


class SimpleDataset(Dataset):

    def __init__(self, x, edge_index, y):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        assert self.x.size(0) == self.y.size(0)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# def read_packed_file(f_path, pack_fmt):
#     length = struct.calcsize(pack_fmt)
#     with open(f_path, 'rb') as f:
#         l = []
#         while True:
#             tmp = f.read(length)
#             if tmp == b'':
#                 break
#             m = struct.unpack(pack_fmt, tmp)[0]
#             # print(m)
#             l.append(m)
#         return l


def read_packed_edges(f_path, pack_fmt):
    length = struct.calcsize(pack_fmt) * 2
    with open(f_path, 'rb') as f:
        edge_src = []
        edge_dst = []
        while True:
            tmp = f.read(length)
            if tmp == b'':
                break
            m = struct.unpack('ii', tmp)
            src = m[0]
            dst = m[1]
            # print(m)
            edge_src.append(src)
            edge_dst.append(dst)
        return edge_src, edge_dst


def msg_count(inserted_edges, edge_dict, n_layers, threshold):
    dst_v = inserted_edges[1]
    dst_v = dst_v.tolist()

    nodes_q = cp.deepcopy(dst_v)
    ngh_per_layer = set()
    # Traverse L-hops nghs of inserted nodes
    for i in range(n_layers - 1):
        for node in nodes_q:
            nghs = edge_dict[node]  # For graph is undirected, we can obtain src_v by dst_v
            dst_v.extend(nghs)
            ngh_per_layer.update(nghs)
        nodes_q = cp.deepcopy(ngh_per_layer)
        ngh_per_layer.clear()

    msg_count = Counter(dst_v)
    return msg_count


# def mag(dataset, num_snap):
#     # dataset = DglNodePropPredDataset(name='ogbn-mag', root='../dataset/')
#     import sklearn.preprocessing
#     from torch_geometric.utils import to_undirected
#     import dgl

#     data, labels = dataset[0]
#     n_classes = np.array(dataset.meta_info['num classes'], dtype=np.int32)
#     split_idx = dataset.get_idx_split()
#     train_idx, val_idx, test_idx = split_idx['train']['paper'], split_idx['valid'][
#         'paper'], split_idx['test']['paper']
#     all_idx = torch.cat([train_idx, val_idx, test_idx])

#     # Feature normalization
#     feat = data.nodes['paper'].data['feat'].numpy()
#     feat = np.array(feat, dtype=np.float64)
#     scaler = sklearn.preprocessing.StandardScaler()
#     scaler.fit(feat)
#     feat = scaler.transform(feat)
#     np.save('./data/mag/mag_feat.npy', feat)

#     #get labels
#     labels = labels['paper']
#     train_labels = labels.data[train_idx]
#     val_labels = labels.data[val_idx]
#     test_labels = labels.data[test_idx]
#     labels = np.array(labels, dtype=np.int32)

#     train_idx = train_idx.numpy()
#     val_idx = val_idx.numpy()
#     test_idx = test_idx.numpy()
#     train_idx = np.array(train_idx, dtype=np.int32)
#     val_idx = np.array(val_idx, dtype=np.int32)
#     test_idx = np.array(test_idx, dtype=np.int32)

#     np.savez('./data/mag/mag_labels.npz',
#              train_idx=train_idx,
#              val_idx=val_idx,
#              test_idx=test_idx,
#              labels=labels,
#              n_classes=n_classes)

#     sub_g = dgl.edge_type_subgraph(data, [('paper', 'cites', 'paper')])
#     h_sub_g = dgl.to_homogeneous(sub_g)
#     edge_index = torch.stack([h_sub_g.edges()[0], h_sub_g.edges()[1]], dim=0)
#     num_nodes = h_sub_g.number_of_nodes()

#     edge_index = to_undirected(edge_index, num_nodes)
#     edge_index, drop_edge_index, _ = dropout_adj(edge_index, train_idx, num_nodes=num_nodes)
#     edge_index = to_undirected(edge_index, num_nodes)

#     row_drop, col_drop = np.array(drop_edge_index)

#     f = open('./data/mag/ogbn-mag_update_full.txt', 'w+')
#     for k in range(row_drop.shape[0]):
#         v_from = row_drop[k]
#         v_to = col_drop[k]
#         f.write('%d %d\n' % (v_from, v_to))
#         f.write('%d %d\n' % (v_to, v_from))
#     f.close()

#     row, col = edge_index
#     print(row_drop.shape)
#     row = row.numpy()
#     col = col.numpy()

#     # # Write edge_idx (src_edge, dst_edge)
#     # write_packed_edges('../data/arxiv/arxiv_init_src_edge.txt', 'I', row)
#     # write_packed_edges('../data/arxiv/arxiv_init_dst_edge.txt', 'I', col)

#     # Write edge_idx (src_edge, dst_edge)
#     write_packed_edges('./data/mag/mag_init_edges.txt', row, col)

#     # save_adj(row, col, N=data.num_nodes, dataset_name='arxiv', savename='arxiv_init', snap='init')
#     # num_snap = 16
#     snapshot = math.floor(row_drop.shape[0] / num_snap)
#     print('num_snap: ', num_snap)

#     for sn in range(num_snap):
#         print(sn)
#         row_sn = row_drop[sn * snapshot:(sn + 1) * snapshot]
#         col_sn = col_drop[sn * snapshot:(sn + 1) * snapshot]
#         if sn == 0:
#             row_tmp = row
#             col_tmp = col

#         row_tmp = np.concatenate((row_tmp, row_sn))
#         col_tmp = np.concatenate((col_tmp, col_sn))
#         row_tmp = np.concatenate((row_tmp, col_sn))
#         col_tmp = np.concatenate((col_tmp, row_sn))
#         # if (sn + 1) % 20 == 0 or (sn + 1) == num_snap:
#         # save_adj(row_tmp,
#         #          col_tmp,
#         #          N=data.num_nodes,
#         #          dataset_name='arxiv',
#         #          savename='arxiv_snap' + str(sn + 1),
#         #          snap=(sn + 1))

#         with open('./data/mag/mag_Edgeupdate_snap' + str(sn + 1) + '.txt', 'w') as f:
#             for i, j in zip(row_sn, col_sn):
#                 f.write("%d %d\n" % (i, j))
#                 f.write("%d %d\n" % (j, i))
#     print('Mag -- save snapshots finish')

# def dropout_adj(edge_index, rmnode_idx, edge_attr=None, force_undirected=True, num_nodes=None):

#     N = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
#     row, col = edge_index

#     if force_undirected:
#         row, col, edge_attr = filter_adj(row, col, edge_attr, row < col)
#     convert_start = time.time()
#     row_convert = row.numpy().tolist()
#     col_convert = col.numpy().tolist()
#     convert_end = time.time()
#     print('convert cost:', convert_end - convert_start)

#     row_mask = np.isin(row, rmnode_idx)
#     col_mask = np.isin(col, rmnode_idx)
#     drop_mask = torch.from_numpy(np.logical_or(row_mask, col_mask)).to(torch.bool)

#     mask = ~drop_mask

#     new_row, new_col, edge_attr = filter_adj(row, col, edge_attr, mask)
#     drop_row, drop_col, edge_attr = filter_adj(row, col, edge_attr, drop_mask)
#     print('init:', len(new_row), ', drop:', len(drop_row))

#     if force_undirected:
#         edge_index = torch.stack(
#             [torch.cat([new_row, new_col], dim=0),
#              torch.cat([new_col, new_row], dim=0)], dim=0)
#         if edge_attr is not None:
#             edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
#         # edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
#     else:
#         edge_index = torch.stack([new_row, new_col], dim=0)
#     drop_edge_index = torch.stack([drop_row, drop_col], dim=0)  ### only u->v (no v->u)

#     return edge_index, drop_edge_index, edge_attr

# def filter_adj(row, col, edge_attr, mask):
#     return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]

# def write_packed_edges(f_path, edge_src, edge_dst):
#     assert len(edge_src) == len(edge_dst)
#     with open(f_path, 'wb') as f:
#         for i in range(len(edge_src)):
#             m = struct.pack("ii", edge_src[i], edge_dst[i])
#             f.write(m)
