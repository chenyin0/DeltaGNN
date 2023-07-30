import argparse
from tqdm import tqdm
import numpy as np
import random
import scipy.sparse as sp
import torch
import torch as th
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from torch_geometric.datasets import Planetoid, FacebookPagePage, WikiCS, Twitch
from torch_geometric.datasets import Reddit
import sklearn.preprocessing
import tracemalloc
import gc
import struct
from torch_sparse import coalesce
import math
import pdb
import time
import pathlib
import util
import dgl

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


def dataset_bfs_sort(dataset, dataset_name):
    # node_q = []
    file_ = pathlib.Path('./dataset/' + dataset_name + '_evo_seq.txt')
    if file_.exists():
        node_q = np.loadtxt(file_, dtype=int).tolist()
    else:
        edge_index = dataset.edge_index
        root_node_q = util.gen_root_node_bfs(edge_index[0])
        adj_csr = util.gen_adj_csr(edge_index, dataset.num_nodes)
        node_q = util.bfs_sort(adj_csr, root_node_q)

        with open(file_, 'w') as f:
            for i in node_q:
                f.write(str(i) + '\n')

    return node_q


def dataset_timestamp_sort(dataset, dataset_name):
    # node_q = []
    file_ = pathlib.Path('./dataset/' + dataset_name + '_evo_seq.txt')
    if file_.exists():
        node_q = np.loadtxt(file_, dtype=int).tolist()
    else:
        node_q = util.sort_node_by_timestamp('./dataset/' + dataset_name + '_node_year.csv')

        with open(file_, 'w') as f:
            for i in node_q:
                f.write(str(i) + '\n')

    return node_q


def dropout_adj(edge_index, rmnode_idx, edge_attr=None, force_undirected=True, num_nodes=None):

    N = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    row, col = edge_index

    if force_undirected:
        row, col, edge_attr = filter_adj(row, col, edge_attr, row < col)
    convert_start = time.time()
    row_convert = row.numpy().tolist()
    col_convert = col.numpy().tolist()
    convert_end = time.time()
    print('convert cost:', convert_end - convert_start)

    row_mask = np.isin(row, rmnode_idx)
    col_mask = np.isin(col, rmnode_idx)
    drop_mask = torch.from_numpy(np.logical_or(row_mask, col_mask)).to(torch.bool)

    mask = ~drop_mask

    new_row, new_col, edge_attr = filter_adj(row, col, edge_attr, mask)
    drop_row, drop_col, edge_attr = filter_adj(row, col, edge_attr, drop_mask)
    print('init:', len(new_row), ', drop:', len(drop_row))

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([new_row, new_col], dim=0),
             torch.cat([new_col, new_row], dim=0)], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        # edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
    else:
        edge_index = torch.stack([new_row, new_col], dim=0)
    drop_edge_index = torch.stack([drop_row, drop_col], dim=0)  ### only u->v (no v->u)

    return edge_index, drop_edge_index, edge_attr


def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


def dataset_split(data, train_ratio, val_ratio, test_ratio):
    v_num = data.num_nodes
    train_num = round(v_num * train_ratio)
    val_num = round(v_num * val_ratio)

    train_mask, val_mask, test_mask = th.zeros(v_num).bool(), th.zeros(v_num).bool(), th.zeros(
        v_num).bool()
    train_mask[0:train_num] = True
    val_mask[train_num:train_num + val_num] = True
    test_mask[train_num + val_num:] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def gen_dataset_snapshot(dataset_name, num_snap):
    if dataset_name == 'Cora' or dataset_name == 'CiteSeer' or dataset_name == 'PubMed':
        planetoid(dataset_name, num_snap)
    elif dataset_name == 'reddit':
        reddit(num_snap)
    elif dataset_name == 'arxiv':
        arxiv(num_snap)
    elif dataset_name == 'mag':
        mag(num_snap)
    elif dataset_name == 'products':
        products(num_snap)
    elif dataset_name == 'arpapers100Mxiv':
        papers100M(num_snap)
    elif dataset_name == 'Facebook':
        facebook(num_snap)
    elif dataset_name == 'WikiCS':
        wikics(num_snap)
    elif dataset_name == 'Twitch':
        twitch(num_snap)


def planetoid(dataset_name, num_snap):
    dataset = Planetoid(name=dataset_name, root='./dataset/')
    data = dataset[0]
    n_classes = np.array(dataset.num_classes, dtype=np.int32)
    train_idx = torch.nonzero(data.train_mask).squeeze()
    val_idx = torch.nonzero(data.val_mask).squeeze()
    test_idx = torch.nonzero(data.test_mask).squeeze()
    # all_idx = torch.cat([train_idx, val_idx, test_idx])

    base_path = './data/' + dataset_name + '/' + dataset_name
    # Feature normalization
    feat = data.x.numpy()
    feat = np.array(feat, dtype=np.float64)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)
    np.save(base_path + '_feat.npy', feat)

    #get labels
    labels = data.y
    labels = torch.unsqueeze(labels, 1)
    # train_labels = labels.data[train_idx]
    # val_labels = labels.data[val_idx]
    # test_labels = labels.data[test_idx]
    labels = np.array(labels, dtype=np.int32)

    # train_idx = train_idx.numpy()
    # val_idx = val_idx.numpy()
    # test_idx = test_idx.numpy()
    train_idx = np.array(train_idx, dtype=np.int32)
    val_idx = np.array(val_idx, dtype=np.int32)
    test_idx = np.array(test_idx, dtype=np.int32)

    np.savez(base_path + '_labels.npz',
             train_idx=train_idx,
             val_idx=val_idx,
             test_idx=test_idx,
             labels=labels,
             n_classes=n_classes)

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    # data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,
    #                                                   train_idx,
    #                                                   num_nodes=data.num_nodes)

    # # test_num = len(test_idx)
    # train_idx = train_idx.tolist()
    # test_idx = test_idx.tolist()
    # val_idx = val_idx.tolist()
    # test_train_ratio = round(len(test_idx) / len(train_idx))

    # train_sample_rate = 0.6  #0.3
    # train_idx = train_idx[:round(len(train_idx) * train_sample_rate)]

    # test_sample_inerval_rate = 0.7
    # test_sample_interval = round(test_sample_inerval_rate * test_train_ratio)
    # print('test_sample_interval: ', test_sample_interval)
    # test_idx = test_idx[::test_sample_interval]

    # drop_idx = train_idx + test_idx + val_idx
    # random.shuffle(drop_idx)
    # drop_idx = np.array(drop_idx, dtype=np.int32)
    # # drop_idx = np.append(train_idx, val_idx)
    # # drop_idx = np.append(drop_idx, test_idx)

    # Load vertex sorted by timestamp
    idx_with_time_seq = dataset_bfs_sort(dataset[0], dataset_name)

    if dataset_name == 'Cora':
        init_ratio = 0.35
    if dataset_name == 'CiteSeer':
        init_ratio = 0.4

    init_num = round(len(idx_with_time_seq) * init_ratio)
    drop_idx = idx_with_time_seq[init_num:]

    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,
                                                      drop_idx,
                                                      num_nodes=data.num_nodes)

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    # r_indexes = np.arange(len(drop_edge_index[0]))
    # np.random.shuffle(r_indexes)
    # drop_edge_index[0] = drop_edge_index[0][r_indexes]
    # drop_edge_index[1] = drop_edge_index[1][r_indexes]

    row_drop, col_drop = np.array(drop_edge_index)

    f = open(base_path + '_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))

    f.close()

    row, col = data.edge_index
    print(row_drop.shape)
    row = row.numpy()
    col = col.numpy()

    # # Write edge_idx (src_edge, dst_edge)
    # write_packed_edges('./data/arxiv/arxiv_init_src_edge.txt', 'I', row)
    # write_packed_edges('./data/arxiv/arxiv_init_dst_edge.txt', 'I', col)

    # Write edge_idx (src_edge, dst_edge)
    write_packed_edges(base_path + '_init_edges.txt', row, col)

    # save_adj(row, col, N=data.num_nodes, dataset_name='arxiv', savename='arxiv_init', snap='init')
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in range(num_snap):
        print(sn)
        row_sn = row_drop[sn * snapshot:(sn + 1) * snapshot]
        col_sn = col_drop[sn * snapshot:(sn + 1) * snapshot]
        if sn == 0:
            row_tmp = row
            col_tmp = col

        row_tmp = np.concatenate((row_tmp, row_sn))
        col_tmp = np.concatenate((col_tmp, col_sn))
        row_tmp = np.concatenate((row_tmp, col_sn))
        col_tmp = np.concatenate((col_tmp, row_sn))
        # if (sn + 1) % 20 == 0 or (sn + 1) == num_snap:
        # save_adj(row_tmp,
        #         col_tmp,
        #         N=data.num_nodes,
        #         dataset_name='arxiv',
        #         savename='arxiv_snap' + str(sn + 1),
        #         snap=(sn + 1))

        with open(base_path + '_Edgeupdate_snap' + str(sn + 1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print(dataset_name + ' -- save snapshots finish')


def facebook(num_snap):
    dataset = FacebookPagePage('./dataset/FacebookPagePage')
    data = dataset[0]
    data = dataset_split(data, 0.2, 0.3, 0.5)
    n_classes = np.array(dataset.num_classes, dtype=np.int32)
    train_idx = torch.nonzero(data.train_mask).squeeze()
    val_idx = torch.nonzero(data.val_mask).squeeze()
    test_idx = torch.nonzero(data.test_mask).squeeze()
    # all_idx = torch.cat([train_idx, val_idx, test_idx])

    dataset_name = 'Facebook'
    base_path = './data/' + dataset_name + '/' + dataset_name
    # Feature normalization
    feat = data.x.numpy()
    feat = np.array(feat, dtype=np.float64)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)
    np.save(base_path + '_feat.npy', feat)

    #get labels
    labels = data.y
    labels = torch.unsqueeze(labels, 1)
    # train_labels = labels.data[train_idx]
    # val_labels = labels.data[val_idx]
    # test_labels = labels.data[test_idx]
    labels = np.array(labels, dtype=np.int32)

    # train_idx = train_idx.numpy()
    # val_idx = val_idx.numpy()
    # test_idx = test_idx.numpy()
    train_idx = np.array(train_idx, dtype=np.int32)
    val_idx = np.array(val_idx, dtype=np.int32)
    test_idx = np.array(test_idx, dtype=np.int32)

    np.savez(base_path + '_labels.npz',
             train_idx=train_idx,
             val_idx=val_idx,
             test_idx=test_idx,
             labels=labels,
             n_classes=n_classes)

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    # Load vertex sorted by timestamp
    idx_with_time_seq = dataset_bfs_sort(dataset[0], dataset_name)
    init_ratio = 0.35

    init_num = round(len(idx_with_time_seq) * init_ratio)
    drop_idx = idx_with_time_seq[init_num:]

    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,
                                                      drop_idx,
                                                      num_nodes=data.num_nodes)

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    # r_indexes = np.arange(len(drop_edge_index[0]))
    # np.random.shuffle(r_indexes)
    # drop_edge_index[0] = drop_edge_index[0][r_indexes]
    # drop_edge_index[1] = drop_edge_index[1][r_indexes]

    row_drop, col_drop = np.array(drop_edge_index)

    f = open(base_path + '_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))

    f.close()

    row, col = data.edge_index
    print(row_drop.shape)
    row = row.numpy()
    col = col.numpy()

    # Write edge_idx (src_edge, dst_edge)
    write_packed_edges(base_path + '_init_edges.txt', row, col)

    # save_adj(row, col, N=data.num_nodes, dataset_name='arxiv', savename='arxiv_init', snap='init')
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in range(num_snap):
        print(sn)
        row_sn = row_drop[sn * snapshot:(sn + 1) * snapshot]
        col_sn = col_drop[sn * snapshot:(sn + 1) * snapshot]
        if sn == 0:
            row_tmp = row
            col_tmp = col

        row_tmp = np.concatenate((row_tmp, row_sn))
        col_tmp = np.concatenate((col_tmp, col_sn))
        row_tmp = np.concatenate((row_tmp, col_sn))
        col_tmp = np.concatenate((col_tmp, row_sn))
        # if (sn + 1) % 20 == 0 or (sn + 1) == num_snap:
        # save_adj(row_tmp,
        #         col_tmp,
        #         N=data.num_nodes,
        #         dataset_name='arxiv',
        #         savename='arxiv_snap' + str(sn + 1),
        #         snap=(sn + 1))

        with open(base_path + '_Edgeupdate_snap' + str(sn + 1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print(dataset_name + ' -- save snapshots finish')


def wikics(num_snap):
    dataset = WikiCS('./dataset/WikiCS')
    data = dataset[0]
    n_classes = np.array(dataset.num_classes, dtype=np.int32)
    train_idx = torch.nonzero(data.train_mask).squeeze()
    val_idx = torch.nonzero(data.val_mask).squeeze()
    test_idx = torch.nonzero(data.test_mask).squeeze()
    # all_idx = torch.cat([train_idx, val_idx, test_idx])

    dataset_name = 'WikiCS'
    base_path = './data/' + dataset_name + '/' + dataset_name
    # Feature normalization
    feat = data.x.numpy()
    feat = np.array(feat, dtype=np.float64)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)
    np.save(base_path + '_feat.npy', feat)

    #get labels
    labels = data.y
    labels = torch.unsqueeze(labels, 1)
    # train_labels = labels.data[train_idx]
    # val_labels = labels.data[val_idx]
    # test_labels = labels.data[test_idx]
    labels = np.array(labels, dtype=np.int32)

    # train_idx = train_idx.numpy()
    # val_idx = val_idx.numpy()
    # test_idx = test_idx.numpy()
    train_idx = np.array(train_idx, dtype=np.int32)
    val_idx = np.array(val_idx, dtype=np.int32)
    test_idx = np.array(test_idx, dtype=np.int32)

    np.savez(base_path + '_labels.npz',
             train_idx=train_idx,
             val_idx=val_idx,
             test_idx=test_idx,
             labels=labels,
             n_classes=n_classes)

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    # Load vertex sorted by timestamp
    idx_with_time_seq = dataset_bfs_sort(dataset[0], dataset_name)
    init_ratio = 0.35

    init_num = round(len(idx_with_time_seq) * init_ratio)
    drop_idx = idx_with_time_seq[init_num:]

    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,
                                                      drop_idx,
                                                      num_nodes=data.num_nodes)

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    # r_indexes = np.arange(len(drop_edge_index[0]))
    # np.random.shuffle(r_indexes)
    # drop_edge_index[0] = drop_edge_index[0][r_indexes]
    # drop_edge_index[1] = drop_edge_index[1][r_indexes]

    row_drop, col_drop = np.array(drop_edge_index)

    f = open(base_path + '_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))

    f.close()

    row, col = data.edge_index
    print(row_drop.shape)
    row = row.numpy()
    col = col.numpy()

    # Write edge_idx (src_edge, dst_edge)
    write_packed_edges(base_path + '_init_edges.txt', row, col)

    # save_adj(row, col, N=data.num_nodes, dataset_name='arxiv', savename='arxiv_init', snap='init')
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in range(num_snap):
        print(sn)
        row_sn = row_drop[sn * snapshot:(sn + 1) * snapshot]
        col_sn = col_drop[sn * snapshot:(sn + 1) * snapshot]
        if sn == 0:
            row_tmp = row
            col_tmp = col

        row_tmp = np.concatenate((row_tmp, row_sn))
        col_tmp = np.concatenate((col_tmp, col_sn))
        row_tmp = np.concatenate((row_tmp, col_sn))
        col_tmp = np.concatenate((col_tmp, row_sn))
        # if (sn + 1) % 20 == 0 or (sn + 1) == num_snap:
        # save_adj(row_tmp,
        #         col_tmp,
        #         N=data.num_nodes,
        #         dataset_name='arxiv',
        #         savename='arxiv_snap' + str(sn + 1),
        #         snap=(sn + 1))

        with open(base_path + '_Edgeupdate_snap' + str(sn + 1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print(dataset_name + ' -- save snapshots finish')


def twitch(num_snap):
    dataset = Twitch('./dataset/Twitch', 'PT')
    data = dataset[0]
    data = dataset_split(data, 0.2, 0.3, 0.5)
    n_classes = np.array(dataset.num_classes, dtype=np.int32)
    train_idx = torch.nonzero(data.train_mask).squeeze()
    val_idx = torch.nonzero(data.val_mask).squeeze()
    test_idx = torch.nonzero(data.test_mask).squeeze()
    # all_idx = torch.cat([train_idx, val_idx, test_idx])

    dataset_name = 'Twitch'
    base_path = './data/' + dataset_name + '/' + dataset_name
    # Feature normalization
    feat = data.x.numpy()
    feat = np.array(feat, dtype=np.float64)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)
    np.save(base_path + '_feat.npy', feat)

    #get labels
    labels = data.y
    labels = torch.unsqueeze(labels, 1)
    # train_labels = labels.data[train_idx]
    # val_labels = labels.data[val_idx]
    # test_labels = labels.data[test_idx]
    labels = np.array(labels, dtype=np.int32)

    # train_idx = train_idx.numpy()
    # val_idx = val_idx.numpy()
    # test_idx = test_idx.numpy()
    train_idx = np.array(train_idx, dtype=np.int32)
    val_idx = np.array(val_idx, dtype=np.int32)
    test_idx = np.array(test_idx, dtype=np.int32)

    np.savez(base_path + '_labels.npz',
             train_idx=train_idx,
             val_idx=val_idx,
             test_idx=test_idx,
             labels=labels,
             n_classes=n_classes)

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    # Load vertex sorted by timestamp
    idx_with_time_seq = dataset_bfs_sort(dataset[0], dataset_name)
    init_ratio = 0.35

    init_num = round(len(idx_with_time_seq) * init_ratio)
    drop_idx = idx_with_time_seq[init_num:]

    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,
                                                      drop_idx,
                                                      num_nodes=data.num_nodes)

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    # r_indexes = np.arange(len(drop_edge_index[0]))
    # np.random.shuffle(r_indexes)
    # drop_edge_index[0] = drop_edge_index[0][r_indexes]
    # drop_edge_index[1] = drop_edge_index[1][r_indexes]

    row_drop, col_drop = np.array(drop_edge_index)

    f = open(base_path + '_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))

    f.close()

    row, col = data.edge_index
    print(row_drop.shape)
    row = row.numpy()
    col = col.numpy()

    # Write edge_idx (src_edge, dst_edge)
    write_packed_edges(base_path + '_init_edges.txt', row, col)

    # save_adj(row, col, N=data.num_nodes, dataset_name='arxiv', savename='arxiv_init', snap='init')
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in range(num_snap):
        print(sn)
        row_sn = row_drop[sn * snapshot:(sn + 1) * snapshot]
        col_sn = col_drop[sn * snapshot:(sn + 1) * snapshot]
        if sn == 0:
            row_tmp = row
            col_tmp = col

        row_tmp = np.concatenate((row_tmp, row_sn))
        col_tmp = np.concatenate((col_tmp, col_sn))
        row_tmp = np.concatenate((row_tmp, col_sn))
        col_tmp = np.concatenate((col_tmp, row_sn))
        # if (sn + 1) % 20 == 0 or (sn + 1) == num_snap:
        # save_adj(row_tmp,
        #         col_tmp,
        #         N=data.num_nodes,
        #         dataset_name='arxiv',
        #         savename='arxiv_snap' + str(sn + 1),
        #         snap=(sn + 1))

        with open(base_path + '_Edgeupdate_snap' + str(sn + 1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print(dataset_name + ' -- save snapshots finish')


def reddit(num_snap):
    dataset_name = 'reddit'
    dataset = Reddit('./dataset')
    data = dataset[0]
    n_classes = np.array(dataset.num_classes, dtype=np.int32)
    train_idx = torch.nonzero(data.train_mask).squeeze()
    val_idx = torch.nonzero(data.val_mask).squeeze()
    test_idx = torch.nonzero(data.test_mask).squeeze()
    # all_idx = torch.cat([train_idx, val_idx, test_idx])

    base_path = './data/' + dataset_name + '/' + dataset_name
    # Feature normalization
    feat = data.x.numpy()
    feat = np.array(feat, dtype=np.float64)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)
    np.save(base_path + '_feat.npy', feat)

    #get labels
    labels = data.y
    labels = torch.unsqueeze(labels, 1)
    # train_labels = labels.data[train_idx]
    # val_labels = labels.data[val_idx]
    # test_labels = labels.data[test_idx]
    labels = np.array(labels, dtype=np.int32)

    train_idx = train_idx.numpy()
    val_idx = val_idx.numpy()
    test_idx = test_idx.numpy()
    train_idx = np.array(train_idx, dtype=np.int32)
    val_idx = np.array(val_idx, dtype=np.int32)
    test_idx = np.array(test_idx, dtype=np.int32)

    np.savez(base_path + '_labels.npz',
             train_idx=train_idx,
             val_idx=val_idx,
             test_idx=test_idx,
             labels=labels,
             n_classes=n_classes)

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    # data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,
    #                                                   train_idx,
    #                                                   num_nodes=data.num_nodes)

    # test_num = len(test_idx)
    drop_idx = train_idx
    # drop_idx = np.append(drop_idx, test_idx)
    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,
                                                      drop_idx,
                                                      num_nodes=data.num_nodes)

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    # r_indexes = np.arange(len(drop_edge_index[0]))
    # np.random.shuffle(r_indexes)
    # drop_edge_index[0] = drop_edge_index[0][r_indexes]
    # drop_edge_index[1] = drop_edge_index[1][r_indexes]

    row_drop, col_drop = np.array(drop_edge_index)

    f = open(base_path + '_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()

    row, col = data.edge_index
    print(row_drop.shape)
    row = row.numpy()
    col = col.numpy()

    # # Write edge_idx (src_edge, dst_edge)
    # write_packed_edges('./data/arxiv/arxiv_init_src_edge.txt', 'I', row)
    # write_packed_edges('./data/arxiv/arxiv_init_dst_edge.txt', 'I', col)

    # Write edge_idx (src_edge, dst_edge)
    write_packed_edges(base_path + '_init_edges.txt', row, col)

    # save_adj(row, col, N=data.num_nodes, dataset_name='arxiv', savename='arxiv_init', snap='init')
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in range(num_snap):
        print(sn)
        row_sn = row_drop[sn * snapshot:(sn + 1) * snapshot]
        col_sn = col_drop[sn * snapshot:(sn + 1) * snapshot]
        if sn == 0:
            row_tmp = row
            col_tmp = col

        row_tmp = np.concatenate((row_tmp, row_sn))
        col_tmp = np.concatenate((col_tmp, col_sn))
        row_tmp = np.concatenate((row_tmp, col_sn))
        col_tmp = np.concatenate((col_tmp, row_sn))
        # if (sn + 1) % 20 == 0 or (sn + 1) == num_snap:
        # save_adj(row_tmp,
        #         col_tmp,
        #         N=data.num_nodes,
        #         dataset_name='arxiv',
        #         savename='arxiv_snap' + str(sn + 1),
        #         snap=(sn + 1))

        with open(base_path + '_Edgeupdate_snap' + str(sn + 1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print(dataset_name + ' -- save snapshots finish')


def arxiv(num_snap):
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./dataset/')
    data = dataset[0]
    n_classes = np.array(dataset.meta_info['num classes'], dtype=np.int32)
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    all_idx = torch.cat([train_idx, val_idx, test_idx])

    # Feature normalization
    feat = data.x.numpy()
    feat = np.array(feat, dtype=np.float64)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)
    np.save('./data/arxiv/arxiv_feat.npy', feat)

    #get labels
    labels = data.y
    train_labels = labels.data[train_idx]
    val_labels = labels.data[val_idx]
    test_labels = labels.data[test_idx]
    labels = np.array(labels, dtype=np.int32)

    train_idx = train_idx.numpy()
    val_idx = val_idx.numpy()
    test_idx = test_idx.numpy()
    train_idx = np.array(train_idx, dtype=np.int32)
    val_idx = np.array(val_idx, dtype=np.int32)
    test_idx = np.array(test_idx, dtype=np.int32)

    np.savez('./data/arxiv/arxiv_labels.npz',
             train_idx=train_idx,
             val_idx=val_idx,
             test_idx=test_idx,
             labels=labels,
             n_classes=n_classes)

    # Load vertex sorted by timestamp
    idx_with_time_seq = dataset_timestamp_sort(dataset[0], 'arxiv')
    init_ratio = 0.3

    init_num = round(len(idx_with_time_seq) * init_ratio)
    drop_idx = idx_with_time_seq[init_num:]

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,
                                                      drop_idx,
                                                      num_nodes=data.num_nodes)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row_drop, col_drop = np.array(drop_edge_index)

    f = open('./data/arxiv/ogbn-arxiv_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()

    row, col = data.edge_index
    print(row_drop.shape)
    row = row.numpy()
    col = col.numpy()

    # # Write edge_idx (src_edge, dst_edge)
    # write_packed_edges('./data/arxiv/arxiv_init_src_edge.txt', 'I', row)
    # write_packed_edges('./data/arxiv/arxiv_init_dst_edge.txt', 'I', col)

    # Write edge_idx (src_edge, dst_edge)
    write_packed_edges('./data/arxiv/arxiv_init_edges.txt', row, col)

    # save_adj(row, col, N=data.num_nodes, dataset_name='arxiv', savename='arxiv_init', snap='init')
    # num_snap = 16
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in range(num_snap):
        print(sn)
        row_sn = row_drop[sn * snapshot:(sn + 1) * snapshot]
        col_sn = col_drop[sn * snapshot:(sn + 1) * snapshot]
        if sn == 0:
            row_tmp = row
            col_tmp = col

        row_tmp = np.concatenate((row_tmp, row_sn))
        col_tmp = np.concatenate((col_tmp, col_sn))
        row_tmp = np.concatenate((row_tmp, col_sn))
        col_tmp = np.concatenate((col_tmp, row_sn))
        # if (sn + 1) % 20 == 0 or (sn + 1) == num_snap:
        # save_adj(row_tmp,
        #          col_tmp,
        #          N=data.num_nodes,
        #          dataset_name='arxiv',
        #          savename='arxiv_snap' + str(sn + 1),
        #          snap=(sn + 1))

        with open('./data/arxiv/arxiv_Edgeupdate_snap' + str(sn + 1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('Arxiv -- save snapshots finish')


def mag(num_snap):
    dataset = DglNodePropPredDataset(name='ogbn-mag', root='./dataset/')

    data, labels = dataset[0]
    n_classes = np.array(dataset.meta_info['num classes'], dtype=np.int32)
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx['train']['paper'], split_idx['valid'][
        'paper'], split_idx['test']['paper']
    all_idx = torch.cat([train_idx, val_idx, test_idx])

    # Feature normalization
    feat = data.nodes['paper'].data['feat'].numpy()
    feat = np.array(feat, dtype=np.float64)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)
    np.save('./data/mag/mag_feat.npy', feat)

    #get labels
    labels = labels['paper']
    train_labels = labels.data[train_idx]
    val_labels = labels.data[val_idx]
    test_labels = labels.data[test_idx]
    labels = np.array(labels, dtype=np.int32)

    train_idx = train_idx.numpy()
    val_idx = val_idx.numpy()
    test_idx = test_idx.numpy()
    train_idx = np.array(train_idx, dtype=np.int32)
    val_idx = np.array(val_idx, dtype=np.int32)
    test_idx = np.array(test_idx, dtype=np.int32)

    np.savez('./data/mag/mag_labels.npz',
             train_idx=train_idx,
             val_idx=val_idx,
             test_idx=test_idx,
             labels=labels,
             n_classes=n_classes)

    sub_g = dgl.edge_type_subgraph(data, [('paper', 'cites', 'paper')])
    h_sub_g = dgl.to_homogeneous(sub_g)
    edge_index = torch.stack([h_sub_g.edges()[0], h_sub_g.edges()[1]], dim=0)
    num_nodes = h_sub_g.number_of_nodes()

    # Load vertex sorted by timestamp
    idx_with_time_seq = dataset_timestamp_sort(dataset[0], 'mag')
    init_ratio = 0.3

    init_num = round(len(idx_with_time_seq) * init_ratio)
    drop_idx = idx_with_time_seq[init_num:]

    edge_index = to_undirected(edge_index, num_nodes)
    edge_index, drop_edge_index, _ = dropout_adj(edge_index, drop_idx, num_nodes=num_nodes)
    edge_index = to_undirected(edge_index, num_nodes)

    row_drop, col_drop = np.array(drop_edge_index)

    f = open('./data/mag/ogbn-mag_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()

    row, col = edge_index
    print(row_drop.shape)
    row = row.numpy()
    col = col.numpy()

    # # Write edge_idx (src_edge, dst_edge)
    # write_packed_edges('../data/arxiv/arxiv_init_src_edge.txt', 'I', row)
    # write_packed_edges('../data/arxiv/arxiv_init_dst_edge.txt', 'I', col)

    # Write edge_idx (src_edge, dst_edge)
    write_packed_edges('./data/mag/mag_init_edges.txt', row, col)

    # save_adj(row, col, N=data.num_nodes, dataset_name='arxiv', savename='arxiv_init', snap='init')
    # num_snap = 16
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in range(num_snap):
        print(sn)
        row_sn = row_drop[sn * snapshot:(sn + 1) * snapshot]
        col_sn = col_drop[sn * snapshot:(sn + 1) * snapshot]
        if sn == 0:
            row_tmp = row
            col_tmp = col

        row_tmp = np.concatenate((row_tmp, row_sn))
        col_tmp = np.concatenate((col_tmp, col_sn))
        row_tmp = np.concatenate((row_tmp, col_sn))
        col_tmp = np.concatenate((col_tmp, row_sn))
        # if (sn + 1) % 20 == 0 or (sn + 1) == num_snap:
        # save_adj(row_tmp,
        #          col_tmp,
        #          N=data.num_nodes,
        #          dataset_name='arxiv',
        #          savename='arxiv_snap' + str(sn + 1),
        #          snap=(sn + 1))

        with open('./data/mag/mag_Edgeupdate_snap' + str(sn + 1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('Mag -- save snapshots finish')


def products(num_snap):
    dataset = PygNodePropPredDataset(name='ogbn-products', root='./dataset/')
    data = dataset[0]
    n_classes = np.array(dataset.meta_info['num classes'], dtype=np.int32)
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    all_idx = torch.cat([train_idx, val_idx, test_idx])

    #save feat
    feat = data.x.numpy()
    feat = np.array(feat, dtype=np.float64)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)
    np.save('./data/products/products_feat.npy', feat)

    #get labels
    print("save labels.....")
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

    labels = data.y
    train_labels = labels.data[train_idx]
    val_labels = labels.data[val_idx]
    test_labels = labels.data[test_idx]
    labels = np.array(labels, dtype=np.int32)

    train_idx = train_idx.numpy()
    val_idx = val_idx.numpy()
    test_idx = test_idx.numpy()
    train_idx = np.array(train_idx, dtype=np.int32)
    val_idx = np.array(val_idx, dtype=np.int32)
    test_idx = np.array(test_idx, dtype=np.int32)

    train_labels = train_labels.numpy().T
    val_labels = val_labels.numpy().T
    test_labels = test_labels.numpy().T

    train_labels = np.array(train_labels, dtype=np.int32)
    val_labels = np.array(val_labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)
    train_labels = train_labels.reshape(train_labels.shape[1])
    val_labels = val_labels.reshape(val_labels.shape[1])
    test_labels = test_labels.reshape(test_labels.shape[1])
    np.savez(
        './data/products/products_labels.npz',
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        #  train_labels=train_labels,
        #  val_labels=val_labels,
        #  test_labels=test_labels,
        labels=labels,
        n_classes=n_classes)

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,
                                                      train_idx,
                                                      num_nodes=data.num_nodes)

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row_drop, col_drop = np.array(drop_edge_index)
    print('row_drop.shape:', row_drop.shape)
    f = open('./data/products/ogbn-products_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()

    row, col = data.edge_index
    row = row.numpy()
    col = col.numpy()

    # Write edge_idx (src_edge, dst_edge)
    write_packed_edges('./data/products/products_init_edges.txt', row, col)

    # save_adj(row,
    #          col,
    #          N=data.num_nodes,
    #          dataset_name='products',
    #          savename='products_init',
    #          snap='init')
    # num_snap = 15
    snapshot = math.floor(row_drop.shape[0] / num_snap)
    print('num_snap: ', num_snap)

    for sn in range(num_snap):
        print(sn)
        row_sn = row_drop[sn * snapshot:(sn + 1) * snapshot]
        col_sn = col_drop[sn * snapshot:(sn + 1) * snapshot]
        if sn == 0:
            row_tmp = row
            col_tmp = col

        row_tmp = np.concatenate((row_tmp, row_sn))
        col_tmp = np.concatenate((col_tmp, col_sn))
        row_tmp = np.concatenate((row_tmp, col_sn))
        col_tmp = np.concatenate((col_tmp, row_sn))

        # save_adj(row_tmp,
        #          col_tmp,
        #          N=data.num_nodes,
        #          dataset_name='products',
        #          savename='products_snap' + str(sn + 1),
        #          snap=(sn + 1))

        with open('./data/products/products_Edgeupdate_snap' + str(sn + 1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('Products -- save snapshots finish')


def papers100M(num_snap):
    s_time = time.time()
    dataset = PygNodePropPredDataset("ogbn-papers100M", root='./dataset/')
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    n_classes = np.array(dataset.meta_info['num classes'], dtype=np.int32)

    feat = data.x.numpy()
    feat = np.array(feat, dtype=np.float64)

    #normalize feats
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)

    #save feats
    np.save('./data/papers100M/papers100M_feat.npy', feat)
    del feat
    gc.collect()

    #get labels
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    all_idx = torch.cat([train_idx, val_idx, test_idx])

    labels = data.y
    train_labels = labels.data[train_idx]
    val_labels = labels.data[val_idx]
    test_labels = labels.data[test_idx]
    labels = np.array(labels, dtype=np.int32)

    train_idx = train_idx.numpy()
    val_idx = val_idx.numpy()
    test_idx = test_idx.numpy()
    train_idx = np.array(train_idx, dtype=np.int32)
    val_idx = np.array(val_idx, dtype=np.int32)
    test_idx = np.array(test_idx, dtype=np.int32)

    train_labels = train_labels.numpy().T
    val_labels = val_labels.numpy().T
    test_labels = test_labels.numpy().T

    train_labels = np.array(train_labels, dtype=np.int32)
    val_labels = np.array(val_labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)
    train_labels = train_labels.reshape(train_labels.shape[1])
    val_labels = val_labels.reshape(val_labels.shape[1])
    test_labels = test_labels.reshape(test_labels.shape[1])
    np.savez(
        './data/papers100M/papers100M_labels.npz',
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        #  train_labels=train_labels,
        #  val_labels=val_labels,
        #  test_labels=test_labels,
        labels=labels,
        n_classes=n_classes)

    print('making the graph undirected')
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    print("process finished cost:", time.time() - s_time)

    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,
                                                      train_idx,
                                                      num_nodes=data.num_nodes)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row_drop, col_drop = np.array(drop_edge_index)
    row, col = data.edge_index
    row = row.numpy()
    col = col.numpy()
    # Write edge_idx (src_edge, dst_edge)
    write_packed_edges('./data/papers100M/papers100M_init_edges.txt', row, col)

    # save_adj(row,
    #          col,
    #          N=data.num_nodes,
    #          dataset_name='papers100M',
    #          savename='papers100M_init',
    #          snap='init')
    row = row.numpy()
    col = col.numpy()
    # num_snap = 20
    print('num_snap: ', num_snap)
    snapshot = math.floor(row_drop.shape[0] / num_snap)

    for sn in range(num_snap):
        st = sn + 1
        print('snap:', st)

        row_sn = row_drop[sn * snapshot:st * snapshot]
        col_sn = col_drop[sn * snapshot:st * snapshot]
        if sn == 0:
            row_tmp = row
            col_tmp = col
        row_tmp = np.concatenate((row_tmp, row_sn))
        col_tmp = np.concatenate((col_tmp, col_sn))
        row_tmp = np.concatenate((row_tmp, col_sn))
        col_tmp = np.concatenate((col_tmp, row_sn))

        #save_adj(row_tmp, col_tmp, N=data.num_nodes, dataset_name='papers100M', savename='papers100M_snap'+str(st), snap=st)

        with open('./data/papers100M/papers100M_Edgeupdate_snap' + str(st) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('Papers100M -- save snapshots finish')


def save_adj(row, col, N, dataset_name, savename, snap, full=False):
    adj = sp.csr_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
    # Add identity matrix (add self-loop)
    adj = adj + sp.eye(adj.shape[0])
    print('snap:', snap, ', edge:', adj.nnz)
    save_path = './data/' + dataset_name + '/'

    EL = adj.indices
    PL = adj.indptr

    del adj
    gc.collect()

    EL = np.array(EL, dtype=np.uint32)
    PL = np.array(PL, dtype=np.uint32)
    EL_re = []

    for i in range(1, PL.shape[0]):
        EL_re += sorted(EL[PL[i - 1]:PL[i]], key=lambda x: PL[x + 1] - PL[x])
    EL_re = np.asarray(EL_re, dtype=np.uint32)

    #save graph
    with open(save_path + savename + '_adj_el.txt', 'wb') as f1:
        for i in EL_re:
            m = struct.pack('I', i)
            f1.write(m)

    with open(save_path + savename + '_adj_pl.txt', 'wb') as f2:
        for i in PL:
            m = struct.pack('I', i)
            f2.write(m)

    del EL
    del PL
    del EL_re
    gc.collect()


# def write_packed_file(f_path, pack_fmt, data):
#     with open(f_path, 'wb') as f:
#         for i in data:
#             m = struct.pack(pack_fmt, i)
#             f.write(m)


def write_packed_edges(f_path, edge_src, edge_dst):
    assert len(edge_src) == len(edge_dst)
    with open(f_path, 'wb') as f:
        for i in range(len(edge_src)):
            m = struct.pack("ii", edge_src[i], edge_dst[i])
            f.write(m)


if __name__ == "__main__":
    # papers100M()
    # products()
    # arxiv()

    # gen_dataset_snapshot('Cora', 10)
    # gen_dataset_snapshot('CiteSeer', 10)
    # gen_dataset_snapshot('Facebook', 10)
    # gen_dataset_snapshot('WikiCS', 10)
    gen_dataset_snapshot('Twitch', 10)
    # gen_dataset_snapshot('PubMed', 10)
    # gen_dataset_snapshot('arxiv', 16)
    # gen_dataset_snapshot('reddit', 16)
    # gen_dataset_snapshot('products', 16)
    # gen_dataset_snapshot('mag', 16)
    # gen_dataset_snapshot('papers100M', 20)
