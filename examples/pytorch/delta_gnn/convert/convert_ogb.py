import argparse
from tqdm import tqdm
import numpy as np
import random
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset
import sklearn.preprocessing
import tracemalloc
import gc
import struct
from torch_sparse import coalesce
import math
import pdb
import time

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


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
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
    else:
        edge_index = torch.stack([new_row, new_col], dim=0)
    drop_edge_index = torch.stack([drop_row, drop_col], dim=0)  ### only u->v (no v->u)

    return edge_index, drop_edge_index, edge_attr


def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


def arxiv():
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='../dataset/')
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
    np.save('../data/arxiv/arxiv_feat.npy', feat)

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

    np.savez('../data/arxiv/arxiv_labels.npz',
             train_idx=train_idx,
             val_idx=val_idx,
             test_idx=test_idx,
             labels=labels,
             n_classes = n_classes)

    # train_labels = train_labels.numpy().T
    # val_labels = val_labels.numpy().T
    # test_labels = test_labels.numpy().T

    # train_labels = np.array(train_labels, dtype=np.int32)
    # val_labels = np.array(val_labels, dtype=np.int32)
    # test_labels = np.array(test_labels, dtype=np.int32)
    # train_labels = train_labels.reshape(train_labels.shape[1])
    # val_labels = val_labels.reshape(val_labels.shape[1])
    # test_labels = test_labels.reshape(test_labels.shape[1])
    # np.savez('../data/arxiv/arxiv_labels.npz',
    #          train_idx=train_idx,
    #          val_idx=val_idx,
    #          test_idx=test_idx,
    #          train_labels=train_labels,
    #          val_labels=val_labels,
    #          test_labels=test_labels)

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,
                                                      train_idx,
                                                      num_nodes=data.num_nodes)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row_drop, col_drop = np.array(drop_edge_index)

    f = open('../data/arxiv/ogbn-arxiv_update_full.txt', 'w+')
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
    write_packed_file('../data/arxiv/arxiv_init_src_edge.txt', 'I', row)
    write_packed_file('../data/arxiv/arxiv_init_dst_edge.txt', 'I', col)

    save_adj(row, col, N=data.num_nodes, dataset_name='arxiv', savename='arxiv_init', snap='init')
    num_snap = 16
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
        if (sn + 1) % 20 == 0 or (sn + 1) == num_snap:
            save_adj(row_tmp,
                     col_tmp,
                     N=data.num_nodes,
                     dataset_name='arxiv',
                     savename='arxiv_snap' + str(sn + 1),
                     snap=(sn + 1))

        with open('../data/arxiv/arxiv_Edgeupdate_snap' + str(sn + 1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('Arxiv -- save snapshots finish')


def products():
    dataset = PygNodePropPredDataset(name='ogbn-products')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    all_idx = torch.cat([train_idx, val_idx, test_idx])

    #save feat
    feat = data.x.numpy()
    feat = np.array(feat, dtype=np.float64)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)
    np.save('../data/products/products_feat.npy', feat)

    #get labels
    print("save labels.....")
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

    labels = data.y
    train_labels = labels.data[train_idx]
    val_labels = labels.data[val_idx]
    test_labels = labels.data[test_idx]

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
    np.savez('../data/products/products_labels.npz',
             train_idx=train_idx,
             val_idx=val_idx,
             test_idx=test_idx,
             train_labels=train_labels,
             val_labels=val_labels,
             test_labels=test_labels)

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,
                                                      train_idx,
                                                      num_nodes=data.num_nodes)

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row_drop, col_drop = np.array(drop_edge_index)
    print('row_drop.shape:', row_drop.shape)
    f = open('../data/products/ogbn-products_update_full.txt', 'w+')
    for k in range(row_drop.shape[0]):
        v_from = row_drop[k]
        v_to = col_drop[k]
        f.write('%d %d\n' % (v_from, v_to))
        f.write('%d %d\n' % (v_to, v_from))
    f.close()

    row, col = data.edge_index
    row = row.numpy()
    col = col.numpy()
    save_adj(row,
             col,
             N=data.num_nodes,
             dataset_name='products',
             savename='products_init',
             snap='init')
    num_snap = 15
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

        save_adj(row_tmp,
                 col_tmp,
                 N=data.num_nodes,
                 dataset_name='products',
                 savename='products_snap' + str(sn + 1),
                 snap=(sn + 1))

        with open('../data/products/products_Edgeupdate_snap' + str(sn + 1) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('Products -- save snapshots finish')


def papers100M():
    s_time = time.time()
    dataset = PygNodePropPredDataset("ogbn-papers100M")
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    feat = data.x.numpy()
    feat = np.array(feat, dtype=np.float64)

    #normalize feats
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat)
    feat = scaler.transform(feat)

    #save feats
    np.save('../data/papers100M/papers100M_feat.npy', feat)
    del feat
    gc.collect()

    #get labels
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    all_idx = torch.cat([train_idx, val_idx, test_idx])

    labels = data.y
    train_labels = labels.data[train_idx]
    val_labels = labels.data[val_idx]
    test_labels = labels.data[test_idx]

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
    np.savez('../data/papers100M/papers100M_labels.npz',
             train_idx=train_idx,
             val_idx=val_idx,
             test_idx=test_idx,
             train_labels=train_labels,
             val_labels=val_labels,
             test_labels=test_labels)

    print('making the graph undirected')
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    print("process finished cost:", time.time() - s_time)

    data.edge_index, drop_edge_index, _ = dropout_adj(data.edge_index,
                                                      train_idx,
                                                      num_nodes=data.num_nodes)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row_drop, col_drop = np.array(drop_edge_index)
    row, col = data.edge_index
    save_adj(row,
             col,
             N=data.num_nodes,
             dataset_name='papers100M',
             savename='papers100M_init',
             snap='init')
    row = row.numpy()
    col = col.numpy()
    num_snap = 20
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

        with open('../data/papers100M/papers100M_Edgeupdate_snap' + str(st) + '.txt', 'w') as f:
            for i, j in zip(row_sn, col_sn):
                f.write("%d %d\n" % (i, j))
                f.write("%d %d\n" % (j, i))
    print('Papers100M -- save snapshots finish')


def save_adj(row, col, N, dataset_name, savename, snap, full=False):
    adj = sp.csr_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
    # Add identity matrix (add self-loop)
    adj = adj + sp.eye(adj.shape[0])
    print('snap:', snap, ', edge:', adj.nnz)
    save_path = '../data/' + dataset_name + '/'

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


def write_packed_file(f_path, pack_fmt, data):
    with open(f_path, 'wb') as f:
        for i in data:
            m = struct.pack(pack_fmt, i)
            f.write(m)


if __name__ == "__main__":
    #papers100M()
    #products()
    arxiv()
