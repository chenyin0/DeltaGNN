import torch
import gc
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
# from propagation import InstantGNN
import pdb
import struct
from torch_sparse import SparseTensor


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
    memory_dataset = py_alg.initial_operation('./data/aminer/', datastr, m, n, rmax, alpha,
                                              features)
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

    return features, train_labels, val_labels, test_labels, train_idx, val_idx, test_idx, memory_dataset, py_alg


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


def insert_edges(orig_edge_index, inserted_edge_index):
    """
    orig_edge_index: format (src, dst)
    inserted _edge_index: format (src, dst)
    """
    return torch.cat([orig_edge_index, inserted_edge_index], dim=-1)


def insert_edges_delta(orig_edge_index, inserted_edge_index, threshold):
    """
    orig_edge_index: format (src, dst)
    inserted _edge_index: format (src, dst)
    """


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