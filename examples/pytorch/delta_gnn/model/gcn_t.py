"""GCN using PyG nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch as th
import torch.nn as nn
# from dgl.nn.pytorch import GraphConv
from torch_geometric.nn import GCNConv
# from .graphconv_delta import GraphConv_delta
import torch.nn.functional as F
from torch.optim import Adam
# from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
# import torch_geometric.transforms as T
import numpy as np
import time
# from tqdm import tqdm


class GCN(nn.Module):

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        if n_layers > 1:
            self.layers.append(GCNConv(in_feats, n_hidden))
            for i in range(1, n_layers - 1):
                self.layers.append(GCNConv(n_hidden, n_hidden))
            self.layers.append(GCNConv(n_hidden, n_classes))
        else:
            self.layers.append(GCNConv(in_feats, n_classes))

        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.layers:
            # print(layer.lin.weight)
            layer.reset_parameters()
            # print(layer.lin.weight)
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, features, edge_index):
        h = features
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layers[-1](h, edge_index)
        # return h.log_softmax(dim=-1)
        return h


def train(model, device, train_loader, lr, weight_decay):
    model.train()
    # pbar = tqdm(total=int(len(train_loader.dataset)))
    # pbar.set_description(f'Epoch {epoch:02d}')
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    time_epoch = 0
    loss_list = []
    for step, batch in enumerate(train_loader):
        t_st = time.time()
        x, edge_index, y = batch.x.to(device), batch.edge_index.to(device), batch.y.to(device)
        optimizer.zero_grad()
        out = model(x, edge_index)
        # loss = F.nll_loss(out, y.squeeze(1))
        loss = F.cross_entropy(out, y.squeeze(1))
        loss.backward()
        optimizer.step()
        time_epoch += (time.time() - t_st)
        loss_list.append(loss.item())
        # print(loss.item())
        # pbar.update(batch.batch_size)
    return np.mean(loss_list), time_epoch


@torch.no_grad()
def validate(model, device, loader):
    model.eval()
    # adj_t = adj_t.cuda(device)
    y_pred, y_true = [], []
    for step, batch in enumerate(loader):
        x, edge_index, y = batch.x.to(device), batch.edge_index.to(device), batch.y
        out = model(x, edge_index)
        # y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_pred.append(torch.argmax(out, dim=-1, keepdim=True).cpu())
        y_true.append(y)
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    correct = torch.sum(y_pred == y_true)
    return correct.item() * 1.0 / len(y_true)


@torch.no_grad()
def test(model, device, loader, checkpt_file):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    # adj_t = adj_t.to(device)
    y_pred, y_true = [], []
    for step, batch in enumerate(loader):
        x, edge_index, y = batch.x.to(device), batch.edge_index.to(device), batch.y
        out = model(x, edge_index)
        # y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_pred.append(torch.argmax(out, dim=-1, keepdim=True).cpu())
        y_true.append(y)
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    correct = torch.sum(y_pred == y_true)
    return correct.item() * 1.0 / len(y_true)


class GCN_delta(nn.Module):
    r"""
    Initial para:
        g: subgraph of original graph
        in_feats: feature matrix of current subgraph
    """

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout, feat_init_num):
        super(GCN_delta, self).__init__()
        # self.g = g
        self.layers = nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        if n_layers > 1:
            self.layers.append(GCNConv(in_feats, n_hidden))
            self.bns.append(torch.nn.BatchNorm1d(n_hidden))
            for i in range(1, n_layers - 1):
                self.layers.append(GCNConv(n_hidden, n_hidden))
                self.bns.append(torch.nn.BatchNorm1d(n_hidden))
            self.layers.append(GCNConv(n_hidden, n_classes))
        else:
            self.layers.append(GCNConv(in_feats, n_classes))

        # self.dropout = nn.Dropout(p=dropout)
        self.dropout = dropout

        # Record previous embedding
        self.embedding = th.Tensor([[0 for i in range(n_classes)]
                                    for j in range(feat_init_num)]).requires_grad_(True)
        # self.embedding = []
        # for i in range(n_layers - 1):
        #     self.embedding.append(
        #         th.Tensor([[0 for i in range(n_hidden)]
        #                    for j in range(g.number_of_nodes())]).requires_grad_(True))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, features, edge_index):
        h = features
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h, edge_index)
            # h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layers[-1](h, edge_index)
        # return h.log_softmax(dim=-1)
        return h


def feature_scatter(dst_t, src_t, index):
    """
    Usage: Replace src_t -> dst_t (https://zhuanlan.zhihu.com/p/339043454)

    Args:
        src_t (Tensor): the source tensor
        dst_t (Tensor): the target tensor
        index (Numpy): the index of dst_t for scatter
    """

    src_index = np.array([[index[row]] * dst_t.shape[1] for row in range(src_t.shape[0])])
    src_index = th.from_numpy(src_index)
    dst_t_tmp = dst_t.scatter(0, src_index, src_t)

    return dst_t_tmp


def feature_scatter_add(dst_t, src_t, index):
    """
    Usage: Replace src_t -> dst_t + src_t (https://zhuanlan.zhihu.com/p/339043454)

    Args:
        src_t (Tensor): the source tensor
        dst_t (Tensor): the target tensor
        index (Numpy): the index of dst_t for scatter
    """

    src_index = np.array([[index[row]] * dst_t.shape[1] for row in range(src_t.shape[0])])
    src_index = th.from_numpy(src_index)
    dst_t_tmp = dst_t.scatter_add(0, src_index, src_t)

    return dst_t_tmp


def feature_merge(embedding, feat, feat_n_id, v_sen_id, v_insen_id):
    """ Update feat and model embedding
    Usage:
        embedding (2d Tensor): all the vertices feature of GNN model
        feat (2d Tensor): updated features in this round forward
        feat_n_id (Tensor): original vertex ID of each vertex feature in "feat"
        v_sen (Tensor): vertex ID of sensititive vertices
        v_insen (Tensor): vertex ID of insensitive vertices
    """

    # Compulsorily execute in CPU (GPU not suits for scalar execution)
    device = feat.device
    time_start = time.perf_counter()
    feat = feat.to('cpu')
    embedding = embedding.to('cpu')
    # load_time = time.perf_counter() - time_start
    # print('>> Load feat: {}'.format(util.time_format(load_time)))

    ##
    """ Workflow:
        1. Update the embedding with updated features
        2. Read out features from the updated embedding
    """

    feat_n_id = feat_n_id.cpu().squeeze().numpy()
    # v_sen_id = np.array(list(v_sen_id))
    v_insen_id = np.array(list(v_insen_id))
    v_sen_id = np.setdiff1d(feat_n_id, v_insen_id)
    # v_sen_id = np.intersect1d(feat_n_id, v_sen_id)
    v_insen_id = np.intersect1d(feat_n_id, v_insen_id)

    v_sen_feat_loc = np.zeros_like(v_sen_id)
    v_insen_feat_loc = np.zeros_like(v_insen_id)
    for i, v_sen in enumerate(v_sen_id):
        v_sen_feat_loc[i] = np.where(feat_n_id == v_sen)[0][0]
    for i, v_insen in enumerate(v_insen_id):
        v_insen_feat_loc[i] = np.where(feat_n_id == v_insen)[0][0]

    v_sen_feat_loc = th.Tensor(v_sen_feat_loc).long()
    v_insen_feat_loc = th.Tensor(v_insen_feat_loc).long()
    feat_sen = feat.index_select(0, v_sen_feat_loc)
    if len(v_insen_feat_loc) != 0:
        feat_insen = feat.index_select(0, v_insen_feat_loc)

    embedding_update_tmp = feature_scatter(embedding, feat_sen, v_sen_id)
    if len(v_insen_feat_loc) != 0:
        embedding_update = feature_scatter_add(embedding_update_tmp, feat_insen, v_insen_id)
    else:
        embedding_update = embedding_update_tmp

    feat_n_id = th.from_numpy(feat_n_id)
    if len(v_insen_feat_loc) != 0:
        feat_update = embedding_update.index_select(0, feat_n_id)
    else:
        feat_update = feat

    feat_update = feat_update.to(device)
    embedding_update = embedding_update.to(device)

    return feat_update, embedding_update


# def combine_embedding(embedding_total, feat, feat_n_id, v_sen, v_insen):
#     """
#     With batch-wise combination

#     ind: the feature index in current batch (index is belongs to the original feature)
#     """
#     # Compulsorily execute in CPU (GPU not suits for scalar execution)
#     device = feat.device
#     time_start = time.perf_counter()
#     feat = feat.to('cpu')
#     embedding_total = embedding_total.to('cpu')
#     # load_time = time.perf_counter() - time_start
#     # print('>> Load feat: {}'.format(util.time_format(load_time)))

#     ##
#     r"""
#     Para:
#     1. feat_prev: features in the last time
#     2. feat: updated features under edge_mask (all-neighbor updating for high degree, and delta-neighbor updating for low degree)
#     3. feat_low_deg: updated features with low degree

#     Method:
#     1. First, replace items in "feat" to which in "feat_prev" with the corresponding node_id
#     2. Then, merge "feat_low_deg" with "add" operation to "feat"
#     """
#     feat_n_id = feat_n_id.cpu().squeeze().numpy()
#     v_sen = np.array(list(v_sen))
#     v_insen = np.array(list(v_insen))
#     # sen_mask = np.isin(ind, v_sen)
#     insen_mask = np.isin(feat_n_id, v_insen)
#     sen_mask = ~insen_mask
#     # sen_mask = [True for i in range(ind.shape[0])]

#     # # sen_mask_resize = [[i for col in range(feat.shape[-1])] for i in sen_mask]
#     # sen_mask_resize = [[i] * feat.shape[-1] for i in sen_mask]
#     # # print(sen_mask_resize)
#     # # insen_mask_resize = [[i for col in range(feat.shape[-1])] for i in insen_mask]
#     # insen_mask_resize = [[i] * feat.shape[-1] for i in insen_mask]

#     # sen_feat = th.masked_select(feat, th.tensor(sen_mask_resize, dtype=torch.bool))
#     sen_mask_ts = torch.unsqueeze(th.tensor(sen_mask, dtype=torch.bool), 1)
#     sen_feat = th.masked_select(feat, sen_mask_ts)
#     sen_feat = th.reshape(sen_feat, (-1, feat.shape[-1]))
#     # insen_feat = th.masked_select(feat, th.tensor(insen_mask_resize, dtype=torch.bool))
#     insen_mask_ts = torch.unsqueeze(th.tensor(insen_mask, dtype=torch.bool), 1)
#     insen_feat = th.masked_select(feat, insen_mask_ts)
#     insen_feat = th.reshape(insen_feat, (-1, feat.shape[-1]))

#     # print(sen_feat.shape[0], insen_feat.shape[0])

#     ind_sen = feat_n_id[sen_mask]
#     ind_insen = feat_n_id[insen_mask]

#     # feat_index_sen = [[ind_sen[row] for col in range(embedding_entire.shape[1])]
#     #                   for row in range(sen_feat.shape[0])]
#     # feat_index_insen = [[ind_insen[row] for col in range(embedding_entire.shape[1])]
#     #                     for row in range(insen_feat.shape[0])]
#     feat_index_sen = np.array([[ind_sen[row]] * embedding_total.shape[1]
#                                for row in range(sen_feat.shape[0])])
#     feat_index_insen = np.array([[ind_insen[row]] * embedding_total.shape[1]
#                                  for row in range(insen_feat.shape[0])])

#     feat_index_sen = th.from_numpy(feat_index_sen)
#     feat_index_insen = th.from_numpy(feat_index_insen)

#     embedding_entire_tmp = embedding_total.scatter(0, feat_index_sen, sen_feat)
#     # embedding_entire_2 = embedding_entire_1.scatter_add(0, feat_index_insen, insen_feat)

#     feat_n_id = th.tensor(feat_n_id)
#     feat = embedding_entire_tmp.index_select(0, feat_n_id).to(device)

#     embedding_entire_tmp = embedding_entire_tmp.to(device)
#     return feat, embedding_entire_tmp


def store_embedding(embedding, feat, ind):
    device = feat.device
    ind = ind.squeeze().cpu().numpy().tolist()
    feat_t = feat.cpu()
    embedding = embedding.cpu()

    # feat_index = [ind[row] for col in range(embedding.shape[1])] for row in range(feat.shape[0])]
    feat_index = np.array([[ind[row]] * embedding.shape[1] for row in range(feat_t.shape[0])])
    feat_index = th.from_numpy(feat_index)
    embedding = embedding.scatter(0, feat_index, feat_t)

    return embedding.to(device)


def train_delta(model, device, data_loader, lr, weight_decay, v_sen=None, v_insen=None):
    # print('>> Start delta train')
    model.train()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    time_epoch = 0
    loss_list = []
    # batch_base = 0
    for step, batch in enumerate(data_loader):
        t_st = time.time()
        x, edge_index, y = batch.x.to(device), batch.edge_index.to(device), batch.y.to(device)
        n_id = batch.n_id
        # If label (y) has index
        optimizer.zero_grad()
        out = model(x, edge_index)

        # if y.shape[-1] > 1:
        # y, ind = torch.split(y, 1, dim=1)
        # ind = data_loader.input_nodes
        if v_sen is not None or v_insen is not None:
            out, embedding = feature_merge(model.embedding, out, n_id, v_sen, v_insen)
            # Update embedding
            model.embedding = torch.nn.Parameter(embedding)
        # model.embedding = store_embedding(model.embedding, out, ind)

        loss = F.cross_entropy(out, y.squeeze(1))
        # loss = F.nll_loss(out, y.squeeze(1))
        loss.backward()
        optimizer.step()
        time_epoch += (time.time() - t_st)
        loss_list.append(loss.item())
        # print(loss.item())
        # pbar.update(batch.batch_size)
        # batch_base += batch.batch_size

    # print('>> Finish delta train')
    return np.mean(loss_list), time_epoch


@torch.no_grad()
def validate_delta(model, device, loader, v_sen=None, v_insen=None):
    model.eval()
    y_pred, y_true = [], []
    for step, batch in enumerate(loader):
        x, edge_index, y = batch.x.to(device), batch.edge_index.to(device), batch.y
        out = model(x, edge_index)
        # y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())

        # if y.shape[-1] > 1:
        # y, ind = torch.split(y, 1, dim=1)
        # if v_sen is not None or v_insen is not None:
        #     out, embedding = combine_embedding(model.embedding, out, ind, v_sen, v_insen)
        #     # Update embedding
        #     model.embedding = torch.nn.Parameter(embedding)

        out = out.log_softmax(dim=-1)
        y_pred.append(torch.argmax(out, dim=-1, keepdim=True).cpu())
        y_true.append(y)
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    correct = torch.sum(y_pred == y_true)
    return correct.item() * 1.0 / len(y_true)


@torch.no_grad()
def test_delta(model, device, data_loader, checkpt_file, v_sen=None, v_insen=None):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    y_pred, y_true = [], []
    for step, batch in enumerate(data_loader):
        x, edge_index, y = batch.x.to(device), batch.edge_index.to(device), batch.y
        n_id = batch.n_id
        out = model(x, edge_index)

        # if y.shape[-1] > 1:
        # y, ind = torch.split(y, 1, dim=1)
        # ind = data_loader.input_nodes
        if v_sen is not None or v_insen is not None:
            out, embedding = feature_merge(model.embedding, out, n_id, v_sen, v_insen)
            model.embedding = torch.nn.Parameter(embedding)
        # model.embedding = store_embedding(model.embedding, out, ind)

        out = out.log_softmax(dim=-1)
        # y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_pred.append(torch.argmax(out, dim=-1, keepdim=True).cpu())
        y_true.append(y)
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    correct = torch.sum(y_pred == y_true)
    return correct.item() * 1.0 / len(y_true)
