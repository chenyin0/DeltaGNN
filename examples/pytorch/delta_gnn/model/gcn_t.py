"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch as th
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from torch_geometric.nn import GCNConv
from .graphconv_delta import GraphConv_delta
import torch.nn.functional as F
from torch.optim import Adam
# from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
# import torch_geometric.transforms as T
import numpy as np
import time
import util
from torch_sparse import SparseTensor
from tqdm import tqdm


class GCN(nn.Module):

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # if n_layers > 1:
        #     self.layers.append(GCNConv(in_feats, n_hidden))
        #     self.bns.append(nn.BatchNorm1d(n_hidden))
        #     for i in range(1, n_layers - 1):
        #         self.layers.append(GCNConv(n_hidden, n_hidden))
        #         self.bns.append(nn.BatchNorm1d(n_hidden))
        #     self.layers.append(GCNConv(n_hidden, n_classes))
        # else:
        #     self.layers.append(GCNConv(in_feats, n_classes))

        # self.dropout = dropout
        # # self.logits = torch.zeros(g.number_of_nodes(), n_classes)

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
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    # def forward(self, features, adj_t):
    #     h = features
    #     for i, layer in enumerate(self.layers[:-1]):
    #         h = layer(h, adj_t)
    #         h = self.bns[i](h)
    #         h = F.relu(h)
    #         h = F.dropout(h, p=self.dropout, training=self.training)
    #     h = self.layers[-1](h, adj_t)
    #     return h.log_softmax(dim=-1)

    def forward(self, features, edge_index):
        h = features
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layers[-1](h, edge_index)
        return h.log_softmax(dim=-1)


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
        loss = F.nll_loss(out, y.squeeze(1))
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


# def evaluate_delta(model, mask, device, updated_nodes):
#     r"""
#     Only update feature of updated nodes in inference
#     """
#     g = model.g
#     features = g.ndata['feat'].to(device)
#     labels = g.ndata['label'].to(device)
#     mask = mask.bool().to(device)  # Convert int8 to bool
#     adj = SparseTensor(row=g.all_edges()[0],
#                        col=g.all_edges()[1],
#                        sparse_sizes=(g.number_of_nodes(), g.number_of_nodes()))
#     adj_t = adj.to_symmetric().to(device)

#     model.eval()
#     with torch.no_grad():
#         logits_prev = model.logits
#         # logits = model(features)
#         logits = model(features, adj_t)
#         # logits = logits[mask]

#         logits_updated = logits
#         logits_updated[0:logits_prev.size()[0]] = logits_prev
#         for node_id in updated_nodes:
#             logits_updated[node_id] = logits[node_id]

#         model.logits = logits_updated  # Record updated logits

#         logits_updated = logits_updated[mask]
#         labels = labels[mask]
#         _, indices = torch.max(logits_updated, dim=1)
#         correct = torch.sum(indices == labels)
#         return correct.item() * 1.0 / len(labels)


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

        # if n_layers > 1:
        #     self.layers.append(
        #         GraphConv_delta(in_feats,
        #                         n_hidden,
        #                         activation=activation,
        #                         allow_zero_in_degree=True))
        #     for i in range(1, n_layers - 1):
        #         self.layers.append(
        #             GraphConv_delta(n_hidden,
        #                             n_hidden,
        #                             activation=activation,
        #                             allow_zero_in_degree=True))
        #     self.layers.append(GraphConv_delta(n_hidden, n_classes, allow_zero_in_degree=True))
        # else:
        #     self.layers.append(
        #         GraphConv_delta(in_feats,
        #                         n_classes,
        #                         activation=activation,
        #                         allow_zero_in_degree=True))

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

    # def forward(self, features, ngh_high_deg=None, ngh_low_deg=None, edge_mask=None):
    #     h = features
    #     for i, layer in enumerate(self.layers):
    #         if i != 0:
    #             h = self.dropout(h)
    #         h = layer(self.g, h, edge_mask)

    #     # if not self.training:
    #     if ngh_high_deg is not None or ngh_low_deg is not None:
    #         # Combine delta-inferenced embedding and previous embedding
    #         h = self.combine_embedding(self.embedding, h, ngh_high_deg, ngh_low_deg)

    #     return h

    # def forward(self, features, edge_index, v_sensitive=None, v_insensitive=None):
    #     h = features
    #     for i, layer in enumerate(self.layers[:-1]):
    #         h = layer(h, edge_index)
    #         h = self.bns[i](h)
    #         h = F.relu(h)
    #         h = F.dropout(h, p=self.dropout, training=self.training)
    #     h = self.layers[-1](h, edge_index)

    #     # if not self.training:
    #     if v_sensitive is not None or v_insensitive is not None:
    #         if not self.training:
    #             time_start = time.perf_counter()
    #         # Combine delta-inferenced embedding and previous embedding
    #         h = self.combine_embedding(self.embedding, h, v_sensitive, v_insensitive)
    #         # if not self.training:
    #         #     print('Time cost of embedding combine: {:.4f}'.format(time.perf_counter() -
    #         #                                                           time_start))

    #     return h.log_softmax(dim=-1), h

    def forward(self, features, edge_index):
        h = features
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h, edge_index)
            # h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layers[-1](h, edge_index)
        return h.log_softmax(dim=-1)

    # def forward(self, features, adj_t, v_sensitive, v_insensitive):
    #     h = features
    #     embedding = []
    #     for i, layer in enumerate(self.layers[:-1]):
    #         h = layer(h, adj_t)
    #         h = self.bns[i](h)
    #         h = F.relu(h)
    #         h = F.dropout(h, p=self.dropout, training=self.training)
    #         if v_sensitive is not None or v_insensitive is not None:
    #             # Combine delta-inferenced embedding and previous embedding
    #             h = self.combine_embedding(self.embedding[i], h, v_sensitive, v_insensitive)
    #         embedding.append(torch.nn.Parameter(h))
    #     h = self.layers[-1](h, adj_t)
    #     if v_sensitive is not None or v_insensitive is not None:
    #         # Combine delta-inferenced embedding and previous embedding
    #         h = self.combine_embedding(self.embedding[-1], h, v_sensitive, v_insensitive)
    #     embedding.append(torch.nn.Parameter(h))

    #     return h.log_softmax(dim=-1), embedding

    # def combine_embedding(self, embedding_prev, feat, v_sensitive, v_insensitive):
    #     # Compulsorily execute in CPU (GPU not suits for scalar execution)
    #     device = feat.device
    #     time_start = time.perf_counter()
    #     feat = feat.to('cpu')
    #     embedding_prev = embedding_prev.to('cpu')
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

    #     # Combine delta rst with feat_prev
    #     feat_prev_ind = list(i for i in range(embedding_prev.shape[0]))
    #     # feat_prev_keep_ind = list(set(feat_prev_ind) - set(ngh_high_deg) - set(ngh_low_deg))
    #     feat_prev_keep_ind = list(set(feat_prev_ind) - set(v_sensitive))

    #     feat_prev_keep_ind = th.tensor(feat_prev_keep_ind, dtype=th.long)
    #     # ngh_high_deg_ind = th.tensor(ngh_high_deg, dtype=th.long)
    #     ngh_low_deg_ind = th.tensor(v_insensitive, dtype=th.long)

    #     # time_prev_ind = time.perf_counter() - time_start - load_time
    #     # print('>> prev_ind: {}'.format(util.time_format(time_prev_ind)))

    #     feat_prev = th.index_select(embedding_prev, 0, feat_prev_keep_ind)
    #     # feat_high_deg = th.index_select(feat, 0, ngh_high_deg_ind)
    #     feat_low_deg = th.index_select(feat, 0, ngh_low_deg_ind)

    #     # time_prev_index_sel = time.perf_counter() - time_start - load_time - time_prev_ind
    #     # print('>> prev_index_sel: {}'.format(util.time_format(time_prev_index_sel)))

    #     # Gen index for scatter
    #     index_feat_prev = [[feat_prev_keep_ind[row].item() for col in range(feat_prev.shape[1])]
    #                        for row in range(feat_prev_keep_ind.shape[0])]
    #     # index_high_deg = [[ngh_high_deg_ind[row].item() for col in range(feat_high_deg.shape[1])]
    #     #                   for row in range(ngh_high_deg_ind.shape[0])]
    #     index_low_deg = [[ngh_low_deg_ind[row].item() for col in range(feat_low_deg.shape[1])]
    #                      for row in range(ngh_low_deg_ind.shape[0])]

    #     # time_feat_sel = time.perf_counter(
    #     # ) - time_start - load_time - time_prev_ind - time_prev_index_sel
    #     # print('>> feat_sel: {}'.format(util.time_format(time_feat_sel)))

    #     index_feat_prev = th.tensor(index_feat_prev)
    #     # index_high_deg = th.tensor(index_high_deg)
    #     index_low_deg = th.tensor(index_low_deg)

    #     # Update feat of the nodes in the high and low degree
    #     feat.scatter(0, index_feat_prev, feat_prev)
    #     # embedding_prev.scatter(0, index_high_deg, feat_high_deg)
    #     feat.scatter(0, index_low_deg, feat_low_deg, reduce='add')
    #     # feat.scatter_reduce(0, index_low_deg, feat_low_deg, reduce='sum')

    #     # time_scatter = time.perf_counter() - time_start - load_time - time_index_sel
    #     # print('>> scatter: {}'.format(util.time_format(time_scatter)))

    #     # Transfer 'feat' to its previous device
    #     feat = feat.to(device)

    #     # time_feat2gpu = time.perf_counter() - time_start - load_time - time_index_sel - time_scatter
    #     # print('>> feat2gpu: {}'.format(util.time_format(time_feat2gpu)))

    #     return feat

    # def combine_embedding(self, embedding_prev, feat, v_sensitive, v_insensitive):
    #     # Compulsorily execute in CPU (GPU not suits for scalar execution)
    #     device = feat.device
    #     time_start = time.perf_counter()
    #     feat = feat.to('cpu')
    #     embedding_prev = embedding_prev.to('cpu')
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

    #     # Combine delta rst with feat_prev
    #     feat_prev_ind = list(i for i in range(embedding_prev.shape[0]))
    #     # feat_prev_keep_ind = list(set(feat_prev_ind) - set(ngh_high_deg) - set(ngh_low_deg))
    #     # feat_prev_keep_ind = list(set(v_insensitive))
    #     feat_prev_keep_ind = list(set(feat_prev_ind) - set(v_sensitive))

    #     feat_prev_keep_ind = th.tensor(feat_prev_keep_ind, dtype=th.long)
    #     # ngh_high_deg_ind = th.tensor(ngh_high_deg, dtype=th.long)
    #     ngh_low_deg_ind = th.tensor(list(v_insensitive), dtype=th.long)

    #     # time_prev_ind = time.perf_counter() - time_start - load_time
    #     # print('>> prev_ind: {}'.format(util.time_format(time_prev_ind)))

    #     feat_prev = th.index_select(embedding_prev, 0, feat_prev_keep_ind)
    #     # feat_high_deg = th.index_select(feat, 0, ngh_high_deg_ind)
    #     feat_low_deg = th.index_select(feat, 0, ngh_low_deg_ind)

    #     # time_prev_index_sel = time.perf_counter() - time_start - load_time - time_prev_ind
    #     # print('>> prev_index_sel: {}'.format(util.time_format(time_prev_index_sel)))

    #     # Gen index for scatter
    #     index_feat_prev = [[feat_prev_keep_ind[row].item() for col in range(feat_prev.shape[1])]
    #                        for row in range(feat_prev_keep_ind.shape[0])]
    #     # index_high_deg = [[ngh_high_deg_ind[row].item() for col in range(feat_high_deg.shape[1])]
    #     #                   for row in range(ngh_high_deg_ind.shape[0])]
    #     index_low_deg = [[ngh_low_deg_ind[row].item() for col in range(feat_low_deg.shape[1])]
    #                      for row in range(ngh_low_deg_ind.shape[0])]

    #     # time_feat_sel = time.perf_counter(
    #     # ) - time_start - load_time - time_prev_ind - time_prev_index_sel
    #     # print('>> feat_sel: {}'.format(util.time_format(time_feat_sel)))

    #     index_feat_prev = th.tensor(index_feat_prev)
    #     # index_high_deg = th.tensor(index_high_deg)
    #     index_low_deg = th.tensor(index_low_deg)

    #     # Update feat of the nodes in the high and low degree
    #     feat.scatter(0, index_feat_prev, feat_prev)
    #     # embedding_prev.scatter(0, index_high_deg, feat_high_deg)
    #     feat.scatter(0, index_low_deg, feat_low_deg, reduce='add')
    #     # feat.scatter_reduce(0, index_low_deg, feat_low_deg, reduce='sum')

    #     # time_scatter = time.perf_counter() - time_start - load_time - time_index_sel
    #     # print('>> scatter: {}'.format(util.time_format(time_scatter)))

    #     # Transfer 'feat' to its previous device
    #     feat = feat.to(device)

    #     # time_feat2gpu = time.perf_counter() - time_start - load_time - time_index_sel - time_scatter
    #     # print('>> feat2gpu: {}'.format(util.time_format(time_feat2gpu)))

    #     return feat


def combine_embedding(embedding_entire, feat, ind, v_sen, v_insen):
    """
    With batch-wise combination 

    ind: the feature index in current batch (index is belongs to the original feature)
    """
    # Compulsorily execute in CPU (GPU not suits for scalar execution)
    device = feat.device
    time_start = time.perf_counter()
    feat = feat.to('cpu')
    embedding_entire = embedding_entire.to('cpu')
    # load_time = time.perf_counter() - time_start
    # print('>> Load feat: {}'.format(util.time_format(load_time)))

    ##
    r"""
    Para:
    1. feat_prev: features in the last time
    2. feat: updated features under edge_mask (all-neighbor updating for high degree, and delta-neighbor updating for low degree)
    3. feat_low_deg: updated features with low degree

    Method:
    1. First, replace items in "feat" to which in "feat_prev" with the corresponding node_id
    2. Then, merge "feat_low_deg" with "add" operation to "feat"
    """
    ind = ind.cpu().squeeze().numpy()
    v_sen = np.array(list(v_sen))
    v_insen = np.array(list(v_insen))
    # sen_mask = np.isin(ind, v_sen)
    insen_mask = np.isin(ind, v_insen)
    sen_mask = ~insen_mask
    # sen_mask = [True for i in range(ind.shape[0])]

    # # sen_mask_resize = [[i for col in range(feat.shape[-1])] for i in sen_mask]
    # sen_mask_resize = [[i] * feat.shape[-1] for i in sen_mask]
    # # print(sen_mask_resize)
    # # insen_mask_resize = [[i for col in range(feat.shape[-1])] for i in insen_mask]
    # insen_mask_resize = [[i] * feat.shape[-1] for i in insen_mask]

    # sen_feat = th.masked_select(feat, th.tensor(sen_mask_resize, dtype=torch.bool))
    sen_mask_ts = torch.unsqueeze(th.tensor(sen_mask, dtype=torch.bool), 1)
    sen_feat = th.masked_select(feat, sen_mask_ts)
    sen_feat = th.reshape(sen_feat, (-1, feat.shape[-1]))
    # insen_feat = th.masked_select(feat, th.tensor(insen_mask_resize, dtype=torch.bool))
    insen_mask_ts = torch.unsqueeze(th.tensor(insen_mask, dtype=torch.bool), 1)
    insen_feat = th.masked_select(feat, insen_mask_ts)
    insen_feat = th.reshape(insen_feat, (-1, feat.shape[-1]))

    # print(sen_feat.shape[0], insen_feat.shape[0])

    ind_sen = ind[sen_mask]
    ind_insen = ind[insen_mask]

    # feat_index_sen = [[ind_sen[row] for col in range(embedding_entire.shape[1])]
    #                   for row in range(sen_feat.shape[0])]
    # feat_index_insen = [[ind_insen[row] for col in range(embedding_entire.shape[1])]
    #                     for row in range(insen_feat.shape[0])]
    feat_index_sen = [[ind_sen[row]] * embedding_entire.shape[1]
                      for row in range(sen_feat.shape[0])]
    feat_index_insen = [[ind_insen[row]] * embedding_entire.shape[1]
                        for row in range(insen_feat.shape[0])]

    feat_index_sen = th.tensor(feat_index_sen)
    feat_index_insen = th.tensor(feat_index_insen)

    embedding_entire_tmp = embedding_entire.scatter(0, feat_index_sen, sen_feat)
    # embedding_entire_2 = embedding_entire_1.scatter_add(0, feat_index_insen, insen_feat)

    ind = th.tensor(ind)
    feat = embedding_entire_tmp.index_select(0, ind).to(device)

    embedding_entire_tmp = embedding_entire_tmp.to(device)
    return feat, embedding_entire_tmp


def store_embedding(embedding, feat, ind):
    device = feat.device
    ind = ind.squeeze().cpu().numpy().tolist()
    feat = feat.cpu()
    embedding = embedding.cpu()

    feat_index = [[ind[row] for col in range(embedding.shape[1])] for row in range(feat.shape[0])]
    feat_index = th.tensor(feat_index)
    embedding = embedding.scatter(0, feat_index, feat)

    return embedding.to(device)


def train_delta(model, device, train_loader, lr, weight_decay, v_sen=None, v_insen=None):
    # print('>> Start training')
    model.train()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    time_epoch = 0
    loss_list = []
    batch_base = 0
    for step, batch in enumerate(train_loader):
        t_st = time.time()
        x, edge_index, y = batch.x.to(device), batch.edge_index.to(device), batch.y.to(device)
        # If label (y) has index
        optimizer.zero_grad()
        out = model(x, edge_index)
        if y.shape[-1] > 1:
            y, ind = torch.split(y, 1, dim=1)
            if v_sen is not None or v_insen is not None:
                out, embedding = combine_embedding(model.embedding, out, ind, v_sen, v_insen)
                # Update embedding
                # model.embedding = torch.nn.Parameter(embedding)
                # model.embedding = store_embedding(model.embedding, out, ind)
        # loss = F.cross_entropy(out, y.squeeze(1))
        loss = F.nll_loss(out, y.squeeze(1))
        loss.backward()
        optimizer.step()
        time_epoch += (time.time() - t_st)
        loss_list.append(loss.item())
        # print(loss.item())
        # pbar.update(batch.batch_size)
        batch_base += batch.batch_size

    return np.mean(loss_list), time_epoch


@torch.no_grad()
def validate_delta(model, device, loader, v_sen=None, v_insen=None):
    model.eval()
    y_pred, y_true = [], []
    for step, batch in enumerate(loader):
        x, edge_index, y = batch.x.to(device), batch.edge_index.to(device), batch.y
        out = model(x, edge_index)
        # y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        if y.shape[-1] > 1:
            y, ind = torch.split(y, 1, dim=1)
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
def test_delta(model, device, loader, checkpt_file, v_sen=None, v_insen=None):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    y_pred, y_true = [], []
    for step, batch in enumerate(loader):
        x, edge_index, y = batch.x.to(device), batch.edge_index.to(device), batch.y
        out = model(x, edge_index)
        if y.shape[-1] > 1:
            y, ind = torch.split(y, 1, dim=1)
            if v_sen is not None or v_insen is not None:
                out, _ = combine_embedding(model.embedding, out, ind, v_sen, v_insen)
        out = out.log_softmax(dim=-1)
        # y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_pred.append(torch.argmax(out, dim=-1, keepdim=True).cpu())
        y_true.append(y)
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    correct = torch.sum(y_pred == y_true)
    return correct.item() * 1.0 / len(y_true)


# def train_delta_edge_masked(args,
#                             model,
#                             device,
#                             lr,
#                             weight_decay,
#                             v_sensitive=None,
#                             v_insensitive=None):
#     g = model.g
#     features = g.ndata['feat']
#     train_mask = g.ndata['train_mask'].bool()
#     val_mask = g.ndata['val_mask']
#     labels = g.ndata['label']
#     n_edges = g.number_of_edges()
#     edge_mask = g.edata['edge_mask']

#     optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     # initialize graph
#     loss_log = []
#     dur = []
#     for epoch in range(args.n_epochs):
#         model.train()
#         if epoch >= 3:
#             t0 = time.time()
#         # forward
#         logits = model(features, v_sensitive, v_insensitive, edge_mask)
#         loss = F.cross_entropy(logits[train_mask], labels[train_mask])

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # # Update embedding
#         # model.embedding = torch.nn.Parameter(logits)

#         # if epoch >= 3:
#         #     dur.append(time.time() - t0)
#         # acc = evaluate_delta_edge_masked(model, val_mask, device, ngh_high_deg, ngh_low_deg)
#         # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
#         #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(), acc,
#         #                                     n_edges / np.mean(dur) / 1000))

#         loss_log.append(round(loss.item(), 2))
#     np.savetxt('../../../results/loss/' + args.dataset + '_evo_delta_loss' + '.txt',
#                loss_log,
#                fmt='%.2f')

#     # Update embedding
#     model.embedding = torch.nn.Parameter(logits)

# def train_delta_update(args,
#                        model,
#                        device,
#                        lr,
#                        weight_decay,
#                        graph_adj_t=None,
#                        v_sensitive=None,
#                        v_insensitive=None):
#     print('>> Start training')
#     time_start = time.perf_counter()
#     g = model.g
#     features = g.ndata['feat']
#     train_mask = g.ndata['train_mask'].bool()
#     val_mask = g.ndata['val_mask']
#     labels = g.ndata['label']
#     n_edges = g.number_of_edges()
#     # edge_mask = g.edata['edge_mask']
#     if graph_adj_t == None:
#         adj = SparseTensor(row=g.all_edges()[0],
#                            col=g.all_edges()[1],
#                            sparse_sizes=(g.number_of_nodes(), g.number_of_nodes()))
#         adj_t = adj.to_symmetric().to(device)
#     else:
#         adj_t = graph_adj_t.to(device)

#     optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     # initialize graph
#     loss_log = []
#     dur = []

#     for epoch in range(args.n_epochs):
#         model.train()
#         if epoch >= 3:
#             t0 = time.time()
#         # forward
#         logits, feat = model(features, adj_t, v_sensitive, v_insensitive)
#         loss = F.cross_entropy(logits[train_mask], labels[train_mask])

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if epoch >= 3:
#             dur.append(time.time() - t0)
#         # acc = evaluate_delta_edge_masked(model, val_mask, device, ngh_high_deg, ngh_low_deg)
#         # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
#         #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(), acc,
#         #                                     n_edges / np.mean(dur) / 1000))

#         loss_log.append(round(loss.item(), 2))
#     np.savetxt('../../../results/loss/' + args.dataset + '_evo_delta_loss' + '.txt',
#                loss_log,
#                fmt='%.2f')

#     acc = evaluate_delta_update(model, val_mask, device, adj_t, v_sensitive, v_insensitive)
#     print("Training time(s) {:.4f} | Accuracy {:.4f}".format(time.perf_counter() - time_start, acc))

#     # Update embedding
#     model.embedding = [torch.nn.Parameter(i) for i in feat]

# def train_delta_update(args,
#                        model,
#                        device,
#                        lr,
#                        weight_decay,
#                        graph_adj_t=None,
#                        v_sensitive=None,
#                        v_insensitive=None):
#     print('>> Start training')
#     time_start = time.perf_counter()
#     g = model.g
#     features = g.ndata['feat']
#     train_mask = g.ndata['train_mask'].bool()
#     val_mask = g.ndata['val_mask']
#     labels = g.ndata['label']
#     n_edges = g.number_of_edges()
#     # edge_mask = g.edata['edge_mask']
#     if graph_adj_t == None:
#         adj = SparseTensor(row=g.all_edges()[0],
#                            col=g.all_edges()[1],
#                            sparse_sizes=(g.number_of_nodes(), g.number_of_nodes()))
#         adj_t = adj.to_symmetric().to(device)
#     else:
#         adj_t = graph_adj_t.to(device)

#     optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     # initialize graph
#     loss_log = []
#     dur = []

#     for epoch in range(args.n_epochs):
#         model.train()
#         if epoch >= 3:
#             t0 = time.time()
#         # forward
#         logits, feat = model(features, adj_t, v_sensitive, v_insensitive)
#         loss = F.cross_entropy(logits[train_mask], labels[train_mask])

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if epoch >= 3:
#             dur.append(time.time() - t0)
#         # acc = evaluate_delta_edge_masked(model, val_mask, device, ngh_high_deg, ngh_low_deg)
#         # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
#         #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(), acc,
#         #                                     n_edges / np.mean(dur) / 1000))

#         loss_log.append(round(loss.item(), 2))
#     np.savetxt('../../../results/loss/' + args.dataset + '_evo_delta_loss' + '.txt',
#                loss_log,
#                fmt='%.2f')

#     acc = evaluate_delta_update(model, val_mask, device, adj_t, v_sensitive, v_insensitive)
#     print("Training time(s) {:.4f} | Accuracy {:.4f}".format(time.perf_counter() - time_start, acc))

#     # Update embedding
#     model.embedding = torch.nn.Parameter(feat)

# def evaluate_delta_update(model, mask, device, v_sensitive=None, v_insensitive=None):
#     r"""
#     Update feature of updated nodes according to node degree
#     "model" should be GCN_delta
#     """

#     g = model.g
#     features = g.ndata['feat']
#     labels = g.ndata['label']
#     mask = mask.bool()  # Convert int8 to bool
#     edge_mask = g.edata['edge_mask']

#     model.eval()
#     with torch.no_grad():
#         logits = model(features, v_sensitive, v_insensitive, edge_mask)
#         # logits = model.embedding
#         logits = logits[mask]
#         labels = labels[mask]
#         _, indices = torch.max(logits, dim=1)
#         correct = torch.sum(indices == labels)
#         # Update embedding
#         model.embedding = torch.nn.Parameter(logits)
#         return correct.item() * 1.0 / len(labels)

# def evaluate_delta_update(model,
#                           mask,
#                           device,
#                           graph_adj_t=None,
#                           v_sensitive=None,
#                           v_insensitive=None):
#     r"""
#     Update feature of updated nodes according to node degree
#     "model" should be GCN_delta
#     """
#     print('>> Start eval')
#     time_start = time.perf_counter()
#     g = model.g
#     features = g.ndata['feat']
#     labels = g.ndata['label']
#     mask = mask.bool()  # Convert int8 to bool
#     if graph_adj_t == None:
#         adj = SparseTensor(row=g.all_edges()[0],
#                            col=g.all_edges()[1],
#                            sparse_sizes=(g.number_of_nodes(), g.number_of_nodes()))
#         adj_t = adj.to_symmetric().to(device)
#     else:
#         adj_t = graph_adj_t.to(device)

#     model.eval()
#     with torch.no_grad():
#         logits, feat = model(features, adj_t, v_sensitive, v_insensitive)
#         # logits = model.embedding
#         logits = logits[mask]
#         labels = labels[mask]
#         _, indices = torch.max(logits, dim=1)
#         correct = torch.sum(indices == labels)
#         # # Update embedding
#         # model.embedding = [torch.nn.Parameter(i) for i in feat]
#         print("Evaluate time(s) {:.4f}".format(time.perf_counter() - time_start))
#         return correct.item() * 1.0 / len(labels)

# def evaluate_delta_update(model,
#                           mask,
#                           device,
#                           graph_adj_t=None,
#                           v_sensitive=None,
#                           v_insensitive=None):
#     r"""
#     Update feature of updated nodes according to node degree
#     "model" should be GCN_delta
#     """
#     print('>> Start eval')
#     time_start = time.perf_counter()
#     g = model.g
#     features = g.ndata['feat']
#     labels = g.ndata['label']
#     mask = mask.bool()  # Convert int8 to bool
#     if graph_adj_t == None:
#         adj = SparseTensor(row=g.all_edges()[0],
#                            col=g.all_edges()[1],
#                            sparse_sizes=(g.number_of_nodes(), g.number_of_nodes()))
#         adj_t = adj.to_symmetric().to(device)
#     else:
#         adj_t = graph_adj_t.to(device)

#     model.eval()
#     with torch.no_grad():
#         logits, feat = model(features, adj_t, v_sensitive, v_insensitive)
#         # logits = model.embedding
#         logits = logits[mask]
#         labels = labels[mask]
#         _, indices = torch.max(logits, dim=1)
#         correct = torch.sum(indices == labels)
#         # # Update embedding
#         # model.embedding = torch.nn.Parameter(feat)
#         print("Evaluate time(s) {:.4f}".format(time.perf_counter() - time_start))
#         return correct.item() * 1.0 / len(labels)
