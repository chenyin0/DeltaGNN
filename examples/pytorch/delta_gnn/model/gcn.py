"""GCN using DGL nn package

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
import util
from torch_sparse import SparseTensor
# from util import gen_graph_adj_t


class GCN(nn.Module):

    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(GCN, self).__init__()
        self.g = g
        # adj = SparseTensor(row=self.g.all_edges()[0],
        #                    col=self.g.all_edges()[1],
        #                    sparse_sizes=(self.g.number_of_nodes(), self.g.number_of_nodes()))
        # self.adj_t = adj.to_symmetric()
        self.layers = nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # if n_layers > 1:
        #     self.layers.append(
        #         GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True))
        #     for i in range(1, n_layers - 1):
        #         self.layers.append(
        #             GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True))
        #     self.layers.append(GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        # else:
        #     self.layers.append(
        #         GraphConv(in_feats, n_classes, activation=activation, allow_zero_in_degree=True))

        # if n_layers > 1:
        #     self.layers.append(GraphConv(in_feats, n_hidden, activation=None))
        #     self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        #     for i in range(1, n_layers - 1):
        #         self.layers.append(GraphConv(n_hidden, n_hidden, activation=None))
        #         self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        #     self.layers.append(GraphConv(n_hidden, n_classes))
        # else:
        #     self.layers.append(GraphConv(in_feats, n_classes, activation=None))

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

        # Record previous logits
        self.logits = torch.zeros(g.number_of_nodes(), n_classes)

    # def forward(self, features):
    #     h = features
    #     for i, layer in enumerate(self.layers):
    #         if i != 0:
    #             h = self.dropout(h)
    #         h = layer(self.g, h)
    #     return h

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, features, adj_t):
        h = features
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h, adj_t)
            h = self.bns[i](h)
            h = F.relu(h)

            sign_a = torch.sign(h).int()
            non_zero_a = torch.count_nonzero(sign_a, dim=1).reshape(1, -1).squeeze()
            # print(non_zero_a.tolist())
            non_zero_sum = sum(non_zero_a.tolist())
            h_size = h.numel()
            zero_num = h_size - non_zero_sum
            print(zero_num, h_size, round(zero_num/h_size, 3))
            
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layers[-1](h, adj_t)
        return h.log_softmax(dim=-1)


def train(args, model, device, lr, weight_decay):
    g = model.g
    features = g.ndata['feat']
    train_mask = g.ndata['train_mask'].bool()
    val_mask = g.ndata['val_mask'].bool()
    labels = g.ndata['label']
    n_edges = g.number_of_edges()
    adj = SparseTensor(row=g.all_edges()[0],
                       col=g.all_edges()[1],
                       sparse_sizes=(g.number_of_nodes(), g.number_of_nodes()))
    adj_t = adj.to_symmetric().to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # initialize graph
    loss_log = []
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features, adj_t)
        # logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        loss_log.append(round(loss.item(), 2))

        # acc = evaluate(model, val_mask, device)
        # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(), acc,
        #                                     n_edges / np.mean(dur) / 1000))

    # np.savetxt('../../../results/loss/' + args.dataset + '_evo_loss' + '.txt', loss_log, fmt='%.2f')
    # np.savetxt('./results/loss/' + args.dataset + '_evo_loss' + '.txt', loss_log, fmt='%.2f')

    acc = evaluate(model, val_mask, device)
    print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
          "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(), acc,
                                        n_edges / np.mean(dur) / 1000))


def evaluate(model, mask, device):
    g = model.g
    features = g.ndata['feat'].to(device)
    labels = g.ndata['label'].to(device)
    mask = mask.bool().to(device)  # Convert int8 to bool
    # adj_t = model.adj_t.to(device)
    adj = SparseTensor(row=g.all_edges()[0],
                       col=g.all_edges()[1],
                       sparse_sizes=(g.number_of_nodes(), g.number_of_nodes()))
    adj_t = adj.to_symmetric().to(device)

    model.eval()
    with torch.no_grad():
        # logits = model(features)
        logits = model(features, adj_t)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

    # evaluator = Evaluator(name='ogbn-arxiv')
    # with torch.no_grad():
    #     # logits = model(features, adj_t)
    #     logits = model(features, adj_t)
    #     y_pred = logits.argmax(dim=-1, keepdim=True)
    #     # labels = labels[mask]
    #     labels = labels.reshape(labels.shape[0], 1)

    #     acc = evaluator.eval({
    #         'y_true': labels[mask],
    #         'y_pred': y_pred[mask],
    #     })['acc']

    #     return acc


def evaluate_delta(model, mask, device, updated_nodes):
    r"""
    Only update feature of updated nodes in inference
    """
    g = model.g
    features = g.ndata['feat'].to(device)
    labels = g.ndata['label'].to(device)
    mask = mask.bool().to(device)  # Convert int8 to bool
    adj = SparseTensor(row=g.all_edges()[0],
                       col=g.all_edges()[1],
                       sparse_sizes=(g.number_of_nodes(), g.number_of_nodes()))
    adj_t = adj.to_symmetric().to(device)

    model.eval()
    with torch.no_grad():
        logits_prev = model.logits
        # logits = model(features)
        logits = model(features, adj_t)
        # logits = logits[mask]

        logits_updated = logits
        logits_updated[0:logits_prev.size()[0]] = logits_prev
        for node_id in updated_nodes:
            logits_updated[node_id] = logits[node_id]

        model.logits = logits_updated  # Record updated logits

        logits_updated = logits_updated[mask]
        labels = labels[mask]
        _, indices = torch.max(logits_updated, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


class GCN_delta(nn.Module):
    r"""
    Initial para:
        g: subgraph of original graph
        in_feats: feature matrix of current subgraph
    """

    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(GCN_delta, self).__init__()
        self.g = g
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
        # self.embedding = th.Tensor([[0 for i in range(n_classes)]
        #                             for j in range(g.number_of_nodes())]).requires_grad_(True)
        self.embedding = []
        for i in range(n_layers - 1):
            self.embedding.append(
                th.Tensor([[0 for i in range(n_hidden)]
                           for j in range(g.number_of_nodes())]).requires_grad_(True))
        self.embedding.append(
            th.Tensor([[0 for i in range(n_classes)]
                       for j in range(g.number_of_nodes())]).requires_grad_(True))

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

    def forward(self, features, adj_t, v_sensitive, v_insensitive):
        h = features
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h, adj_t)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layers[-1](h, adj_t)

        # if not self.training:
        if v_sensitive is not None or v_insensitive is not None:
            if not self.training:
                time_start = time.perf_counter()
            # Combine delta-inferenced embedding and previous embedding
            h = self.combine_embedding(self.embedding, h, v_sensitive, v_insensitive)
            if not self.training:
                print('Time cost of embedding combine: {:.4f}'.format(time.perf_counter() -
                                                                      time_start))

        return h.log_softmax(dim=-1), h

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

    def combine_embedding(self, embedding_prev, feat, v_sensitive, v_insensitive):
        # Compulsorily execute in CPU (GPU not suits for scalar execution)
        device = feat.device
        time_start = time.perf_counter()
        feat = feat.to('cpu')
        embedding_prev = embedding_prev.to('cpu')
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

        # Combine delta rst with feat_prev
        feat_prev_ind = list(i for i in range(embedding_prev.shape[0]))
        # feat_prev_keep_ind = list(set(feat_prev_ind) - set(ngh_high_deg) - set(ngh_low_deg))
        # feat_prev_keep_ind = list(set(v_insensitive))
        feat_prev_keep_ind = list(set(feat_prev_ind) - set(v_sensitive))

        feat_prev_keep_ind = th.tensor(feat_prev_keep_ind, dtype=th.long)
        # ngh_high_deg_ind = th.tensor(ngh_high_deg, dtype=th.long)
        ngh_low_deg_ind = th.tensor(v_insensitive, dtype=th.long)

        # time_prev_ind = time.perf_counter() - time_start - load_time
        # print('>> prev_ind: {}'.format(util.time_format(time_prev_ind)))

        feat_prev = th.index_select(embedding_prev, 0, feat_prev_keep_ind)
        # feat_high_deg = th.index_select(feat, 0, ngh_high_deg_ind)
        feat_low_deg = th.index_select(feat, 0, ngh_low_deg_ind)

        # time_prev_index_sel = time.perf_counter() - time_start - load_time - time_prev_ind
        # print('>> prev_index_sel: {}'.format(util.time_format(time_prev_index_sel)))

        # Gen index for scatter
        index_feat_prev = [[feat_prev_keep_ind[row].item() for col in range(feat_prev.shape[1])]
                           for row in range(feat_prev_keep_ind.shape[0])]
        # index_high_deg = [[ngh_high_deg_ind[row].item() for col in range(feat_high_deg.shape[1])]
        #                   for row in range(ngh_high_deg_ind.shape[0])]
        index_low_deg = [[ngh_low_deg_ind[row].item() for col in range(feat_low_deg.shape[1])]
                         for row in range(ngh_low_deg_ind.shape[0])]

        # time_feat_sel = time.perf_counter(
        # ) - time_start - load_time - time_prev_ind - time_prev_index_sel
        # print('>> feat_sel: {}'.format(util.time_format(time_feat_sel)))

        index_feat_prev = th.tensor(index_feat_prev)
        # index_high_deg = th.tensor(index_high_deg)
        index_low_deg = th.tensor(index_low_deg)

        # Update feat of the nodes in the high and low degree
        feat.scatter(0, index_feat_prev, feat_prev)
        # embedding_prev.scatter(0, index_high_deg, feat_high_deg)
        feat.scatter(0, index_low_deg, feat_low_deg, reduce='add')
        # feat.scatter_reduce(0, index_low_deg, feat_low_deg, reduce='sum')

        # time_scatter = time.perf_counter() - time_start - load_time - time_index_sel
        # print('>> scatter: {}'.format(util.time_format(time_scatter)))

        # Transfer 'feat' to its previous device
        feat = feat.to(device)

        # time_feat2gpu = time.perf_counter() - time_start - load_time - time_index_sel - time_scatter
        # print('>> feat2gpu: {}'.format(util.time_format(time_feat2gpu)))

        return feat


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


def train_delta_update(args,
                       model,
                       device,
                       lr,
                       weight_decay,
                       graph_adj_t=None,
                       v_sensitive=None,
                       v_insensitive=None):
    print('>> Start training')
    time_start = time.perf_counter()
    g = model.g
    features = g.ndata['feat']
    train_mask = g.ndata['train_mask'].bool()
    val_mask = g.ndata['val_mask']
    labels = g.ndata['label']
    n_edges = g.number_of_edges()
    # edge_mask = g.edata['edge_mask']
    if graph_adj_t == None:
        adj = SparseTensor(row=g.all_edges()[0],
                           col=g.all_edges()[1],
                           sparse_sizes=(g.number_of_nodes(), g.number_of_nodes()))
        adj_t = adj.to_symmetric().to(device)
    else:
        adj_t = graph_adj_t.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # initialize graph
    loss_log = []
    dur = []

    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits, feat = model(features, adj_t, v_sensitive, v_insensitive)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)
        # acc = evaluate_delta_edge_masked(model, val_mask, device, ngh_high_deg, ngh_low_deg)
        # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(), acc,
        #                                     n_edges / np.mean(dur) / 1000))

        loss_log.append(round(loss.item(), 2))
    np.savetxt('../../../results/loss/' + args.dataset + '_evo_delta_loss' + '.txt',
               loss_log,
               fmt='%.2f')

    acc = evaluate_delta_update(model, val_mask, device, adj_t, v_sensitive, v_insensitive)
    print("Training time(s) {:.4f} | Accuracy {:.4f}".format(time.perf_counter() - time_start, acc))

    # Update embedding
    model.embedding = torch.nn.Parameter(feat)


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


def evaluate_delta_update(model,
                          mask,
                          device,
                          graph_adj_t=None,
                          v_sensitive=None,
                          v_insensitive=None):
    r"""
    Update feature of updated nodes according to node degree
    "model" should be GCN_delta
    """
    print('>> Start eval')
    time_start = time.perf_counter()
    g = model.g
    features = g.ndata['feat']
    labels = g.ndata['label']
    mask = mask.bool()  # Convert int8 to bool
    if graph_adj_t == None:
        adj = SparseTensor(row=g.all_edges()[0],
                           col=g.all_edges()[1],
                           sparse_sizes=(g.number_of_nodes(), g.number_of_nodes()))
        adj_t = adj.to_symmetric().to(device)
    else:
        adj_t = graph_adj_t.to(device)

    model.eval()
    with torch.no_grad():
        logits, feat = model(features, adj_t, v_sensitive, v_insensitive)
        # logits = model.embedding
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        # # Update embedding
        # model.embedding = [torch.nn.Parameter(i) for i in feat]
        print("Evaluate time(s) {:.4f}".format(time.perf_counter() - time_start))
        return correct.item() * 1.0 / len(labels)


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
