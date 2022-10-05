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
from .graphconv_delta import GraphConv_delta


class GCN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        # self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        self.layers.append(
            GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True))
        # hidden layers
        for i in range(n_layers - 1):
            # self.layers.append(
            #     GraphConv(n_hidden, n_hidden, activation=activation))
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True))
        # output layer
        # self.layers.append(
        #     GraphConv(n_hidden, n_classes))
        self.layers.append(GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(p=dropout)

        # Record previous logits
        self.logits = torch.zeros(g.number_of_nodes(), n_classes)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h


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
        # input layer
        self.layers.append(GraphConv_delta(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv_delta(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv_delta(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

        # # Record whether a node has been inferenced embedding
        # self.nodes_has_inferenced_mask = [0 for i in range(g.number_of_nodes())]

        # Record previous embedding
        # self.embedding = nn.Parameter(
        #     th.Tensor([[0 for i in range(n_classes)] for j in range(g.number_of_nodes())]))
        self.embedding = th.Tensor([[0 for i in range(n_classes)]
                                    for j in range(g.number_of_nodes())]).requires_grad_(True)

    def forward(self, features, ngh_high_deg=None, ngh_low_deg=None):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h, ngh_high_deg, ngh_low_deg)

        # for name, parameters in self.named_parameters():
        #     print(name, ':', parameters.size())
        #     print(parameters.detach().is_leaf)
        #     print(parameters.detach().grad)
        #     print(parameters.detach().grad_fn)

        if (ngh_high_deg is not None) or (ngh_low_deg is not None):
            # Combine delta-inferenced embedding and previous embedding
            h = self.combine_embedding(self.embedding, h, ngh_high_deg, ngh_low_deg)

        return h

    def combine_embedding(self, embedding_prev, feat, ngh_high_deg, ngh_low_deg):
        # Compulsorily execute in CPU (GPU not suits for scalar execution)
        device = feat.device
        feat = feat.to('cpu')
        embedding_prev = embedding_prev.to('cpu')

        # Combine delta rst with feat_prev
        feat_prev_ind = list(i for i in range(embedding_prev.shape[0]))
        feat_prev_keep_ind = list(set(feat_prev_ind) - set(ngh_high_deg) - set(ngh_low_deg))

        feat_prev_keep_ind = th.tensor(feat_prev_keep_ind, dtype=th.long)
        # ngh_high_deg_ind = th.tensor(ngh_high_deg, dtype=th.long)
        ngh_low_deg_ind = th.tensor(ngh_low_deg, dtype=th.long)

        feat_prev = th.index_select(embedding_prev, 0, feat_prev_keep_ind)
        # feat_high_deg = th.index_select(feat, 0, ngh_high_deg_ind)
        feat_low_deg = th.index_select(feat, 0, ngh_low_deg_ind)

        # Gen index for scatter
        index_feat_prev = [[feat_prev_keep_ind[row].item() for col in range(feat_prev.shape[1])]
                           for row in range(feat_prev_keep_ind.shape[0])]
        # index_high_deg = [[ngh_high_deg_ind[row].item() for col in range(feat_high_deg.shape[1])]
        #                   for row in range(ngh_high_deg_ind.shape[0])]
        index_low_deg = [[ngh_low_deg_ind[row].item() for col in range(feat_low_deg.shape[1])]
                         for row in range(ngh_low_deg_ind.shape[0])]

        index_feat_prev = th.tensor(index_feat_prev)
        # index_high_deg = th.tensor(index_high_deg)
        index_low_deg = th.tensor(index_low_deg)

        # Update feat of the nodes in the high and low degree
        feat.scatter(0, index_feat_prev, feat_prev)
        # embedding_prev.scatter(0, index_high_deg, feat_high_deg)
        feat.scatter(0, index_low_deg, feat_low_deg, reduce='add')

        # Transfer 'feat' to its previous device
        feat = feat.to(device)

        return feat
