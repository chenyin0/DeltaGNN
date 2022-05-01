"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from graphconv_block import GraphConv_block


class GCN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        # self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        self.layers.append(
            GraphConv(in_feats,
                      n_hidden,
                      activation=activation,
                      allow_zero_in_degree=True))
        # hidden layers
        for i in range(n_layers - 1):
            # self.layers.append(
            #     GraphConv(n_hidden, n_hidden, activation=activation))
            self.layers.append(
                GraphConv(n_hidden,
                          n_hidden,
                          activation=activation,
                          allow_zero_in_degree=True))
        # output layer
        # self.layers.append(
        #     GraphConv(n_hidden, n_classes))
        self.layers.append(
            GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        h1 = h
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h


class GCN_evolve(nn.Module):
    """
        Initial para:
            g: subgraph of original graph
            in_feats: feature matrix of current subgraph
    """
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv_block(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv_block(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv_block(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        h1 = h
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)

            # h1 = torch.matmul(h1, layer.weight)
            # print(h.equal(h1))
        return h
