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
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import time


class GCN(nn.Module):

    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()

        if n_layers > 1:
            self.layers.append(
                GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True))
            for i in range(1, n_layers - 1):
                self.layers.append(
                    GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True))
            self.layers.append(GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        else:
            self.layers.append(
                GraphConv(in_feats, n_classes, activation=activation, allow_zero_in_degree=True))

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


def train(args, model, device, lr, weight_decay):
    g = model.g
    features = g.ndata['feat'].to(device)
    train_mask = g.ndata['train_mask'].bool().to(device)
    val_mask = g.ndata['val_mask'].to(device)
    labels = g.ndata['label'].to(device)
    n_edges = g.number_of_edges()

    # print(train_mask, val_mask)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, val_mask, device)
        # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
        #                                     acc, n_edges / np.mean(dur) / 1000))


def evaluate(model, mask, device):
    g = model.g
    features = g.ndata['feat'].to(device)
    labels = g.ndata['label'].to(device)
    mask = mask.bool().to(device)  # Convert int8 to bool

    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def evaluate_delta(model, mask, device, updated_nodes):
    r"""
    Only update feature of updated nodes in inference
    """
    g = model.g
    features = g.ndata['feat'].to(device)
    labels = g.ndata['label'].to(device)
    mask = mask.bool().to(device)  # Convert int8 to bool

    model.eval()
    with torch.no_grad():
        logits_prev = model.logits
        logits = model(features)
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

        if n_layers > 1:
            self.layers.append(
                GraphConv_delta(in_feats,
                                n_hidden,
                                activation=activation,
                                allow_zero_in_degree=True))
            for i in range(1, n_layers - 1):
                self.layers.append(
                    GraphConv_delta(n_hidden,
                                    n_hidden,
                                    activation=activation,
                                    allow_zero_in_degree=True))
            self.layers.append(GraphConv_delta(n_hidden, n_classes, allow_zero_in_degree=True))
        else:
            self.layers.append(
                GraphConv_delta(in_feats,
                                n_classes,
                                activation=activation,
                                allow_zero_in_degree=True))

        self.dropout = nn.Dropout(p=dropout)

        # Record previous embedding
        self.embedding = th.Tensor([[0 for i in range(n_classes)]
                                    for j in range(g.number_of_nodes())]).requires_grad_(True)

    def forward(self, features, ngh_high_deg=None, ngh_low_deg=None):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h, ngh_high_deg, ngh_low_deg)

        if (ngh_high_deg is not None) or (ngh_low_deg is not None):
            # Combine delta-inferenced embedding and previous embedding
            h = self.combine_embedding(self.embedding, h, ngh_high_deg, ngh_low_deg)

        return h

    def combine_embedding(self, embedding_prev, feat, ngh_high_deg, ngh_low_deg):
        # Compulsorily execute in CPU (GPU not suits for scalar execution)
        device = feat.device
        feat = feat.to('cpu')
        embedding_prev = embedding_prev.to('cpu')

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
        feat_prev_keep_ind = list(set(feat_prev_ind) - set(ngh_high_deg))

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


def train_delta_edge_masked(args,
                            model,
                            features,
                            n_edges,
                            train_mask,
                            val_mask,
                            labels,
                            lr,
                            weight_decay,
                            ngh_high_deg=None,
                            ngh_low_deg=None):

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features, ngh_high_deg, ngh_low_deg)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate_delta_edge_masked(model, features, labels, val_mask, ngh_high_deg,
                                         ngh_low_deg)
        # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(), acc,
        #                                     n_edges / np.mean(dur) / 1000))

    # Update embedding
    model.embedding = torch.nn.Parameter(logits)


def evaluate_delta_edge_masked(model,
                               features,
                               labels,
                               mask,
                               nodes_high_deg=None,
                               nodes_low_deg=None):
    r"""
    Update feature of updated nodes according to node degree

    "model" should be GCN_delta
    """
    model.eval()
    with torch.no_grad():
        logits = model(features, nodes_high_deg, nodes_low_deg)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
