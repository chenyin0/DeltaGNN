import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
import time

from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GINConv
from .ginconv_delta import GINConv_delta
from dgl.nn.pytorch.glob import SumPooling, AvgPooling


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GIN(nn.Module):

    def __init__(self, g, input_dim, hidden_dim, output_dim, n_layers, activation, dropout):
        super().__init__()
        self.g = g
        self.ginlayers = nn.ModuleList()
        # self.batch_norms = nn.ModuleList()
        if n_layers > 1:
            self.ginlayers.append(GINConv(Linear(input_dim, hidden_dim), activation=activation))
            # self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            for i in range(1, n_layers - 1):
                self.ginlayers.append(GINConv(Linear(hidden_dim, hidden_dim),
                                              activation=activation))
                # self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.ginlayers.append(GINConv(Linear(hidden_dim, output_dim), activation=activation))
            # self.batch_norms.append(nn.BatchNorm1d(output_dim))
        else:
            self.ginlayers.append(GINConv(Linear(input_dim, output_dim), activation=activation))
            # self.batch_norms.append(nn.BatchNorm1d(output_dim))

    def forward(self, h):
        # list of hidden representation at each layer (including the input layer)
        # hidden_rep = [h]
        g = self.g
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            # h = self.batch_norms[i](h)
            # h = F.relu(h)
        #     hidden_rep.append(h)
        # score_over_layer = 0
        # # perform graph sum pooling over all nodes in each layer
        # for i, h in enumerate(hidden_rep):
        #     pooled_h = self.pool(g, h)
        #     score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        # return score_over_layer
        h = F.log_softmax(h, dim=-1)
        return h


def train(args, model, device, lr, weight_decay):
    g = model.g
    features = g.ndata['feat']
    train_mask = g.ndata['train_mask'].bool()
    val_mask = g.ndata['val_mask']
    labels = g.ndata['label']
    n_edges = g.number_of_edges()

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # initialize graph
    loss_log = []
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        loss_log.append(round(loss.item(), 2))

    np.savetxt('../../../results/loss/' + args.dataset + '_evo_loss' + '.txt', loss_log, fmt='%.2f')
    # np.savetxt('./results/loss/' + args.dataset + '_evo_loss' + '.txt', loss_log, fmt='%.2f')


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


class GIN_delta(nn.Module):
    r"""
    Initial para:
        g: subgraph of original graph
    """

    def __init__(self, g, input_dim, hidden_dim, output_dim, n_layers, activation, dropout):
        super(GIN_delta, self).__init__()
        self.g = g
        self.ginlayers = nn.ModuleList()
        # self.batch_norms = nn.ModuleList()
        if n_layers > 1:
            self.ginlayers.append(GINConv(Linear(input_dim, hidden_dim), activation=activation))
            # self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            for i in range(1, n_layers - 1):
                self.ginlayers.append(GINConv(Linear(hidden_dim, hidden_dim),
                                              activation=activation))
                # self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.ginlayers.append(GINConv(Linear(hidden_dim, output_dim), activation=activation))
            # self.batch_norms.append(nn.BatchNorm1d(output_dim))
        else:
            self.ginlayers.append(GINConv(Linear(input_dim, output_dim), activation=activation))
            # self.batch_norms.append(nn.BatchNorm1d(output_dim))

        # Record previous embedding
        self.embedding = th.Tensor([[0 for i in range(output_dim)]
                                    for j in range(g.number_of_nodes())]).requires_grad_(True)

    def forward(self, features, ngh_high_deg=None, ngh_low_deg=None, edge_mask=None):
        # list of hidden representation at each layer (including the input layer)
        # hidden_rep = [h]
        g = self.g
        h = features
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h, edge_mask)
            # h = self.batch_norms[i](h)
            # h = F.relu(h)

            # Delta update features
            if ngh_high_deg is not None or ngh_low_deg is not None:
                # Combine delta-inferenced embedding and previous embedding
                h = self.combine_embedding(self.embedding, h, ngh_high_deg, ngh_low_deg)

        #     hidden_rep.append(h)
        # score_over_layer = 0
        # # perform graph sum pooling over all nodes in each layer
        # for i, h in enumerate(hidden_rep):
        #     pooled_h = self.pool(g, h)
        #     score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        # return score_over_layer

        # # Delta update features
        # if ngh_high_deg is not None or ngh_low_deg is not None:
        #     # Combine delta-inferenced embedding and previous embedding
        #     h = self.combine_embedding(self.embedding, h, ngh_high_deg, ngh_low_deg)

        h = F.log_softmax(h, dim=-1)
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
                            device,
                            lr,
                            weight_decay,
                            ngh_high_deg=None,
                            ngh_low_deg=None):
    g = model.g
    features = g.ndata['feat']
    train_mask = g.ndata['train_mask'].bool()
    val_mask = g.ndata['val_mask']
    labels = g.ndata['label']
    n_edges = g.number_of_edges()
    edge_mask = g.edata['edge_mask']

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # initialize graph
    loss_log = []
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features, ngh_high_deg, ngh_low_deg, edge_mask)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        # # Update embedding
        # model.embedding = torch.nn.Parameter(logits)

        # acc = evaluate_delta_edge_masked(model, val_mask, device, ngh_high_deg, ngh_low_deg)
        # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(), acc,
        #                                     n_edges / np.mean(dur) / 1000))

        loss_log.append(round(loss.item(), 2))
    np.savetxt('../../../results/loss/' + args.dataset + '_evo_delta_loss' + '.txt',
               loss_log,
               fmt='%.2f')

    # Update embedding
    model.embedding = torch.nn.Parameter(logits)


def evaluate_delta_edge_masked(model, mask, device, nodes_high_deg=None, nodes_low_deg=None):
    r"""
    Update feature of updated nodes according to node degree
    "model" should be GCN_delta
    """

    g = model.g
    features = g.ndata['feat']
    labels = g.ndata['label']
    mask = mask.bool()  # Convert int8 to bool
    edge_mask = g.edata['edge_mask']

    model.eval()
    with torch.no_grad():
        logits = model(features, nodes_high_deg, nodes_low_deg, edge_mask)
        # logits = model.embedding
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        # Update embedding
        model.embedding = torch.nn.Parameter(logits)
        return correct.item() * 1.0 / len(labels)