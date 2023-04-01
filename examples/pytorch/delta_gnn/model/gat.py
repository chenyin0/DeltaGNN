import torch
import torch as th
import torch.nn as nn

import dgl.nn as dglnn
import dgl
from .gatconv_delta import GATConv_delta


class GAT(nn.Module):

    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, feat_drop, attn_drop,
                 heads):
        super().__init__()
        self.g = g
        self.layers = nn.ModuleList()

        if n_layers != len(heads):
            assert ('The size of "heads" not match with "n_layers"')

        if n_layers > 1:
            self.layers.append(
                dglnn.GATConv(in_feats,
                              n_hidden,
                              heads[0],
                              feat_drop=feat_drop,
                              attn_drop=attn_drop,
                              activation=activation))
            for i in range(1, n_layers - 1):
                self.layers.append(
                    dglnn.GATConv(n_hidden * heads[i - 1],
                                  n_hidden,
                                  heads[i],
                                  feat_drop=feat_drop,
                                  attn_drop=attn_drop,
                                  activation=activation))
            self.layers.append(
                dglnn.GATConv(n_hidden * heads[-2],
                              n_classes,
                              heads[-1],
                              feat_drop=feat_drop,
                              attn_drop=attn_drop,
                              activation=activation))
        else:
            self.layers.append(
                dglnn.GATConv(in_feats,
                              n_classes,
                              heads[0],
                              feat_drop=feat_drop,
                              attn_drop=attn_drop,
                              activation=activation))

        # Record previous logits
        self.logits = torch.zeros(g.number_of_nodes(), n_classes)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        # for bn in self.bns:
        #     bn.reset_parameters()

    def forward(self, g, inputs):
        h = inputs

        # GAT need add_self_loop
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
            
            sign_a = torch.sign(h).int()
            non_zero_a = torch.count_nonzero(sign_a, dim=1).reshape(1, -1).squeeze()
            # print(non_zero_a.tolist())
            non_zero_sum = sum(non_zero_a.tolist())
            h_size = h.numel()
            zero_num = h_size - non_zero_sum
            print('Feat Sparsity: ', zero_num, h_size, round(zero_num / h_size, 3))
            
        return h


def train(args, model, device, lr, weight_decay):
    # define train/val samples, loss function and optimizer
    g = model.g
    features = g.ndata['feat'].to(device)
    train_mask = g.ndata['train_mask'].bool().to(device)
    val_mask = g.ndata['val_mask'].to(device)
    labels = g.ndata['label'].to(device)

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # training loop
    for epoch in range(args.n_epochs):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # acc = evaluate(model, val_mask, device)
        # print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, loss.item(), acc))


def evaluate(model, mask, device):
    g = model.g
    features = g.ndata['feat'].to(device)
    labels = g.ndata['label'].to(device)
    mask = mask.bool().to(device)  # Convert int8 to bool

    model.eval()
    with torch.no_grad():
        logits = model(g, features)
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
        logits = model(g, features)
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


class GAT_delta(nn.Module):

    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, feat_drop, attn_drop,
                 heads):
        super().__init__()
        self.g = g
        self.layers = nn.ModuleList()

        if n_layers != len(heads):
            assert ('The size of "heads" not match with "n_layers"')

        if n_layers > 1:
            self.layers.append(
                GATConv_delta(in_feats,
                              n_hidden,
                              heads[0],
                              feat_drop=feat_drop,
                              attn_drop=attn_drop,
                              activation=activation))
            for i in range(1, n_layers - 1):
                self.layers.append(
                    GATConv_delta(n_hidden * heads[i - 1],
                                  n_hidden,
                                  heads[i],
                                  feat_drop=feat_drop,
                                  attn_drop=attn_drop,
                                  activation=activation))
            self.layers.append(
                GATConv_delta(n_hidden * heads[-2],
                              n_classes,
                              heads[-1],
                              feat_drop=feat_drop,
                              attn_drop=attn_drop,
                              activation=activation))
        else:
            self.layers.append(
                dglnn.GATConv(in_feats,
                              n_classes,
                              heads[0],
                              feat_drop=feat_drop,
                              attn_drop=attn_drop,
                              activation=activation))

        # Record previous embedding
        self.embedding = th.Tensor([[0 for i in range(n_classes)]
                                    for j in range(g.number_of_nodes())]).requires_grad_(True)

    def forward(self, g, inputs, ngh_high_deg=None, ngh_low_deg=None, edge_mask=None):
        h = inputs

        # # GAT need add_self_loop
        # g = dgl.remove_self_loop(g)
        # g = dgl.add_self_loop(g)

        for i, layer in enumerate(self.layers):
            h = layer(g, h, edge_mask)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)

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
                            device,
                            lr,
                            weight_decay,
                            ngh_high_deg=None,
                            ngh_low_deg=None):
    # define train/val samples, loss function and optimizer
    g = model.g
    features = g.ndata['feat']
    train_mask = g.ndata['train_mask'].bool()
    val_mask = g.ndata['val_mask']
    labels = g.ndata['label']
    edge_mask = g.edata['edge_mask']

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # training loop
    for epoch in range(args.n_epochs):
        model.train()
        logits = model(g, features, ngh_high_deg, ngh_low_deg, edge_mask)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # acc = evaluate_delta_edge_masked(model, val_mask, device, nodes_high_deg, nodes_low_deg)
        # print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, loss.item(), acc))

    # Update embedding
    model.embedding = torch.nn.Parameter(logits)


def evaluate_delta_edge_masked(model, mask, device, nodes_high_deg=None, nodes_low_deg=None):
    g = model.g
    features = g.ndata['feat']
    labels = g.ndata['label']
    mask = mask.bool()  # Convert int8 to bool
    edge_mask = g.edata['edge_mask']

    model.eval()
    with torch.no_grad():
        logits = model(g, features, nodes_high_deg, nodes_low_deg, edge_mask)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        # Update embedding
        model.embedding = torch.nn.Parameter(logits)
        return correct.item() * 1.0 / len(labels)
