import torch
import torch.nn as nn

import dgl.nn as dglnn
import dgl


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