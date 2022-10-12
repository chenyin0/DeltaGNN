import torch as th
import torch.nn as nn
import torch.functional as F
import torchmetrics.functional as MF
import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import tqdm
from model.sageconv_delta import SAGEConv_delta
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
import numpy as np


class SAGE(nn.Module):

    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(g, in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.g = g
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, features):
        h = features
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(g,
                                th.arange(g.num_nodes()).to(g.device),
                                sampler,
                                device=device,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=0)
        buffer_device = th.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = th.empty(g.num_nodes(),
                         self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
                         device=buffer_device,
                         pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = th.nn.functional.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def train(args, g, model, device, fanout, batch_size, optimizer):
    # create sampler & dataloader
    train_mask = g.ndata['train_mask']
    train_idx = th.Tensor(np.nonzero(train_mask.numpy())[0]).long()
    val_mask = g.ndata['val_mask']
    val_idx = th.Tensor(np.nonzero(val_mask.numpy())[0]).long()

    sampler = NeighborSampler([int(fanout_) for fanout_ in fanout.split(',')],
                              prefetch_node_feats=['feat'],
                              prefetch_labels=['label'])
    use_uva = (args.mode == 'mixed')
    train_dataloader = DataLoader(g,
                                  train_idx,
                                  sampler,
                                  device=device,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=0,
                                  use_uva=use_uva)

    val_dataloader = DataLoader(g,
                                val_idx,
                                sampler,
                                device=device,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False,
                                num_workers=0,
                                use_uva=use_uva)

    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            loss = th.nn.functional.cross_entropy(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        acc = evaluate_with_sample(model, g, val_dataloader)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, total_loss / (it + 1),
                                                                     acc.item()))


def evaluate_with_sample(model, g, dataloader):
    r"""
    Evaluate in training, include neighbor sampling (Used in training)
    """

    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with th.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(blocks[-1].dstdata['label'])
            y_hats.append(model(blocks, x))
    return MF.accuracy(th.cat(y_hats), th.cat(ys))


def evaluate(device, model, g, mask, batch_size):
    r"""
    Evaluate with all neighbor aggregation (Used in test)
    """

    model.eval()
    with th.no_grad():
        pred = model.inference(g, device, batch_size)  # pred in buffer_device
        pred = pred[mask]
        label = g.ndata['label'][mask].to(pred.device)
        return MF.accuracy(pred, label)


class SAGE_delta(nn.Module):
    r"""
    Initial para:
        g: subgraph of original graph
        in_feats: feature matrix of current subgraph
    """

    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(g, in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.g = g
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(SAGEConv_delta(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(SAGEConv_delta(n_hidden, n_hidden, 'mean'))
            self.layers.append(SAGEConv_delta(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(SAGEConv_delta(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # Record previous embedding
        self.embedding = th.Tensor([[0 for i in range(n_classes)]
                                    for j in range(g.number_of_nodes())]).requires_grad_(True)

    def forward(self, blocks, features, ngh_high_deg=None, ngh_low_deg=None):
        h = features
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)

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
