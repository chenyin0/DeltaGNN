import torch
import torch as th
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import time
# from tqdm import tqdm


class GraphSAGE(nn.Module):

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()

        if n_layers > 1:
            self.layers.append(SAGEConv(in_feats, n_hidden))
            for i in range(1, n_layers - 1):
                self.layers.append(SAGEConv(n_hidden, n_hidden))
            self.layers.append(SAGEConv(n_hidden, n_classes))
        else:
            self.layers.append(SAGEConv(in_feats, n_classes))

        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

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


class GraphSAGE_delta(nn.Module):
    r"""
    Initial para:
        g: subgraph of original graph
        in_feats: feature matrix of current subgraph
    """

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout, feat_init_num):
        super().__init__()
        # self.g = g
        self.layers = nn.ModuleList()

        if n_layers > 1:
            self.layers.append(SAGEConv(in_feats, n_hidden))
            for i in range(1, n_layers - 1):
                self.layers.append(SAGEConv(n_hidden, n_hidden))
            self.layers.append(SAGEConv(n_hidden, n_classes))
        else:
            self.layers.append(SAGEConv(in_feats, n_classes))

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

    def forward(self, features, edge_index):
        h = features
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layers[-1](h, edge_index)
        return h.log_softmax(dim=-1)


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

    # print(sen_mask, insen_mask)

    # sen_mask_resize = [[i for col in range(feat.shape[-1])] for i in sen_mask]
    # # print(sen_mask_resize)
    # insen_mask_resize = [[i for col in range(feat.shape[-1])] for i in insen_mask]

    sen_mask_ts = torch.unsqueeze(th.tensor(sen_mask, dtype=torch.bool), 1)
    sen_feat = th.masked_select(feat, sen_mask_ts)
    sen_feat = th.reshape(sen_feat, (-1, feat.shape[-1]))
    insen_mask_ts = torch.unsqueeze(th.tensor(insen_mask, dtype=torch.bool), 1)
    insen_feat = th.masked_select(feat, insen_mask_ts)
    insen_feat = th.reshape(insen_feat, (-1, feat.shape[-1]))

    # print(sen_feat.shape[0], insen_feat.shape[0])

    ind_sen = ind[sen_mask]
    ind_insen = ind[insen_mask]

    feat_index_sen = np.array([[ind_sen[row]] * embedding_entire.shape[1]
                      for row in range(sen_feat.shape[0])])
    feat_index_insen = np.array([[ind_insen[row]] * embedding_entire.shape[1]
                        for row in range(insen_feat.shape[0])])

    feat_index_sen = th.from_numpy(feat_index_sen)
    feat_index_insen = th.from_numpy(feat_index_insen)

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

    feat_index = np.array([[ind[row]] * embedding.shape[1] for row in range(feat.shape[0])])
    feat_index = th.from_numpy(feat_index)
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
            # if v_sen is not None or v_insen is not None:
            #     out, embedding = combine_embedding(model.embedding, out, ind, v_sen, v_insen)
                # Update embedding
                # model.embedding = torch.nn.Parameter(embedding)
                # model.embedding = store_embedding(model.embedding, out, ind)
            model.embedding = store_embedding(model.embedding, out, ind)
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
