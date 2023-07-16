import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import examples.pytorch.delta_gnn.model.model_utils as model_utils
import torch_geometric.nn.inits as inits

from torch.nn import LayerNorm, Linear, ReLU
# from tqdm import tqdm
# from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import DeepGCNLayer, GENConv
# from torch_geometric.utils import scatter
from torch.optim import Adam


class DeeperGCN(torch.nn.Module):

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout):
        super().__init__()

        self.node_encoder = Linear(in_feats, n_hidden)
        # self.edge_encoder = Linear(in_feats, n_hidden)

        self.layers = torch.nn.ModuleList()
        for i in range(1, n_layers + 1):
            conv = GENConv(n_hidden,
                           n_hidden,
                           aggr='softmax',
                           t=1.0,
                           learn_t=True,
                           num_layers=2,
                           norm='layer')
            norm = LayerNorm(n_hidden, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(n_hidden, n_classes)
        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, features, edge_index):
        h = self.node_encoder(features)
        # edge_attr = self.edge_encoder(edge_attr)

        h = self.layers[0].conv(h, edge_index)

        for layer in self.layers[1:]:
            h = layer(h, edge_index)

        h = self.layers[0].act(self.layers[0].norm(h))
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.lin(h)

        return h


def train(model, device, train_loader, lr):
    model.train()
    # pbar = tqdm(total=int(len(train_loader.dataset)))
    # pbar.set_description(f'Epoch {epoch:02d}')
    optimizer = Adam(model.parameters(), lr=lr)
    time_epoch = 0
    loss_list = []
    for step, batch in enumerate(train_loader):
        t_st = time.time()
        x, edge_index, y = batch.x.to(device), batch.edge_index.to(device), batch.y.to(device)
        optimizer.zero_grad()
        out = model(x, edge_index)
        # loss = F.nll_loss(out, y.squeeze(1))
        loss = F.cross_entropy(out, y.squeeze(1))
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


class DeeperGCN_delta(DeeperGCN):
    r"""
    Initial para:
        g: subgraph of original graph
        in_feats: feature matrix of current subgraph
    """

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout, feat_init_num):
        super(DeeperGCN_delta, self).__init__(in_feats, n_hidden, n_classes, n_layers, dropout)

        # Record previous embedding
        self.embedding = th.Tensor([[0 for i in range(n_classes)]
                                    for j in range(feat_init_num)]).requires_grad_(True)

    def reset_parameters(self):
        DeeperGCN.reset_parameters(self)
        inits.zeros(self.embedding)

def train_delta(model,
                device,
                data_loader,
                lr,
                v_sen_feat_loc,
                v_insen_feat_loc,
                v_sen=None,
                v_insen=None):
    # print('>> Start delta train')
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)

    time_epoch = 0
    loss_list = []
    # batch_base = 0
    for step, batch in enumerate(data_loader):
        t_st = time.time()
        x, edge_index, y = batch.x.to(device), batch.edge_index.to(device), batch.y.to(device)
        n_id = batch.n_id
        # If label (y) has index
        optimizer.zero_grad()
        out = model(x, edge_index)
        if v_sen is not None or v_insen is not None:
            out, embedding = model_utils.feature_merge(model.embedding, out, n_id, v_sen_feat_loc,
                                                       v_insen_feat_loc, v_sen, v_insen)
            model.embedding = torch.nn.Parameter(embedding)  # Update embedding
        loss = F.cross_entropy(out, y.squeeze(1))
        # loss = F.nll_loss(out, y.squeeze(1))
        loss.backward()
        optimizer.step()
        time_epoch += (time.time() - t_st)
        loss_list.append(loss.item())
        # print(loss.item())
        # pbar.update(batch.batch_size)
        # batch_base += batch.batch_size

    # print('>> Finish delta train')
    return np.mean(loss_list), time_epoch


@torch.no_grad()
def validate_delta(model,
                   device,
                   loader,
                   v_sen_feat_loc,
                   v_insen_feat_loc,
                   v_sen=None,
                   v_insen=None):
    model.eval()
    y_pred, y_true = [], []
    for step, batch in enumerate(loader):
        x, edge_index, y = batch.x.to(device), batch.edge_index.to(device), batch.y
        n_id = batch.n_id
        out = model(x, edge_index)
        if v_sen is not None or v_insen is not None:
            out, embedding = model_utils.feature_merge(model.embedding, out, n_id, v_sen_feat_loc,
                                                       v_insen_feat_loc, v_sen, v_insen)
            # model.embedding = torch.nn.Parameter(embedding)
        out = out.log_softmax(dim=-1)
        y_pred.append(torch.argmax(out, dim=-1, keepdim=True).cpu())
        y_true.append(y)
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    correct = torch.sum(y_pred == y_true)
    return correct.item() * 1.0 / len(y_true)


@torch.no_grad()
def test_delta(model,
               device,
               data_loader,
               checkpt_file,
               v_sen_feat_loc,
               v_insen_feat_loc,
               v_sen=None,
               v_insen=None):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    y_pred, y_true = [], []
    for step, batch in enumerate(data_loader):
        x, edge_index, y = batch.x.to(device), batch.edge_index.to(device), batch.y
        n_id = batch.n_id
        out = model(x, edge_index)
        if v_sen is not None or v_insen is not None:
            out, embedding = model_utils.feature_merge(model.embedding, out, n_id, v_sen_feat_loc,
                                                       v_insen_feat_loc, v_sen, v_insen)
            model.embedding = torch.nn.Parameter(embedding)
        out = out.log_softmax(dim=-1)
        y_pred.append(torch.argmax(out, dim=-1, keepdim=True).cpu())
        y_true.append(y)
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    correct = torch.sum(y_pred == y_true)
    return correct.item() * 1.0 / len(y_true)
