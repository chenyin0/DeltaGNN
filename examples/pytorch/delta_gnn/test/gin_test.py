import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler

from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling

from dgl.data import CoraGraphDataset, CiteseerGraphDataset
from torch.optim import Adam
import time
from dgl import AddSelfLoop
import torch as th


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

    def __init__(self, g, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.g = g
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(GINConv(mlp, learn_eps=False))  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (SumPooling())  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        score_over_layer = score_over_layer.view(score_over_layer.shape[1])
        return score_over_layer


def split_fold10(labels, fold_idx=0):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, valid_idx = idx_list[fold_idx]
    return train_idx, valid_idx


# def evaluate(dataloader, device, model):
#     model.eval()
#     total = 0
#     total_correct = 0
#     for batched_graph, labels in dataloader:
#         batched_graph = batched_graph.to(device)
#         labels = labels.to(device)
#         feat = batched_graph.ndata.pop("attr")
#         total += len(labels)
#         logits = model(batched_graph, feat)
#         _, predicted = torch.max(logits, 1)
#         total_correct += (predicted == labels).sum().item()
#     acc = 1.0 * total_correct / total
#     return acc

# def train(train_loader, val_loader, device, model):
#     # loss function, optimizer and scheduler
#     loss_fcn = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.01)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

#     # training loop
#     for epoch in range(350):
#         model.train()
#         total_loss = 0
#         for batch, (batched_graph, labels) in enumerate(train_loader):
#             batched_graph = batched_graph.to(device)
#             labels = labels.to(device)
#             feat = batched_graph.ndata.pop("attr")
#             logits = model(batched_graph, feat)
#             loss = loss_fcn(logits, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         scheduler.step()
#         train_acc = evaluate(train_loader, device, model)
#         valid_acc = evaluate(val_loader, device, model)
#         print(
#             "Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | Validation Acc. {:.4f} ".format(
#                 epoch, total_loss / (batch + 1), train_acc, valid_acc
#             )
#         )


def train(args, model, device, lr, weight_decay):
    g = model.g
    features = g.ndata['feat']
    train_mask = g.ndata['train_mask'].bool()
    val_mask = g.ndata['val_mask']
    labels = g.ndata['label']
    n_edges = g.number_of_edges()

    # print(train_mask, val_mask)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # initialize graph
    loss_log = []
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        loss_log.append(round(loss.item(), 2))

    np.savetxt('./results/loss/' + args.dataset + '_evo_loss' + '.txt', loss_log, fmt='%.2f')

    # acc = evaluate(model, val_mask, device)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default="MUTAG",
    #     choices=["MUTAG", "PTC", "NCI1", "PROTEINS"],
    #     help="name of dataset (default: MUTAG)",
    # )
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="Model name ('gcn', 'graphsage', 'gin', 'gat').")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help=
        "Dataset name ('cora', 'citeseer', 'pubmed', 'reddit', 'ogbn-arxiv', 'ogbn-mag', 'amazon_comp')."
    )
    # parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    # parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200, help="number of training epochs")
    # parser.add_argument("--n-hidden", type=int, default=16, help="number of hidden gcn units")
    # parser.add_argument("--n-layers", type=int, default=2, help="number of gcn layers")
    # parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true', help="graph self-loop (default=False)")
    parser.add_argument(
        "--mode",
        default='mixed',
        choices=['cpu', 'mixed', 'puregpu'],
        help=
        "Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, 'puregpu' for pure-GPU training."
    )
    parser.add_argument("--deg-threshold",
                        type=int,
                        default=None,
                        help="degree threshold of neighbors nodes")
    args = parser.parse_args()

    args.dataset = 'cora'

    print(f"Training with DGL built-in GINConv module with a fixed epsilon = 0")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # load and split dataset
    # dataset = GINDataset(
    #     args.dataset, raw_dir='./dataset', self_loop=True, degree_as_nlabel=False
    # )  # add self_loop and disable one-hot encoding for input features

    # load and preprocess dataset
    transform = (AddSelfLoop()
                 )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == 'cora':
        dataset = CoraGraphDataset(raw_dir='../../../../dataset', transform=transform)
    elif args.dataset == 'citeseer':
        dataset = CiteseerGraphDataset(raw_dir='../../../../dataset', transform=transform)

    g = dataset[0]

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        gpu_id = args.gpu

    device = th.device("cuda:" + str(gpu_id) if cuda else "cpu")
    mode = args.mode

    # labels = [l for _, l in dataset]
    # train_idx, val_idx = split_fold10(labels)

    # # create dataloader
    # train_loader = GraphDataLoader(
    #     dataset,
    #     sampler=SubsetRandomSampler(train_idx),
    #     batch_size=128,
    #     pin_memory=torch.cuda.is_available(),
    # )
    # val_loader = GraphDataLoader(
    #     dataset,
    #     sampler=SubsetRandomSampler(val_idx),
    #     batch_size=128,
    #     pin_memory=torch.cuda.is_available(),
    # )

    # create GIN model
    in_size = g.ndata['feat'].shape[1]
    out_size = g.ndata['label'].shape[0]
    model = GIN(g, in_size, 16, out_size).to(device)

    # # model training/validating
    # print("Training...")
    # train(train_loader, val_loader, device, model)

    lr = 0.01
    weight_decay = 0
    test_mask = g.ndata['test_mask']
    train(args, model, device, lr, weight_decay)
    acc = eval(model, test_mask, device)
    print(acc)