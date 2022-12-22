import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset
import time


class GCN(nn.Module):

    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(dglnn.GraphConv(hid_size, hid_size, activation=F.relu))
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(200):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        # print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, loss.item(), acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-arxiv",
        help="Dataset name ('cora', 'citeseer', 'pubmed').",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")

    # load and preprocess dataset
    transform = (AddSelfLoop()
                 )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == "cora":
        data = CoraGraphDataset(raw_dir='../../../../dataset', transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(raw_dir='../../../../dataset', transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(raw_dir='../../../../dataset', transform=transform)
    elif args.dataset == "ogbn-arxiv":
        data = AsNodePredDataset(DglNodePropPredDataset('ogbn-arxiv', root='../../../../dataset'))
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    g = data[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    g = g.int().to(device)
    g = dgl.add_self_loop(g)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    # train_mask = g.ndata["train_mask"]
    # val_mask = g.ndata["val_mask"]
    # test_mask = g.ndata["test_mask"]
    # print(train_mask.cpu().numpy().tolist(), torch.count_nonzero(train_mask).item())
    # print()
    # print(val_mask.cpu().numpy().tolist(), torch.count_nonzero(val_mask).item())
    # print()
    # print(test_mask.cpu().numpy().tolist(), torch.count_nonzero(test_mask).item())
    # input()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5).to(device)
    norm[torch.isinf(norm)] = 0
    g.ndata["norm"] = norm.unsqueeze(1)

    # create GCN model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GCN(in_size, 128, out_size).to(device)

    # device = g.device
    # g = dgl.to_simple(g.cpu(), return_counts='cnt')
    # print(g.edata['cnt'].cpu().numpy().tolist())
    # g = g.to(device)

    # model training
    print("Training...")
    time_start = time.perf_counter()
    train(g, features, labels, masks, model)
    print('Training time: ', time.perf_counter() - time_start)

    # test the model
    print("Testing...")
    time_start = time.perf_counter()
    acc = evaluate(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))
    print('Inference time: ', time.perf_counter() - time_start)