import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset
import dgl
import time

def ogbn_mag_preprocess(dataset):
    """
    Generate subgraph (paper, cites, paper) from ogbn-mag
    """

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']['paper']
    valid_idx = split_idx['valid']['paper']
    test_idx = split_idx['test']['paper']

    g_original, labels = dataset[0]
    labels = labels['paper'].squeeze()
    sub_g = dgl.edge_type_subgraph(g_original, [('paper', 'cites', 'paper')])
    h_sub_g = dgl.to_homogeneous(sub_g)
    h_sub_g.ndata['feat'] = g_original.nodes['paper'].data['feat']
    h_sub_g.ndata['label'] = labels

    # Initialize mask
    train_mask = torch.zeros((h_sub_g.num_nodes(), ), dtype=torch.bool)
    train_mask[train_idx] = True
    valid_mask = torch.zeros((h_sub_g.num_nodes(), ), dtype=torch.bool)
    valid_mask[valid_idx] = True
    test_mask = torch.zeros((h_sub_g.num_nodes(), ), dtype=torch.bool)
    test_mask[test_idx] = True

    h_sub_g.ndata['train_mask'] = train_mask
    h_sub_g.ndata['val_mask'] = valid_mask
    h_sub_g.ndata['test_mask'] = test_mask

    return h_sub_g

class GAT(nn.Module):

    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            ))
        self.gat_layers.append(
            dglnn.GATConv(
                hid_size * heads[0],
                out_size,
                heads[1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            ))

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

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
    print(f"Training with DGL built-in GATConv module.")

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
    elif args.dataset == "ogbn-mag":
        data = DglNodePropPredDataset('ogbn-mag', root='../../../../dataset')
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    if args.dataset == 'ogbn-mag':
        g = ogbn_mag_preprocess(data)
    else:
        g = data[0]
    # g = data[0]
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    # g = g.int().to(device)
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    # create GAT model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GAT(in_size, 128, out_size, heads=[8, 1]).to(device)

    # model training
    print("Training...")
    train(g, features, labels, masks, model)

    # test the model
    print("Testing...")
    time_start = time.perf_counter()
    acc = evaluate(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))
    print('Task time: ', time.perf_counter() - time_start)