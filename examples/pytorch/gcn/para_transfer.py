import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

import util
import copy

from gcn import GCN
#from gcn_mp import GCN
#from gcn_spmv import GCN


def train(model, features, n_edges, train_mask, val_mask, labels, loss_fcn,
          optimizer):
    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        # w1 = copy.deepcopy(model.layers[0].weight)
        # print('\n>> Before train:')
        # for param in model.parameters():
        #     print(param)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('\n>> After train:')
        # for param in model.parameters():
        #     print(param)
        # w2 = model.layers[0].weight
        # print(w1.equal(w2))

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
        #                                     acc, n_edges / np.mean(dur) / 1000))


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    # load and preprocess dataset
    # if args.dataset == 'cora':
    #     data = CoraGraphDataset(raw_dir='./dataset')
    # elif args.dataset == 'citeseer':
    #     data = CiteseerGraphDataset(raw_dir='./dataset')
    # elif args.dataset == 'pubmed':
    #     data = PubmedGraphDataset(raw_dir='./dataset')
    # else:
    #     raise ValueError('Unknown dataset: {}'.format(args.dataset))

    data = CiteseerGraphDataset(raw_dir='./dataset')
    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    # print("""----Data statistics------'
    #   #Edges %d
    #   #Classes %d
    #   #Train samples %d
    #   #Val samples %d
    #   #Test samples %d""" %
    #       (n_edges, n_classes, train_mask.int().sum().item(),
    #        val_mask.int().sum().item(), test_mask.int().sum().item()))

    # features = g_evo.ndata['feat']
    # labels = g_evo.ndata['label']
    # train_mask = g_evo.ndata['train_mask']
    # val_mask = g_evo.ndata['val_mask']
    # test_mask = g_evo.ndata['test_mask']

    # features = g.ndata['feat']
    # labels = g.ndata['label']
    # train_mask = g.ndata['train_mask']
    # val_mask = g.ndata['val_mask']
    # test_mask = g.ndata['test_mask']

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g, in_feats, args.n_hidden, n_classes, args.n_layers, F.relu,
                args.dropout)

    if cuda:
        model.cuda()
    else:
        model.cpu()

    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Train graph
    train(model, features, model.g.number_of_edges(), train_mask, val_mask,
          labels, loss_fcn, optimizer)

    print()
    print(">>> Accuracy on original graph: ")
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))

    # for param in model.parameters():
    #     print(param)

    ##
    """
    Parameter transfer
    """
    print("\n>>> Accuracy on weight transfer: ")

    data = PubmedGraphDataset(raw_dir='./dataset')
    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    # print("""----Data statistics------'
    #   #Edges %d
    #   #Classes %d
    #   #Train samples %d
    #   #Val samples %d
    #   #Test samples %d""" %
    #       (n_edges, n_classes, train_mask.int().sum().item(),
    #        val_mask.int().sum().item(), test_mask.int().sum().item()))

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model_trans = GCN(g, in_feats, args.n_hidden, n_classes, args.n_layers,
                      F.relu, args.dropout)

    if cuda:
        model_trans.cuda()
    else:
        model_trans.cpu()

    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model_trans.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # # Train graph
    # train(model, features, model.g.number_of_edges(), train_mask, val_mask,
    #       labels, loss_fcn, optimizer)

    # # Para transfer
    # model_trans.parameters = model.parameters
    # for param in model_trans.parameters():
    #     print(param)

    print()
    print(">>> Accuracy on parameter transfer: ")
    acc = evaluate(model_trans, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset",
                        type=str,
                        default="pubmed",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-epochs",
                        type=int,
                        default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden",
                        type=int,
                        default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers",
                        type=int,
                        default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay",
                        type=float,
                        default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop",
                        action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
