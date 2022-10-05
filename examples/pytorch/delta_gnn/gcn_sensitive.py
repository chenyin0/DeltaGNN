import argparse
import time
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from gcn import GCN
#from gcn_mp import GCN
#from gcn_spmv import GCN

import plt_graph
import random
import math
import copy
import util


def train(model, features, n_edges, train_mask, val_mask, labels, loss_fcn, optimizer):
    # initialize graph
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


def degree_distribute(g):
    # Statistic graph typology
    in_degree = g.in_degrees()
    # print(in_degree.tolist())
    # out_degree = g.out_degrees()
    # print()
    # print(out_degree.tolist())
    in_degree_max = torch.max(in_degree, 0)[0]
    in_degree_min = torch.min(in_degree, 0)[0]

    degree_list = [[] for i in range(in_degree_max + 1)]
    for node_id in range(len(in_degree)):
        degree_list[in_degree[node_id]].append(node_id)

    return degree_list


def add_noise_out_deg(g, degree_list, degree_begin, degree_end, intensity, node_ratio):
    """
    node ratio: the proportion of the total nodes which added noise
    intensity: the proportion of the noise amplitude with the original feature value
    """
    print()
    in_feats = g.ndata['feat'].shape[1]
    degree_end = min(degree_end, len(degree_list))  # Avoid over boundary
    num = round(g.number_of_nodes() * node_ratio)
    for degree_val in range(degree_begin, degree_end):
        node_list = degree_list[degree_val]
        # node_num = len(node_list)
        # num = round(node_num * node_ratio)
        loc_list = range(len(node_list))
        print('>> Deg: ', degree_val, ' Node num: ', len(loc_list), ' Selet node num: ', num)

        noise = torch.rand(in_feats)
        device = g.device
        noise = noise.to(device)
        noise = th.mul(noise, intensity)
        loc = random.sample(loc_list, min(num, len(loc_list)))
        # loc = random.sample(loc_list, num)
        for i in loc:
            node_id = node_list[i]
            # Add random noise on node feature
            # print(g.ndata['feat'][node_id].tolist())

            g.ndata['feat'][node_id] = th.add(noise, g.ndata['feat'][node_id])


def add_noise_in_deg(g, degree_list, degree_begin, degree_end, intensity, node_ratio):
    """
    node ratio: the proportion of the total nodes which added noise
    intensity: the proportion of the noise amplitude with the original feature value

    Note: intensity should be amortized by degree
    """
    print()
    in_feats = g.ndata['feat'].shape[1]
    degree_end = min(degree_end, len(degree_list))  # Avoid over boundary
    num = round(g.number_of_nodes() * node_ratio)
    for degree_val in range(degree_begin, degree_end):
        node_list = degree_list[degree_val]
        # node_num = len(node_list)
        # num = round(node_num * node_ratio)
        loc_list = range(len(node_list))
        print('>> Deg: ', degree_val, ' Node num: ', len(loc_list), ' Selet node num: ', num)

        noise = torch.rand(in_feats)
        device = g.device
        noise = noise.to(device)
        noise = th.mul(noise, intensity / max(degree_val, 1))  # Avoid division by zero (degree = 0)
        loc = random.sample(loc_list, min(num, len(loc_list)))
        # loc = random.sample(loc_list, num)
        for i in loc:
            node_id = node_list[i]
            # Add random noise on node feature
            # print(g.ndata['feat'][node_id].tolist())

            g.ndata['feat'][node_id] = th.add(noise, g.ndata['feat'][node_id])


def main(args):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset(raw_dir='./dataset', reverse_edge=False)
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset(raw_dir='./dataset', reverse_edge=False)
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset(raw_dir='./dataset', reverse_edge=False)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    # g = data[0]
    # if args.gpu < 0:
    #     cuda = False
    # else:
    #     cuda = True
    #     g = g.int().to(args.gpu)

    g = data[0]
    # util.save_graph_csr(g, args.dataset)

    task_round = 5
    acc_task = [0] * task_round
    for task_round_id in range(task_round):

        g = copy.deepcopy(data[0])
        if args.gpu < 0:
            cuda = False
        else:
            cuda = True
            g = g.int().to(args.gpu)

        # Add noise
        degree_list = degree_distribute(g)
        # plt_graph.plot_degree_distribution(degree_list)
        # add_noise(g, degree_list, 8, 9, 0.1, 33)

        features = g.ndata['feat']
        # torch.set_printoptions(profile="full")
        # print(features)
        # torch.set_printoptions(profile="default")
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_labels
        n_edges = data.graph.number_of_edges()
        # print("""----Data statistics------'
        # #Edges %d
        # #Classes %d
        # #Train samples %d
        # #Val samples %d
        # #Test samples %d""" %
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
        model = GCN(g, in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout)

        if cuda:
            model.cuda()
        loss_fcn = torch.nn.CrossEntropyLoss()

        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # # initialize graph
        # dur = []
        # for epoch in range(args.n_epochs):
        #     model.train()
        #     if epoch >= 3:
        #         t0 = time.time()
        #     # forward
        #     logits = model(features)
        #     loss = loss_fcn(logits[train_mask], labels[train_mask])

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        #     if epoch >= 3:
        #         dur.append(time.time() - t0)

        #     acc = evaluate(model, features, labels, val_mask)
        #     print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #           "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(), acc, n_edges / np.mean(dur) / 1000))

        train(model, features, model.g.number_of_edges(), train_mask, val_mask, labels, loss_fcn,
              optimizer)

        deg_begin = args.deg_begin
        deg_end = args.deg_end
        noise_intensity = args.noise_intensity
        node_ratio = args.node_ratio
        """ Add noise according to out-degree """
        # add_noise_out_deg(g, degree_list, deg_begin, deg_end, noise_intensity, node_ratio)
        """ Add noise according to in-degree """
        add_noise_in_deg(g, degree_list, deg_begin, deg_end, noise_intensity * 4, node_ratio)
        features = g.ndata['feat']

        print()
        acc = evaluate(model, features, labels, test_mask)
        print("Test accuracy {:.2%}".format(acc))
        acc_task[task_round_id] = acc

    acc_total = 0
    for i in acc_task:
        acc_total += i

    print()
    print("Deg: [{:d}, {:d}), Node_ratio: {:.2f}, Intensity: {:.2f}".format(
        deg_begin, deg_end, node_ratio, noise_intensity))
    print("Task round: {:d}, Test accuracy {:.2%}".format(task_round, acc_total / task_round))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset",
                        type=str,
                        default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16, help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true', help="graph self-loop (default=False)")
    parser.add_argument("--noise-intensity",
                        type=float,
                        default=0,
                        help="intensity of the noise error")
    parser.add_argument("--deg-begin", type=int, default=0, help="inital of the degree interval")
    parser.add_argument("--deg-end", type=int, default=0, help="tail of the degree interval")
    parser.add_argument("--node-ratio",
                        type=float,
                        default=0,
                        help="the ratio of the nodes added noise")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()

    # args.gpu = 0
    # args.dataset = 'cora'
    # args.noise_intensity = 0.1
    # args.node_ratio = 0.6
    # args.deg_begin = 2
    # args.deg_end = 3

    # print(args)

    main(args)
