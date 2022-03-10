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
    if args.dataset == 'cora':
        data = CoraGraphDataset(raw_dir='./dataset')
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset(raw_dir='./dataset')
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset(raw_dir='./dataset')
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    # features = g.ndata['feat']
    # # print(features)
    # labels = g.ndata['label']
    # train_mask = g.ndata['train_mask']
    # val_mask = g.ndata['val_mask']
    # test_mask = g.ndata['test_mask']
    # in_feats = features.shape[1]
    # n_classes = data.num_labels
    # n_edges = data.graph.number_of_edges()
    # print("""----Data statistics------'
    #   #Edges %d
    #   #Classes %d
    #   #Train samples %d
    #   #Val samples %d
    #   #Test samples %d""" %
    #       (n_edges, n_classes, train_mask.int().sum().item(),
    #        val_mask.int().sum().item(), test_mask.int().sum().item()))

    ##
    """ Construct evolve graph """
    g_csr = g.adj_sparse('csr')
    root_node_q = util.gen_root_node_queue(g)
    node_q = util.bfs_traverse(g_csr, root_node_q)

    init_node_rate = 0.1
    init_node_num = round(len(node_q) * init_node_rate)
    init_nodes = node_q[0:init_node_num]
    print('\n>> Initial node num', len(init_nodes))
    # Pop nodes which have been added
    node_q = node_q[init_node_num:]

    # Gen node_mapping from g_orig to g_evo, for DGL compels consecutive node id
    node_map_orig2evo = dict()
    node_map_evo2orig = dict()

    g_evo, features_evo, labels_evo, train_mask_evo, val_mask_evo, test_mask_evo = util.graph_evolve(
        init_nodes, g_csr, g, node_map_orig2evo, node_map_evo2orig)
    ##

    in_feats = features_evo.shape[1]
    n_classes = data.num_labels

    # add self loop
    if args.self_loop:
        g_evo = dgl.remove_self_loop(g_evo)
        g_evo = dgl.add_self_loop(g_evo)
    n_edges = g_evo.number_of_edges()

    # normalization
    degs = g_evo.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g_evo.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g_evo, in_feats, args.n_hidden, n_classes, args.n_layers,
                F.relu, args.dropout)
    model_retrain = GCN(g_evo, in_feats, args.n_hidden, n_classes,
                        args.n_layers, F.relu, args.dropout)

    if cuda:
        model.cuda()
    else:
        model.cpu()

    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    loss_fcn_retrain = torch.nn.CrossEntropyLoss()
    optimizer_retrain = torch.optim.Adam(model_retrain.parameters(),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay)

    # for name, param in model.named_parameters(): # View optimizable parameter
    #     if param.requires_grad:
    #         print(name)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features_evo)
        loss = loss_fcn(logits[train_mask_evo], labels_evo[train_mask_evo])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features_evo, labels_evo, val_mask_evo)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            acc, n_edges / np.mean(dur) / 1000))

    print()
    print(">>> Accuracy on original graph: ")
    acc = evaluate(model, features_evo, labels_evo, test_mask_evo)
    print("Test accuracy {:.2%}".format(acc))

    # Evolve graph
    print(">>> Accuracy on evolove graph: ")
    # model_evolve = copy.deepcopy(model)
    # model_evolve.cpu()
    # loss_fcn_evolve = copy.deepcopy(loss_fcn)
    # optimizer_evolve = copy.deepcopy(optimizer)

    # Add new edges
    n_nodes = model.g.number_of_nodes()
    iter = 8
    node_batch = 100
    # edge_epoch = np.arange(0, iter * edge_batch, edge_batch)
    accuracy = np.zeros((iter, 3))
    for i in range(iter):
        print('\n>> Add node-batch @ iter = {:d}'.format(i))
        print('>> node_q size: {:d}'.format(len(node_q)))
        add_node_num = i * node_batch
        if add_node_num < len(node_q):
            add_nodes = node_q[:add_node_num]
            node_q = node_q[add_node_num:]
        else:
            add_nodes = node_q
            node_q.clear()

        g_evo, features_evo, labels_evo, train_mask_evo, val_mask_evo, test_mask_evo = util.graph_evolve(
            add_nodes, g_csr, g, node_map_orig2evo, node_map_evo2orig, model.g)

        acc = evaluate(model, features_evo, labels_evo, test_mask_evo)
        print("Test accuracy of non-retrain @ {:d} nodes {:.2%}".format(
            model.g.number_of_nodes(), acc))
        accuracy[i][0] = add_node_num
        accuracy[i][1] = acc * 100

        # Retrain
        # train(model_evolve, features, model_evolve.g.number_of_edges(),
        #       train_mask, val_mask, labels, loss_fcn_evolve, optimizer_evolve)

        g_evo, features_evo, labels_evo, train_mask_evo, val_mask_evo, test_mask_evo = util.graph_evolve(
            add_nodes, g_csr, g, node_map_orig2evo, node_map_evo2orig,
            model_retrain.g)

        train(model_retrain, features_evo, model_retrain.g.number_of_edges(),
              train_mask_evo, val_mask_evo, labels_evo, loss_fcn_retrain,
              optimizer_retrain)

        # dur = []
        # for epoch in range(args.n_epochs):
        #     model.train()
        #     if epoch >= 3:
        #         t0 = time.time()
        #     # forward
        #     logits = model(features)
        #     loss = loss_fcn(logits[train_mask], labels[train_mask])

        #     # w1 = copy.deepcopy(model.layers[0].weight)
        #     print('\n>> Before train:')
        #     for param in model.parameters():
        #         print(param)

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        #     print('\n>> After train:')
        #     for param in model.parameters():
        #         print(param)

        #     # w2 = model.layers[0].weight
        #     # print(w1.equal(w2))

        #     if epoch >= 3:
        #         dur.append(time.time() - t0)

        #     acc = evaluate(model, features, labels, val_mask)
        # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        # "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
        #                             acc, n_edges / np.mean(dur) / 1000))

        # w1 = model.layers[0].weight
        # w11 = model_evolve.layers[0].weight
        # print('>> Training result: ')
        # print(w1.equal(w11))

        # Evaluate retrain acc
        acc = evaluate(model_retrain, features_evo, labels_evo, test_mask_evo)
        print("Test accuracy of retrain @ {:d} nodes {:.2%}".format(
            model_retrain.g.number_of_nodes(), acc))
        accuracy[i][2] = acc * 100

    if args.dataset == 'cora':
        np.savetxt('./results/cora_add_edge.txt',
                   accuracy,
                   fmt='%d, %.2f, %.2f')
    elif args.dataset == 'citeseer':
        np.savetxt('./results/citeseer_add_edge.txt',
                   accuracy,
                   fmt='%d, %.2f, %.2f')
    elif args.dataset == 'pubmed':
        np.savetxt('./results/pubmed_add_edge.txt',
                   accuracy,
                   fmt='%d, %.2f, %.2f')
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    # plot.plt_edge_epoch()

    # plot.plt_edge_epoch(edge_epoch, result)

    # ## Test diff-graph
    # print()
    # print(">>> Accuracy on different graph: ")
    # data = CiteseerGraphDataset()
    # # data = CoraGraphDataset()
    # g_t = data[0]

    # print(g_t.nodes())
    # print(g_t.edges())

    # # g_t.ndata['feat'] = g_t.ndata['feat'][:features.size()[0], :features.size()[1]]
    # features_t = g_t.ndata['feat']
    # features_t = features_t[:features.size()[0], :features.size()[1]]

    # # g_t.ndata['label'] = g_t.ndata['label'][:labels.size()[0]]
    # labels_t = g_t.ndata['label']
    # labels_t=labels_t[:labels.size()[0]]

    # # train_mask = g.ndata['train_mask']
    # # val_mask = g.ndata['val_mask']
    # # test_mask = g.ndata['test_mask']
    # # in_feats = features.shape[1]
    # # n_classes = data.num_labels
    # # n_edges = data.graph.number_of_edges()

    # # Initial for g_t
    # # add self loop
    # if args.self_loop:
    #     g_t = dgl.remove_self_loop(g_t)
    #     g_t = dgl.add_self_loop(g_t)
    # n_edges = g_t.number_of_edges()

    # # normalization
    # degs = g_t.in_degrees().float()
    # norm = torch.pow(degs, -0.5)
    # norm[torch.isinf(norm)] = 0
    # if cuda:
    #     norm = norm.cuda()
    # g_t.ndata['norm'] = norm.unsqueeze(1)

    # # create GCN model
    # model_diff = GCN(g_t, in_feats, args.n_hidden, n_classes, args.n_layers,
    #                  F.relu, args.dropout)

    # # Update diff_model weight
    # model_diff.layers[0].weight = model.layers[0].weight
    # model_diff.layers[1].weight = model.layers[1].weight

    # acc = evaluate(model_diff, features_t, labels_t, test_mask)
    # print("Test accuracy {:.2%}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset",
                        type=str,
                        default="cora",
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
