import argparse
import time
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset, AmazonCoBuyComputerDataset

import util
import time

from gcn import GCN
from gcn import GCN_delta

from torchviz import make_dot

import os


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


def evaluate_delta(model, features, labels, mask, updated_nodes):
    r"""
    Only update feature of updated nodes in inference
    """
    model.eval()
    with torch.no_grad():
        logits_prev = model.logits
        logits = model(features)
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


def main(args):
    # Overall task execution time
    Task_time_start = time.perf_counter()

    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset(raw_dir='./dataset')
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset(raw_dir='./dataset')
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset(raw_dir='./dataset')
    elif args.dataset == 'reddit':
        data = RedditDataset(raw_dir='./dataset')
    elif args.dataset == 'amazon_comp':
        data = AmazonCoBuyComputerDataset(raw_dir='./dataset')
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

    g_evo = util.graph_evolve(init_nodes, g_csr, g, node_map_orig2evo, node_map_evo2orig)
    ##

    features = g_evo.ndata['feat']
    labels = g_evo.ndata['label']
    train_mask = g_evo.ndata['train_mask']
    val_mask = g_evo.ndata['val_mask']
    test_mask = g_evo.ndata['test_mask']

    # features = g.ndata['feat']
    # labels = g.ndata['label']
    # train_mask = g.ndata['train_mask']
    # val_mask = g.ndata['val_mask']
    # test_mask = g.ndata['test_mask']

    in_feats = features.shape[1]
    if args.dataset == 'amazon_comp':
        n_classes = data.num_classes
    else:
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
    model = GCN(g_evo, in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout)
    model_retrain = GCN(g_evo, in_feats, args.n_hidden, n_classes, args.n_layers, F.relu,
                        args.dropout)
    model_delta = GCN(g_evo, in_feats, args.n_hidden, n_classes, args.n_layers, F.relu,
                      args.dropout)

    # for param in model.parameters():
    #     print(param)

    if cuda:
        model.cuda()
    else:
        model.cpu()

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_fcn_retrain = torch.nn.CrossEntropyLoss()
    optimizer_retrain = torch.optim.Adam(model_retrain.parameters(),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay)

    loss_fcn_delta = torch.nn.CrossEntropyLoss()
    optimizer_delta = torch.optim.Adam(model_delta.parameters(),
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
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(), acc,
        #                                     n_edges / np.mean(dur) / 1000))

    print()
    print(">>> Accuracy on original graph: ")
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))

    # Evolve graph
    print(">>> Accuracy on evolove graph: ")

    # Add new edges
    # n_nodes = model.g.number_of_nodes()
    # iter = 8
    i = 0
    node_batch = round(g.number_of_nodes() / 10)  # default = 10
    # edge_epoch = np.arange(0, iter * edge_batch, edge_batch)
    accuracy = []
    deg_th = args.deg_threshold
    delta_neighbor = []

    mem_access_q_full_retrain = []  # For gen mem trace
    mem_access_q_all_ngh = []  # For gen mem trace
    trace_path_full_retrain = './results/' + args.dataset + '_mem_trace_full_retrain.txt'
    trace_path_all_ngh = './results/' + args.dataset + '_mem_trace_all_ngh.txt'

    while len(node_q) > 0:
        print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('Add node-batch @ iter = {:d}'.format(i))
        print('node_q size: {:d}'.format(len(node_q)))
        # add_node_num = i * node_batch
        if node_batch < len(node_q):
            inserted_nodes = node_q[:node_batch]
            node_q = node_q[node_batch:]
        else:
            inserted_nodes = node_q
            node_q.clear()

        print('Add node size: ', len(inserted_nodes))

        ##
        """
        # No retraining
        """
        print('\n>> No retraining')
        util.graph_evolve(inserted_nodes, g_csr, g, node_map_orig2evo, node_map_evo2orig, model.g)

        print('Node_number:', model.g.number_of_nodes())
        print('Edge_number:', model.g.number_of_edges())

        # features = model.g.ndata['feat']
        # labels = model.g.ndata['label']
        # train_mask = model.g.ndata['train_mask']
        # val_mask = model.g.ndata['val_mask']
        # test_mask = model.g.ndata['test_mask']

        # acc = evaluate(model, features, labels, test_mask)
        # print("Test accuracy of non-retrain @ {:d} nodes {:.2%}".format(
        #     model.g.number_of_nodes(), acc))
        # acc_non_retrain = acc * 100

        # Get node index of added_nodes in evolve graph
        inserted_nodes_evo = util.nodes_reindex(node_map_orig2evo, inserted_nodes)
        # inserted_nodes_evo.sort()

        g_csr_evo = model.g.adj_sparse('csr')
        # Statistic neighbor edges and nodes
        node_full_retrain, edge_full_retrain = util.count_neighbor(model.g.nodes().tolist(),
                                                                   g_csr_evo, node_map_orig2evo,
                                                                   args.n_layers + 1,
                                                                   mem_access_q_full_retrain)

        # They are all full graph retrain in the initial time
        if i == 0:
            delta_neighbor.append(
                [node_full_retrain, edge_full_retrain, node_full_retrain, edge_full_retrain])

            # Record mem trace
            util.gen_mem_trace(mem_access_q_full_retrain, '')
        else:
            node_ngh_all_sum, edge_ngh_all_sum = util.count_neighbor_delta(
                inserted_nodes_evo, g_csr_evo, node_map_orig2evo, args.n_layers + 1, 0)

            print('>>', node_full_retrain, edge_full_retrain, node_ngh_all_sum, edge_ngh_all_sum)
            delta_neighbor.append(
                [node_full_retrain, edge_full_retrain, node_ngh_all_sum, edge_ngh_all_sum])

        # # Plot graph structure
        # g_evo_csr = model.g.adj_sparse('csr')
        # indptr = g_evo_csr[0]
        # indices = g_evo_csr[1]
        # plt_graph.graph_visualize(indptr, indices, None)

        # ##
        # """
        # # Full graph retraining
        # """
        # print('\n>> Full graph retraining')
        # util.graph_evolve(inserted_nodes, g_csr, g, node_map_orig2evo, node_map_evo2orig,
        #                   model_retrain.g)

        # features = model_retrain.g.ndata['feat']
        # labels = model_retrain.g.ndata['label']
        # train_mask = model_retrain.g.ndata['train_mask']
        # val_mask = model_retrain.g.ndata['val_mask']
        # test_mask = model_retrain.g.ndata['test_mask']
        # # if len(node_q) > 0:
        # #     train(model_retrain, features, model_retrain.g.number_of_edges(),
        # #           train_mask, val_mask, labels, loss_fcn_retrain,
        # #           optimizer_retrain)

        # time_start = time.perf_counter()
        # train(model_retrain, features, model_retrain.g.number_of_edges(), train_mask, val_mask,
        #       labels, loss_fcn_retrain, optimizer_retrain)
        # time_full_retrain = time.perf_counter() - time_start
        # print('>> Epoch training time with full nodes: {:.4}s'.format(time_full_retrain))

        # acc = evaluate(model_retrain, features, labels, test_mask)
        # print("Test accuracy of retrain @ {:d} nodes {:.2%}".format(
        #     model_retrain.g.number_of_nodes(), acc))
        # acc_retrain = acc * 100

        # ##
        # """
        # # Delta retraining only on inserted nodes
        # """
        # print('\n>> Delta retraining')
        # # Execute full retraining at the beginning
        # if i <= 0:
        #     util.graph_evolve(inserted_nodes, g_csr, g, node_map_orig2evo, node_map_evo2orig,
        #                       model_delta.g)
        # else:
        #     util.graph_evolve_delta(inserted_nodes, g_csr, g, node_map_orig2evo, node_map_evo2orig,
        #                             model_delta.g)

        # features = model_delta.g.ndata['feat']
        # labels = model_delta.g.ndata['label']
        # train_mask = model_delta.g.ndata['train_mask']
        # val_mask = model_delta.g.ndata['val_mask']
        # test_mask = model_delta.g.ndata['test_mask']

        # if len(node_q) > 0:
        #     time_start = time.perf_counter()
        #     train(model_delta, features, model_delta.g.number_of_edges(), train_mask, val_mask,
        #           labels, loss_fcn_delta, optimizer_delta)
        #     time_delta_retrain = time.perf_counter() - time_start
        #     print('>> Epoch training time in delta: {:.4}s'.format(time_delta_retrain))

        # if i <= 3:
        #     acc = evaluate(model_delta, features, labels, test_mask)
        # else:
        #     acc = evaluate_delta(model_delta, features, labels, test_mask, inserted_nodes_evo)
        #     # acc = evaluate(model_delta, features, labels, test_mask)
        # # acc = evaluate(model_delta, features, labels, test_mask)
        # print("Test accuracy of delta @ {:d} nodes {:.2%}".format(model_delta.g.number_of_nodes(),
        #                                                           acc))
        # acc_retrain_delta = acc * 100

        # accuracy.append([
        #     model.g.number_of_nodes(),
        #     model.g.number_of_edges(), acc_non_retrain, acc_retrain, acc_retrain_delta,
        #     acc_retrain_delta_all_ngh
        # ])

        i += 1

    deg_th = str(args.deg_threshold)
    # Dump log
    # np.savetxt('./results/' + args.dataset + '_accuracy_evo' + '.txt',
    #            accuracy,
    #            fmt='%d, %d, %.2f, %.2f', %.2f')
    np.savetxt('./results/' + args.dataset + '_delta_ngh.txt', delta_neighbor, fmt='%d, %d, %d, %d')

    # plot.plt_edge_epoch()
    # plot.plt_edge_epoch(edge_epoch, result)

    print('\n>> Task {:s} execution time: {:.4}s'.format(args.dataset,
                                                         time.perf_counter() - Task_time_start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed', 'reddit', 'amazon_comp').")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16, help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true', help="graph self-loop (default=False)")
    parser.add_argument("--deg-threshold",
                        type=int,
                        default=None,
                        help="degree threshold of neighbors nodes")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()

    # args.dataset = 'amazon_comp'
    # args.n_epochs = 0
    # args.deg_threshold = 300

    print('\n************ {:s} ************'.format(args.dataset))
    print(args)

    # new_val = torch.arange(1, 21).reshape((4, 5))
    # print(new_val)
    # src = torch.ones(4, 5, dtype=new_val.dtype)
    # row_id = th.tensor([0, 2, 3])
    # index = [[row_id[k].item() for i in range(new_val.shape[1])] for k in range(row_id.shape[0])]
    # print(index)
    # index=th.tensor(index)

    # # src = torch.arange(1, 11).reshape((2, 5))
    # # index = torch.tensor([[0, 0, 0, 0, 0], [2,2,2,2, 2]])
    # print(src.scatter_(0, index, new_val, reduce='add'))

    main(args)
