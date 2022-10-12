import argparse
import time
import xxlimited
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset, AmazonCoBuyComputerDataset
from dgl.data import AsNodePredDataset

import util
import time
import random

from model.gcn import GCN
import model.gcn as gcn
from model.graphsage import SAGE
import model.graphsage as graphsage
# from model.gcn import GCN_delta

import os
import plt.plt_workload
import plt.plt_graph

import json
import pathlib


def main(args):
    # Overall task execution time
    Task_time_start = time.perf_counter()

    # Load GNN model parameter
    model_name = args.model
    if model_name == 'gcn':
        path = os.getcwd()
        print(path)
        with open("./examples/pytorch/delta_gnn/gcn_para.json", 'r') as f:
            para = json.load(f)
            n_hidden = para['--n-hidden']
            n_layers = para['--n-layers']
            lr = para['--lr']
            weight_decay = para['--weight-decay']
            dropout = para['--dropout']
    elif model_name == 'graphsage':
        with open('./examples/pytorch/delta_gnn/graphsage_para.json', 'r') as f:
            para = json.load(f)
            n_hidden = para['--n-hidden']
            n_layers = para['--n-layers']
            # num_negs = para['--num-negs']
            fan_out = str(para['--fan-out'])
            batch_size = para['--batch-size']
            # log_every = para['--log-every']
            # eval_every = para['--eval-every']
            lr = para['--lr']
            weight_decay = para['--weight-decay']
            dropout = para['--dropout']
    else:
        assert ('Not define GNN model')

    # load and preprocess dataset
    if args.dataset == 'cora':
        dataset = CoraGraphDataset(raw_dir='./dataset')
    elif args.dataset == 'citeseer':
        dataset = CiteseerGraphDataset(raw_dir='./dataset')
    elif args.dataset == 'pubmed':
        dataset = PubmedGraphDataset(raw_dir='./dataset')
    elif args.dataset == 'reddit':
        dataset = RedditDataset(raw_dir='./dataset')
    elif args.dataset == 'amazon_comp':
        dataset = AmazonCoBuyComputerDataset(raw_dir='./dataset')
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = dataset[0]

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        gpu_id = args.gpu

    device = th.device("cuda:" + str(gpu_id) if cuda else "cpu")
    mode = args.mode

    # features = g.ndata['feat']
    # # print(features)
    # labels = g.ndata['label']
    # train_mask = g.ndata['train_mask']
    # val_mask = g.ndata['val_mask']
    # test_mask = g.ndata['test_mask']
    # in_feats = features.shape[1]
    # n_classes = dataset.num_labels
    # n_edges = dataset.graph.number_of_edges()
    # print("""----Data statistics------'
    #   #Edges %d
    #   #Classes %d
    #   #Train samples %d
    #   #Val samples %d
    #   #Test samples %d""" %
    #       (n_edges, n_classes, train_mask.int().sum().item(),
    #        val_mask.int().sum().item(), test_mask.int().sum().item()))

    ##

    g_csr = g.adj_sparse('csr')
    """ Traverse to get graph evolving snapshot """
    node_q = []
    file_ = pathlib.Path('./dataset/' + args.dataset + '_evo_seq.txt')
    if file_.exists():
        f = open(file_, "r")
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')  # Delete '\n'
            node_q.append(int(line))
    else:
        root_node_q = util.gen_root_node_queue(g)
        node_q = util.bfs_traverse(g_csr, root_node_q)

        with open(file_, 'w') as f:
            for i in node_q:
                f.write(str(i) + '\n')

    # ##
    # """ Profiling node locality with degree """
    # node_q_tmp = util.bfs_traverse(g_csr, [i for i in range(len(node_q))])
    # deg_node_trace_sorted = util.gen_trace_sorted_by_node_deg(g_csr, node_q_tmp)
    # util.rm_repeat_data(deg_node_trace_sorted)
    # deg_node_trace_wo_sort = util.gen_trace_without_sorted(g_csr, node_q_tmp)

    # locality_node_degs_sorted = util.eval_node_locality(deg_node_trace_sorted)
    # print(locality_node_degs_sorted)

    # # for i in deg_node_trace_wo_sort:
    # #     print(i[0])

    # locality_node_degs_wo_sort = util.eval_node_locality(deg_node_trace_wo_sort)
    # print(locality_node_degs_wo_sort)

    # # util.save_txt_2d('./results/mem_trace/' + args.dataset + '_node_deg_trace' + '.txt',
    # #                  deg_node_trace_sorted)

    # # ##
    # """ Plot degree distribution """
    # deg_dist = util.gen_degree_distribution(g_csr)
    # plt.plt_graph.plot_degree_distribution(deg_dist)

    # cnt = 0
    # total = len(node_q)
    # for i in range(len(deg_dist)):
    #     cnt += deg_dist[i]
    #     percentage = round(cnt / total * 100, 2)
    #     print(i, percentage)

    # ##
    # """ Profiling workloads imbalance """
    # deg_th = 6
    # plt.plt_workload.plt_workload_imbalance(g_csr, deg_th)

    ##
    """ Initial Graph """
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

    features = g_evo.ndata['feat'].to(device)
    labels = g_evo.ndata['label'].to(device)
    train_mask = g_evo.ndata['train_mask'].to(device)
    val_mask = g_evo.ndata['val_mask'].to(device)
    test_mask = g_evo.ndata['test_mask'].to(device)

    # features = g.ndata['feat']
    # labels = g.ndata['label']
    # train_mask = g.ndata['train_mask']
    # val_mask = g.ndata['val_mask']
    # test_mask = g.ndata['test_mask']

    in_feats = features.shape[1]
    n_classes = dataset.num_classes

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
        norm = norm.to(gpu_id)
        g = g.to(gpu_id)
        g_evo = g_evo.to(gpu_id)
    g_evo.ndata['norm'] = norm.unsqueeze(1)

    # create GNN model
    if model_name == 'gcn':
        model = GCN(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu, dropout).to(device)
        loss_fcn = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        model_retrain = GCN(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                            dropout).to(device)
        loss_fcn_retrain = CrossEntropyLoss()
        optimizer_retrain = Adam(model_retrain.parameters(), lr=lr, weight_decay=weight_decay)

        model_delta = GCN(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                          dropout).to(device)
        loss_fcn_delta = CrossEntropyLoss()
        optimizer_delta = Adam(model_delta.parameters(), lr=lr, weight_decay=weight_decay)

    elif model_name == 'graphsage':
        # model = SAGE(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu, dropout).to(device)

        ## Debug_yin
        g = dataset[0]
        model = SAGE(g, g.ndata['feat'].shape[1], n_hidden, n_classes, n_layers, F.relu,
                     dropout).to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        model_retrain = SAGE(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                             dropout).to(device)
        optimizer_retrain = Adam(model_retrain.parameters(), lr=lr, weight_decay=weight_decay)

        model_delta = SAGE(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                           dropout).to(device)
        optimizer_delta = Adam(model_delta.parameters(), lr=lr, weight_decay=weight_decay)

    # for param in model.parameters():
    #     print(param)

    if cuda:
        model.cuda()
    else:
        model.cpu()

    # for name, param in model.named_parameters(): # View optimizable parameter
    #     if param.requires_grad:
    #         print(name)

    # Train the initial graph (timestamp = 0)
    print("\n>>> Accuracy on initial graph (timestamp=0):")
    if model_name == 'gcn':
        gcn.train(args, model, features, model.g.number_of_edges(), train_mask, val_mask, labels,
                  loss_fcn, optimizer)
        acc = gcn.evaluate(model, features, labels, test_mask)
    elif model_name == 'graphsage':
        graphsage.train(args, model.g, model, device, fan_out, batch_size, optimizer)
        acc = graphsage.evaluate(device, model, model.g, test_mask, batch_size)

    print("Test accuracy {:.2%}".format(acc))

    # Evolve graph
    print("\n>>> Accuracy on evolove graph: ")

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
    if dump_mem_trace_flag:
        trace_path_full_retrain = './results/mem_trace/' + args.dataset + '_full_retrain.txt'
        trace_path_all_ngh = './results/mem_trace/' + args.dataset + '_all_ngh.txt'

        os.system('rm ' + trace_path_full_retrain)  # Reset mem trace
        os.system('rm ' + trace_path_all_ngh)  # Reset mem trace

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

        features = model.g.ndata['feat']
        labels = model.g.ndata['label']
        train_mask = model.g.ndata['train_mask']
        val_mask = model.g.ndata['val_mask']
        test_mask = model.g.ndata['test_mask']

        if model_name == 'gcn':
            acc = gcn.evaluate(model, features, labels, test_mask)
        elif model_name == 'graphsage':
            acc = graphsage.evaluate(device, model, model.g, test_mask, batch_size)
        print("Test accuracy of non-retrain @ {:d} nodes {:.2%}".format(
            model.g.number_of_nodes(), acc))
        acc_no_retrain = acc * 100

        # Get node index of added_nodes in evolve graph
        inserted_nodes_evo = util.nodes_reindex(node_map_orig2evo, inserted_nodes)
        # inserted_nodes_evo.sort()

        if dump_node_access_flag or dump_mem_trace_flag:
            g_csr_evo = model.g.adj_sparse('csr')
            # Statistic neighbor edges and nodes
            node_full_retrain, edge_full_retrain = util.count_neighbor(
                model.g.nodes().tolist(), g_csr_evo, node_map_orig2evo, n_layers,
                mem_access_q_full_retrain)

            if i == 0:
                # There are all full retrain in the initial time of train_full and train_all_ngh
                if dump_node_access_flag:
                    delta_neighbor.append([
                        node_full_retrain, edge_full_retrain, node_full_retrain, edge_full_retrain
                    ])

                if dump_mem_trace_flag:
                    # Record mem trace
                    mem_access_q_full_retrain = util.nodes_reindex(node_map_evo2orig,
                                                                   mem_access_q_full_retrain)
                    random.shuffle(mem_access_q_full_retrain)
                    util.dump_mem_trace(mem_access_q_full_retrain, trace_path_full_retrain)
                    util.dump_mem_trace(mem_access_q_full_retrain, trace_path_all_ngh)
                mem_access_q_full_retrain = []  # Reset queue
            else:
                if dump_node_access_flag:
                    node_ngh_all, edge_ngh_all = util.count_neighbor_delta(
                        inserted_nodes_evo, g_csr_evo, node_map_orig2evo, n_layers, 0,
                        mem_access_q_all_ngh)

                    print('>>', node_full_retrain, edge_full_retrain, node_ngh_all, edge_ngh_all)
                    delta_neighbor.append(
                        [node_full_retrain, edge_full_retrain, node_ngh_all, edge_ngh_all])

                if dump_mem_trace_flag:
                    # Record mem trace
                    mem_access_q_full_retrain = util.nodes_reindex(node_map_evo2orig,
                                                                   mem_access_q_full_retrain)
                    mem_access_q_all_ngh = util.nodes_reindex(node_map_evo2orig,
                                                              mem_access_q_all_ngh)
                    random.shuffle(mem_access_q_full_retrain)
                    random.shuffle(mem_access_q_all_ngh)
                    util.dump_mem_trace(mem_access_q_full_retrain, trace_path_full_retrain)
                    util.dump_mem_trace(mem_access_q_all_ngh, trace_path_all_ngh)
                mem_access_q_full_retrain = []  # Reset queue
                mem_access_q_all_ngh = []  # Reset queue

        # # Plot graph structure
        # g_evo_csr = model.g.adj_sparse('csr')
        # indptr = g_evo_csr[0]
        # indices = g_evo_csr[1]
        # plt_graph.graph_visualize(indptr, indices, None)

        if dump_accuracy_flag:
            ##
            """
            # Full graph retraining
            """
            print('\n>> Full graph retraining')
            util.graph_evolve(inserted_nodes, g_csr, g, node_map_orig2evo, node_map_evo2orig,
                              model_retrain.g)

            features = model_retrain.g.ndata['feat']
            labels = model_retrain.g.ndata['label']
            train_mask = model_retrain.g.ndata['train_mask']
            val_mask = model_retrain.g.ndata['val_mask']
            test_mask = model_retrain.g.ndata['test_mask']
            # if len(node_q) > 0:
            #     train(model_retrain, features, model_retrain.g.number_of_edges(),
            #           train_mask, val_mask, labels, loss_fcn_retrain,
            #           optimizer_retrain)

            time_start = time.perf_counter()
            if model_name == 'gcn':
                gcn.train(args, model_retrain, features, model_retrain.g.number_of_edges(),
                          train_mask, val_mask, labels, loss_fcn_retrain, optimizer_retrain)
                acc = gcn.evaluate(model_retrain, features, labels, test_mask)
            elif model_name == 'graphsage':
                graphsage.train(args, model_retrain.g, model_retrain, device, fan_out, batch_size,
                                optimizer)
                acc = graphsage.evaluate(device, model_retrain, model_retrain.g, test_mask,
                                         batch_size)

            time_full_retrain = time.perf_counter() - time_start
            print('>> Epoch training time with full nodes: {:.4}s'.format(time_full_retrain))
            print("Test accuracy of retrain @ {:d} nodes {:.2%}".format(
                model_retrain.g.number_of_nodes(), acc))
            acc_retrain = acc * 100

            ##
            """
            # Delta retraining only on inserted nodes
            """
            print('\n>> Delta retraining')
            # Execute full retraining at the beginning
            if i <= 0:
                util.graph_evolve(inserted_nodes, g_csr, g, node_map_orig2evo, node_map_evo2orig,
                                  model_delta.g)
            else:
                util.graph_evolve_delta(inserted_nodes, g_csr, g, node_map_orig2evo,
                                        node_map_evo2orig, model_delta.g)

            features = model_delta.g.ndata['feat']
            labels = model_delta.g.ndata['label']
            train_mask = model_delta.g.ndata['train_mask']
            val_mask = model_delta.g.ndata['val_mask']
            test_mask = model_delta.g.ndata['test_mask']

            if len(node_q) > 0:
                time_start = time.perf_counter()
                if model_name == 'gcn':
                    gcn.train(args, model_delta, features, model_delta.g.number_of_edges(),
                              train_mask, val_mask, labels, loss_fcn_delta, optimizer_delta)
                    if i <= 3:
                        acc = gcn.evaluate(model_delta, features, labels, test_mask)
                    else:
                        acc = gcn.evaluate_delta(model_delta, features, labels, test_mask,
                                                 inserted_nodes_evo)
                        # acc = evaluate(model_delta, features, labels, test_mask)
                elif model_name == 'graphsage':
                    graphsage.train(args, model_delta.g, model_delta, device, fan_out, batch_size,
                                    optimizer)
                    acc = graphsage.evaluate(device, model_delta, model_delta.g, test_mask,
                                             batch_size)

                time_delta_retrain = time.perf_counter() - time_start
                print('>> Epoch training time in delta: {:.4}s'.format(time_delta_retrain))
                print("Test accuracy of delta @ {:d} nodes {:.2%}".format(
                    model_delta.g.number_of_nodes(), acc))
                acc_retrain_delta = acc * 100

            accuracy.append([
                model.g.number_of_nodes(),
                model.g.number_of_edges(), acc_no_retrain, acc_retrain, acc_retrain_delta
            ])

        i += 1

    # Dump log
    if dump_accuracy_flag:
        np.savetxt('./results/accuracy/' + args.dataset + '_evo' + '.txt',
                   accuracy,
                   fmt='%d, %d, %.2f, %.2f, %.2f')
    if dump_node_access_flag:
        np.savetxt('./results/node_access/' + args.dataset + '_evo.txt',
                   delta_neighbor,
                   fmt='%d, %d, %d, %d')

    # plot.plt_edge_epoch()
    # plot.plt_edge_epoch(edge_epoch, result)

    print('\n>> Task {:s} execution time: {:.4}s'.format(args.dataset,
                                                         time.perf_counter() - Task_time_start))

    for i in range(len(accuracy)):
        print(i, round(accuracy[i][3], 2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN')
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="Model name ('gcn', 'graphsage', 'gin', 'gat').")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed', 'reddit', 'amazon_comp').")
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
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()

    args.model = 'graphsage'
    args.dataset = 'cora'
    args.n_epochs = 200
    args.gpu = 0

    dump_accuracy_flag = 1
    dump_mem_trace_flag = 0
    dump_node_access_flag = 0

    print('\n************ {:s} ************'.format(args.dataset))
    print(args)

    main(args)
