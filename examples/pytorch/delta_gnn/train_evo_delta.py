import argparse
import time
import numpy as np
import torch
import torch as th
import torch.nn.functional as F

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset, AmazonCoBuyComputerDataset
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset

import util
import time
import random

from model.gcn import GCN
from model.gcn import GCN_delta
import model.gcn as gcn
from model.graphsage import SAGE, SAGE_delta
import model.graphsage as graphsage

import os
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
    elif args.dataset == 'ogbn-arxiv':
        dataset_raw = DglNodePropPredDataset('ogbn-arxiv', root='./dataset')
        dataset = AsNodePredDataset(dataset_raw)
    # elif args.dataset == 'ogbn-mag':
    #     dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-mag', root='./dataset'))
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
    # g_csr = g.adj_sparse('csr')
    # root_node_q = util.gen_root_node_queue(g)
    # node_q = util.bfs_traverse(g_csr, root_node_q)

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
        if args.dataset == 'cora' or args.dataset == 'citeseer':
            root_node_q = util.gen_root_node_queue(g)
            node_q = util.bfs_traverse(g_csr, root_node_q)
        elif args.dataset == 'ogbn-arxiv':
            node_q = util.sort_node_by_timestamp('./dataset/' + args.dataset + '_node_year.csv')

        with open(file_, 'w') as f:
            for i in node_q:
                f.write(str(i) + '\n')

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

    # features = g_evo.ndata['feat'].to(device)
    # labels = g_evo.ndata['label'].to(device)
    # train_mask = g_evo.ndata['train_mask'].to(device)
    # val_mask = g_evo.ndata['val_mask'].to(device)
    # test_mask = g_evo.ndata['test_mask'].to(device)

    # features = g.ndata['feat']
    # labels = g.ndata['label']
    # train_mask = g.ndata['train_mask']
    # val_mask = g.ndata['val_mask']
    # test_mask = g.ndata['test_mask']

    in_feats = g_evo.ndata['feat'].shape[1]
    n_classes = dataset.num_classes

    # add self loop
    if args.self_loop:
        g_evo = dgl.remove_self_loop(g_evo)
        g_evo = dgl.add_self_loop(g_evo)

    # normalization
    degs = g_evo.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g_evo.ndata['norm'] = norm.unsqueeze(1)

    if cuda:
        if model_name == 'graphsage':
            if mode == 'puregpu':
                norm = norm.to(gpu_id)
                g = g.to(gpu_id)
                g_evo = g_evo.to(gpu_id)
        else:
            norm = norm.to(gpu_id)
            g = g.to(gpu_id)
            g_evo = g_evo.to(gpu_id)

    # create GCN model
    if model_name == 'gcn':
        model = GCN(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu, dropout).to(device)
        model_delta_all_ngh = GCN_delta(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                                        dropout).to(device)
    elif model_name == 'graphsage':
        model = SAGE(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu, dropout).to(device)
        model_delta_all_ngh = SAGE_delta(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                                         dropout).to(device)

    # Train the initial graph (timestamp = 0)
    print("\n>>> Accuracy on initial graph (timestamp=0):")
    test_mask = model.g.ndata['test_mask']
    if model_name == 'gcn':
        gcn.train(args, model, device, lr, weight_decay)
        acc = gcn.evaluate(model, test_mask, device)
    elif model_name == 'graphsage':
        graphsage.train(args, model, device, fan_out, batch_size, lr, weight_decay)
        acc = graphsage.evaluate(device, model, test_mask, batch_size)

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

    mem_access_q_delta_ngh = []  # For gen mem trace
    if dump_mem_trace_flag:
        trace_path_delta_ngh = './results/mem_trace/' + args.dataset + '_delta_ngh_deg_' + str(
            deg_th) + '.txt'
        os.system('rm ' + trace_path_delta_ngh)  # Reset mem trace

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

        # Execute full retraining at the beginning
        if i <= 0:
            util.graph_evolve(inserted_nodes, g_csr, g, node_map_orig2evo, node_map_evo2orig,
                              model_delta_all_ngh.g)
        else:
            util.graph_evolve_delta_all_ngh(inserted_nodes, g_csr, g, node_map_orig2evo,
                                            node_map_evo2orig, model_delta_all_ngh.g)

        # Get node index of added_nodes in evolve graph
        inserted_nodes_evo = util.get_nodes_reindex(node_map_orig2evo, inserted_nodes)
        # inserted_nodes_evo.sort()

        if dump_node_access_flag or dump_mem_trace_flag:
            # Count ngh access
            # g_csr_evo = model.g.adj_sparse('csr')
            g_csr_evo = model_delta_all_ngh.g.adj_sparse('csr')
            # They are all full graph retrain in the initial time
            if i == 0:
                # Statistic neighbor edges and nodes
                node_full_retrain, edge_full_retrain = util.count_neighbor(
                    model_delta_all_ngh.g.nodes().tolist(), g_csr_evo, node_map_orig2evo,
                    args.n_layers + 1, mem_access_q_delta_ngh)

                delta_neighbor.append([node_full_retrain, edge_full_retrain])
            else:
                node_ngh_delta_sum, edge_ngh_delta_sum = util.count_neighbor_delta(
                    inserted_nodes_evo, g_csr_evo, node_map_orig2evo, args.n_layers + 1, deg_th,
                    mem_access_q_delta_ngh)

                print('>>', node_ngh_delta_sum, edge_ngh_delta_sum)
                delta_neighbor.append([node_ngh_delta_sum, edge_ngh_delta_sum])

        if dump_mem_trace_flag:
            # Record mem trace
            mem_access_q_delta_ngh = util.get_nodes_reindex(node_map_evo2orig,
                                                            mem_access_q_delta_ngh)
            random.shuffle(mem_access_q_delta_ngh)
            util.dump_mem_trace(mem_access_q_delta_ngh, trace_path_delta_ngh)
        mem_access_q_delta_ngh = []  # Reset queue

        # # Plot graph structure
        # g_evo_csr = model.g.adj_sparse('csr')
        # indptr = g_evo_csr[0]
        # indices = g_evo_csr[1]
        # plt_graph.graph_visualize(indptr, indices, None)

        if dump_accuracy_flag:
            ##
            """
            # Delta retraining on delta neighbors
            """
            print('\n>> Delta neighbor retraining')
            # Get ngh with high deg and low deg
            ngh_high_deg, ngh_low_deg = util.get_ngh_with_deg_th(model_delta_all_ngh.g,
                                                                 inserted_nodes_evo, deg_th)
            print('High_deg_ngh_num: {:d}, Low_deg_ngh_num: {:d}'.format(
                len(ngh_high_deg), len(ngh_low_deg)))

            # Merge ngh_high_deg and ngh_low_deg
            ngh_dict = ngh_high_deg.copy()
            ngh_dict.update(ngh_low_deg)

            #Set edge weight (mask) according to node degree
            util.gen_edge_mask(model_delta_all_ngh.g, ngh_dict)

            features = model_delta_all_ngh.g.ndata['feat']
            labels = model_delta_all_ngh.g.ndata['label']
            train_mask = model_delta_all_ngh.g.ndata['train_mask']
            val_mask = model_delta_all_ngh.g.ndata['val_mask']
            test_mask = model_delta_all_ngh.g.ndata['test_mask']

            if len(node_q) > 0:
                time_start = time.perf_counter()
                test_mask = model_delta_all_ngh.g.ndata['test_mask']
                if model_name == 'gcn':
                    gcn.train_delta_edge_masked(args, model_delta_all_ngh, device, lr, weight_decay,
                                                list(ngh_high_deg.keys()), list(ngh_low_deg.keys()))
                    acc = gcn.evaluate_delta_edge_masked(model_delta_all_ngh, test_mask, device,
                                                         list(ngh_high_deg.keys()),
                                                         list(ngh_low_deg.keys()))
                elif model_name == 'graphsage':
                    graphsage.train_delta_edge_masked(args, model_delta_all_ngh, device, fan_out,
                                                      batch_size, lr, weight_decay,
                                                      list(ngh_high_deg.keys()),
                                                      list(ngh_low_deg.keys()))
                    acc = graphsage.evaluate_delta_edge_masked(device, model_delta_all_ngh,
                                                               test_mask, batch_size,
                                                               list(ngh_high_deg.keys()),
                                                               list(ngh_low_deg.keys()))

                time_full_retrain = time.perf_counter() - time_start
                print('>> Epoch training time with full nodes: {:.4}s'.format(time_full_retrain))
                print("Test accuracy of delta_ngh @ {:d} nodes {:.2%}".format(
                    model_delta_all_ngh.g.number_of_nodes(), acc))
                acc_retrain_delta_ngh = acc * 100

            accuracy.append(
                [model.g.number_of_nodes(),
                 model.g.number_of_edges(), acc_retrain_delta_ngh])

        i += 1

    deg_th = str(args.deg_threshold)
    # Dump log
    if dump_accuracy_flag:
        np.savetxt('./results/accuracy/' + args.dataset + '_evo_delta_' + deg_th + '.txt',
                   accuracy,
                   fmt='%d, %d, %.2f')
    if dump_node_access_flag:
        np.savetxt('./results/node_access/' + args.dataset + '_evo_delta_deg_' + deg_th + '.txt',
                   delta_neighbor,
                   fmt='%d, %d')

    # plot.plt_edge_epoch()
    # plot.plt_edge_epoch(edge_epoch, result)

    print('\n>> Task {:s} execution time: {:.4}s'.format(args.dataset,
                                                         time.perf_counter() - Task_time_start))

    for i in range(len(accuracy)):
        print(i, round(accuracy[i][2], 2))


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
                        default=0,
                        help="degree threshold of neighbors nodes")
    # parser.set_defaults(self_loop=False)
    args = parser.parse_args()

    args.model = 'graphsage'
    # args.dataset = 'cora'
    args.dataset = 'ogbn-arxiv'
    args.n_epochs = 200
    args.deg_threshold = 5
    args.gpu = 0

    dump_accuracy_flag = 1
    dump_mem_trace_flag = 0
    dump_node_access_flag = 0

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
