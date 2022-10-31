import argparse
import time
from matplotlib import test
import numpy as np
import torch
import torch as th
import torch.nn.functional as F

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset, AmazonCoBuyComputerDataset
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl import AddSelfLoop
import preprocess

import util
import g_update
import random

from model.gcn import GCN
import model.gcn as gcn
from model.graphsage import SAGE
import model.graphsage as graphsage
from model.gat import GAT
import model.gat as gat

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
    elif model_name == 'gat':
        with open('./examples/pytorch/delta_gnn/gat_para.json', 'r') as f:
            para = json.load(f)
            n_hidden = para['--n-hidden']
            n_layers = para['--n-layers']
            lr = para['--lr']
            weight_decay = para['--weight-decay']
            feat_dropout = para['--feat-drop']
            attn_dropout = para['--attn-drop']
            heads_str = str(para['--heads'])
            heads = [int(i) for i in heads_str.split(',')]
    else:
        assert ('Not define GNN model')

    # load and preprocess dataset
    transform = (AddSelfLoop()
                 )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == 'cora':
        dataset = CoraGraphDataset(raw_dir='./dataset', transform=transform)
    elif args.dataset == 'citeseer':
        dataset = CiteseerGraphDataset(raw_dir='./dataset', transform=transform)
    elif args.dataset == 'pubmed':
        dataset = PubmedGraphDataset(raw_dir='./dataset', transform=transform)
    elif args.dataset == 'reddit':
        dataset = RedditDataset(raw_dir='./dataset', transform=transform)
    elif args.dataset == 'ogbn-arxiv':
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-arxiv', root='./dataset'))
    elif args.dataset == 'ogbn-mag':
        dataset = DglNodePropPredDataset('ogbn-mag', root='./dataset')
    elif args.dataset == 'amazon_comp':
        dataset = AmazonCoBuyComputerDataset(raw_dir='./dataset')
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.dataset == 'ogbn-mag':
        g = preprocess.ogbn_mag_preprocess(dataset)
    else:
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
        if args.dataset == 'cora' or args.dataset == 'citeseer':
            # root_node_q = util.gen_root_node_queue(g)
            # node_q = util.bfs_traverse(g_csr, root_node_q)
            node_q = g.nodes().numpy().tolist()
        elif args.dataset == 'ogbn-arxiv' or args.dataset == 'ogbn-mag':
            node_q = util.sort_node_by_timestamp('./dataset/' + args.dataset + '_node_year.csv')

        with open(file_, 'w') as f:
            for i in node_q:
                f.write(str(i) + '\n')

    # Gen node_seq
    g_struct_init_ratio = 0.5
    node_seq = util.gen_snapshot(g_struct_init_ratio, 10, g.number_of_nodes())

    # # Graph evolved size
    # cora_seq = [540, 691, 892, 1168, 1530, 2036, 2708]
    # citeseer_seq = [639, 824, 1080, 1414, 1881, 2502, 3327]
    # ogbn_arxiv_seq = [41125, 53160, 69499, 90941, 120740, 160451, 169343]
    # ogbn_mag_seq = []

    # ## Plot degree follows node id increasing
    # nodes = g.nodes()
    # in_degree = g.in_degrees(nodes)
    # # in_degree = g.in_degrees(node_q)
    # plt.plt_graph.plot_node_degree(in_degree.numpy().tolist(), args.dataset)
    # input('>> input')

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

    # ##
    # """ Initial Graph """
    # init_node_rate = 0.1
    # init_node_num = round(len(node_q) * init_node_rate)
    # init_nodes = node_q[0:init_node_num]
    # print('\n>> Initial node num', len(init_nodes))
    # # Pop nodes which have been added
    # node_q = node_q[init_node_num:]

    # if args.dataset == 'cora':
    #     node_seq = cora_seq
    # elif args.dataset == 'citeseer':
    #     node_seq = citeseer_seq
    # elif args.dataset == 'ogbn-arxiv':
    #     node_seq = ogbn_arxiv_seq
    # elif args.dataset == 'ogbn-mag':
    #     node_seq = ogbn_mag_seq

    # Gen node_mapping from g_orig to g_evo, for DGL compels consecutive node id
    node_map_orig2evo = dict()
    node_map_evo2orig = dict()

    init_nodes = node_q[:node_seq[0]]
    # g_evo = g_update.graph_evolve(init_nodes, g_csr, g, node_map_orig2evo, node_map_evo2orig,
    #                               n_layers)
    g_evo = g_update.graph_struct_init(args, g_struct_init_ratio, init_nodes, g, node_map_orig2evo,
                                       node_map_evo2orig)
    ##

    # # Update train/val/test mask for ogbn-arxiv (Optional)
    # if args.dataset == 'ogbn-arxiv':
    #     from dgl.data.utils import generate_mask_tensor
    #     from dgl.data.citation_graph import _sample_mask

    #     splitted_idx = dataset_raw.get_idx_split()
    #     train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx[
    #         "test"]
    #     labels = g.ndata['label']

    #     train_mask = generate_mask_tensor(_sample_mask(train_idx, labels.shape[0]))
    #     val_mask = generate_mask_tensor(_sample_mask(val_idx, labels.shape[0]))
    #     test_mask = generate_mask_tensor(_sample_mask(test_idx, labels.shape[0]))

    #     g.ndata['train_mask'] = train_mask
    #     g.ndata['val_mask'] = val_mask
    #     g.ndata['test_mask'] = test_mask

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

    # create GNN model
    if model_name == 'gcn':
        model_golden = GCN(g, in_feats, n_hidden, n_classes, n_layers, F.relu, dropout).to(device)
        model = GCN(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu, dropout).to(device)
        model_retrain = GCN(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                            dropout).to(device)
        model_delta = GCN(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                          dropout).to(device)
    elif model_name == 'graphsage':
        model_golden = SAGE(g, in_feats, n_hidden, n_classes, n_layers, F.relu, dropout).to(device)
        model = SAGE(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu, dropout).to(device)
        model_retrain = SAGE(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                             dropout).to(device)
        model_delta = SAGE(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                           dropout).to(device)
    elif model_name == 'gat':
        model_golden = GAT(g, in_feats, n_hidden, n_classes, n_layers, F.relu, feat_dropout,
                           attn_dropout, heads).to(device)
        model = GAT(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu, feat_dropout,
                    attn_dropout, heads).to(device)
        model_retrain = GAT(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu, feat_dropout,
                            attn_dropout, heads).to(device)
        model_delta = GAT(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu, feat_dropout,
                          attn_dropout, heads).to(device)

    # for param in model.parameters():
    #     print(param)

    # for name, param in model.named_parameters(): # View optimizable parameter
    #     if param.requires_grad:
    #         print(name)

    # Train model_golden
    print("\n>>> Accuracy on full graph with model_golden:")
    test_mask = model_golden.g.ndata['test_mask']
    if model_name == 'gcn':
        gcn.train(args, model_golden, device, lr, weight_decay)
        acc = gcn.evaluate(model_golden, test_mask, device)
    elif model_name == 'graphsage':
        graphsage.train(args, model_golden, device, fan_out, batch_size, lr, weight_decay)
        acc = graphsage.evaluate(device, model_golden, test_mask, batch_size)
    elif model_name == 'gat':
        gat.train(args, model_golden, device, lr, weight_decay)
        acc = gat.evaluate(model_golden, test_mask, device)

    print("Test accuracy {:.2%}".format(acc))

    # Train the initial graph (timestamp = 0)
    print("\n>>> Accuracy on initial graph (timestamp=0):")
    test_mask = model.g.ndata['test_mask']
    if model_name == 'gcn':
        gcn.train(args, model, device, lr, weight_decay)
        acc = gcn.evaluate(model, test_mask, device)
    elif model_name == 'graphsage':
        graphsage.train(args, model, device, fan_out, batch_size, lr, weight_decay)
        acc = graphsage.evaluate(device, model, test_mask, batch_size)
    elif model_name == 'gat':
        gat.train(args, model, device, lr, weight_decay)
        acc = gat.evaluate(model, test_mask, device)

    accuracy = []
    acc_no_retrain = acc_retrain = acc_retrain_delta = acc * 100
    # accuracy.append([
    #     model.g.number_of_nodes(),
    #     model.g.number_of_edges(), acc_no_retrain, acc_retrain, acc_retrain_delta
    # ])

    print("Test accuracy {:.2%}".format(acc))

    # Evolve graph
    print("\n>>> Accuracy on evolove graph: ")
    # # Add new edges
    # interval = 10
    # i = 0
    # node_batch = round(g.number_of_nodes() / interval)  # default = 10
    # # edge_epoch = np.arange(0, iter * edge_batch, edge_batch)

    deg_th = args.deg_threshold
    delta_neighbor = []

    mem_access_q_full_retrain = []  # For gen mem trace
    mem_access_q_all_ngh = []  # For gen mem trace
    if dump_mem_trace_flag:
        trace_path_full_retrain = './results/mem_trace/' + args.dataset + '_full_retrain.txt'
        trace_path_all_ngh = './results/mem_trace/' + args.dataset + '_all_ngh.txt'

        os.system('rm ' + trace_path_full_retrain)  # Reset mem trace
        os.system('rm ' + trace_path_all_ngh)  # Reset mem trace

    # while len(node_q) > 0:
    #     print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    #     print('Add node-batch @ iter = {:d}'.format(i))
    #     print('node_q size: {:d}'.format(len(node_q)))
    #     # add_node_num = i * node_batch
    #     if node_batch < len(node_q):
    #         inserted_nodes = node_q[:node_batch]
    #         node_q = node_q[node_batch:]
    #     else:
    #         inserted_nodes = node_q
    #         node_q.clear()

    # Add new nodes
    for i in range(len(node_seq[1:])):
        print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('Add nodes @ iter = {:d}'.format(i + 1))
        inserted_nodes = node_q[node_seq[i]:node_seq[i + 1] - 1]
        print('Add nodes: {:d}, Total nodes: {:d}'.format(len(inserted_nodes),
                                                          model_retrain.g.number_of_nodes()))

        ##
        """
        # Periodic retraining
        """
        print('\n>> Periodic retraining')
        model.g = g_update.graph_evolve(inserted_nodes, g_csr, g, node_map_orig2evo,
                                        node_map_evo2orig, n_layers, model.g)

        print('Node_number:', model.g.number_of_nodes())
        print('Edge_number:', model.g.number_of_edges())

        test_mask = model.g.ndata['test_mask']
        retrain_epoch = 3
        if model_name == 'gcn':
            if (i + 1) % retrain_epoch == 0 or i == 0:
                gcn.train(args, model, device, lr, weight_decay)
            acc = gcn.evaluate(model, test_mask, device)
        elif model_name == 'graphsage':
            if (i + 1) % retrain_epoch == 0 or i == 0:
                graphsage.train(args, model, device, fan_out, batch_size, lr, weight_decay)
            acc = graphsage.evaluate(device, model, test_mask, batch_size)
        elif model_name == 'gat':
            if (i + 1) % retrain_epoch == 0 or i == 0:
                gat.train(args, model, device, lr, weight_decay)
            acc = gat.evaluate(model, test_mask, device)

        print("Test accuracy of non-retrain @ {:d} nodes {:.2%}".format(
            model.g.number_of_nodes(), acc))
        acc_no_retrain = acc * 100

        # Get node index of added_nodes in evolve graph
        inserted_nodes_evo = g_update.get_nodes_reindex(node_map_orig2evo, inserted_nodes)
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
                    mem_access_q_full_retrain = util.get_nodes_reindex(
                        node_map_evo2orig, mem_access_q_full_retrain)
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
                    mem_access_q_full_retrain = util.get_nodes_reindex(
                        node_map_evo2orig, mem_access_q_full_retrain)
                    mem_access_q_all_ngh = util.get_nodes_reindex(node_map_evo2orig,
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
            model_retrain.g = g_update.graph_evolve(inserted_nodes, g_csr, g, node_map_orig2evo,
                                                    node_map_evo2orig, n_layers, model_retrain.g)

            time_start = time.perf_counter()
            test_mask = model_retrain.g.ndata['test_mask']
            if model_name == 'gcn':
                gcn.train(args, model_retrain, device, lr, weight_decay)
                acc = gcn.evaluate(model_retrain, test_mask, device)
            elif model_name == 'graphsage':
                graphsage.train(args, model_retrain, device, fan_out, batch_size, lr, weight_decay)
                acc = graphsage.evaluate(device, model_retrain, test_mask, batch_size)
            elif model_name == 'gat':
                gat.train(args, model_retrain, device, lr, weight_decay)
                acc = gat.evaluate(model_retrain, test_mask, device)

            time_full_retrain = time.perf_counter() - time_start
            print('>> Epoch training time with full nodes: {}'.format(
                util.time_format(time_full_retrain)))
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
                model_delta.g = g_update.graph_evolve(inserted_nodes, g_csr, g, node_map_orig2evo,
                                                      node_map_evo2orig, n_layers, model_delta.g)
            else:
                model_delta.g = g_update.graph_evolve_delta(inserted_nodes, g_csr, g,
                                                            node_map_orig2evo, node_map_evo2orig,
                                                            model_delta.g)

            time_start = time.perf_counter()
            test_mask = model_delta.g.ndata['test_mask']
            if model_name == 'gcn':
                gcn.train(args, model_delta, device, lr, weight_decay)
                if i <= 3:
                    acc = gcn.evaluate(model_delta, test_mask, device)
                else:
                    acc = gcn.evaluate_delta(model_delta, test_mask, device, inserted_nodes_evo)
                    # acc = evaluate(model_delta, test_mask, device)
            elif model_name == 'graphsage':
                graphsage.train(args, model_delta, device, fan_out, batch_size, lr, weight_decay)
                acc = graphsage.evaluate(device, model_delta, test_mask, batch_size)
            elif model_name == 'gat':
                gat.train(args, model_delta, device, lr, weight_decay)
                acc = gat.evaluate(model_delta, test_mask, device)

            time_delta_retrain = time.perf_counter() - time_start
            print('>> Epoch training time in delta: {}'.format(
                util.time_format(time_delta_retrain)))
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

    print('\n>> Task {:s} execution time: {}'.format(
        args.dataset, util.time_format(time.perf_counter() - Task_time_start)))

    for i in range(len(accuracy)):
        print(i, round(accuracy[i][3], 2), round(accuracy[i][2], 2))


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
                        default=None,
                        help="degree threshold of neighbors nodes")
    # parser.set_defaults(self_loop=True)
    args = parser.parse_args()

    # args.model = 'gcn'
    # args.model = 'graphsage'
    args.model = 'gat'

    args.dataset = 'cora'
    # args.dataset = 'citeseer'
    # args.dataset = 'ogbn-arxiv'
    # args.dataset = 'ogbn-mag'

    args.n_epochs = 200
    args.gpu = 0
    # args.mode = 'mixed'

    dump_accuracy_flag = 1
    dump_mem_trace_flag = 0
    dump_node_access_flag = 0

    print('\n************ {:s} ************'.format(args.dataset))
    print(args)

    main(args)
