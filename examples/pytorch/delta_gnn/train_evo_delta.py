import argparse
import time
import numpy as np
import torch
import torch as th
import torch.nn.functional as F

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset, AmazonCoBuyComputerDataset
from dgl.data import AsNodePredDataset
from dgl import AddSelfLoop
from ogb.nodeproppred import DglNodePropPredDataset
import preprocess

import util
import g_update
import time
import random

from model.gcn import GCN, GCN_delta
import model.gcn as gcn
from model.graphsage import SAGE, SAGE_delta
import model.graphsage as graphsage
from model.gat import GAT, GAT_delta
import model.gat as gat
from model.gin import GIN, GIN_delta
import model.gin as gin

import os
import json
import pathlib
import faulthandler
import sys


def main(args):
    faulthandler.enable()
    # sys.stdout = open('log_train_evo_delta.txt', 'w')
    # Overall task execution time
    Task_time_start = time.perf_counter()
    print('>> Task start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # Load GNN model parameter
    model_name = args.model
    dataset_name = args.dataset
    if model_name == 'gcn':
        # path = os.getcwd()
        # print(path)
        with open('./gcn_para.json', 'r') as f:
            para = json.load(f)
            n_hidden = para[dataset_name]['--n-hidden']
            n_layers = para[dataset_name]['--n-layers']
            lr = para[dataset_name]['--lr']
            weight_decay = para[dataset_name]['--weight-decay']
            dropout = para[dataset_name]['--dropout']
    elif model_name == 'graphsage':
        with open('./graphsage_para.json', 'r') as f:
            para = json.load(f)
            n_hidden = para[dataset_name]['--n-hidden']
            n_layers = para[dataset_name]['--n-layers']
            # num_negs = para['--num-negs']
            fan_out = str(para[dataset_name]['--fan-out'])
            batch_size = para[dataset_name]['--batch-size']
            # log_every = para['--log-every']
            # eval_every = para['--eval-every']
            lr = para[dataset_name]['--lr']
            weight_decay = para[dataset_name]['--weight-decay']
            dropout = para[dataset_name]['--dropout']
    elif model_name == 'gat':
        with open('./gat_para.json', 'r') as f:
            para = json.load(f)
            n_hidden = para[dataset_name]['--n-hidden']
            n_layers = para[dataset_name]['--n-layers']
            lr = para[dataset_name]['--lr']
            weight_decay = para[dataset_name]['--weight-decay']
            feat_dropout = para[dataset_name]['--feat-drop']
            attn_dropout = para[dataset_name]['--attn-drop']
            heads_str = str(para[dataset_name]['--heads'])
            heads = [int(i) for i in heads_str.split(',')]
    elif model_name == 'gin':
        with open('./gin_para.json', 'r') as f:
            para = json.load(f)
            n_hidden = para[dataset_name]['--n-hidden']
            n_layers = para[dataset_name]['--n-layers']
            lr = para[dataset_name]['--lr']
            weight_decay = para[dataset_name]['--weight-decay']
            dropout = para[dataset_name]['--dropout']
    else:
        assert ('Not define GNN model')

    # load and preprocess dataset
    transform = (AddSelfLoop()
                 )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == 'cora':
        dataset = CoraGraphDataset(raw_dir='../../../dataset', transform=transform)
    elif args.dataset == 'citeseer':
        dataset = CiteseerGraphDataset(raw_dir='../../../dataset', transform=transform)
    elif args.dataset == 'pubmed':
        dataset = PubmedGraphDataset(raw_dir='../../../dataset', transform=transform)
    elif args.dataset == 'reddit':
        dataset = RedditDataset(raw_dir='../../../dataset', transform=transform)
    elif args.dataset == 'ogbn-arxiv':
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-arxiv', root='../../../dataset'))
    elif args.dataset == 'ogbn-mag':
        dataset = DglNodePropPredDataset('ogbn-mag', root='../../../dataset')
    # elif args.dataset == 'ogbn-mag':
    #     dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-mag', root='../../../dataset'))
    elif args.dataset == 'amazon_comp':
        dataset = AmazonCoBuyComputerDataset(raw_dir='../../../dataset')
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

    # Dump the train/val/test vertice index
    # global train_idx_orig, val_idx_orig, test_idx_orig
    # train_idx_orig, val_idx_orig, test_idx_orig = util.dump_dataset_index(
    #     g, args.dataset, './data/')
    util.dump_dataset_index(g, args.dataset, './data/')

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
    file_ = pathlib.Path('../../../dataset/' + args.dataset + '_evo_delta_seq.txt')
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
            node_q = util.sort_node_by_timestamp('../../../dataset/' + args.dataset +
                                                 '_node_year.csv')

        with open(file_, 'w') as f:
            for i in node_q:
                f.write(str(i) + '\n')

    # Gen node_seq
    g_struct_init_ratio = 0.3
    g_snapshot_total_num = 16
    node_seq = util.gen_snapshot(g_struct_init_ratio, g_snapshot_total_num, g.number_of_nodes())

    # # Graph evolved size
    # cora_seq = [540, 691, 892, 1168, 1530, 2036, 2708]
    # citeseer_seq = [639, 824, 1080, 1414, 1881, 2502, 3327]
    # ogbn_arxiv_seq = [41125, 53160, 69499, 90941, 120740, 160451, 169343]
    # ogbn_mag_seq = []

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
    g_evo = g_update.graph_struct_init(args, g_struct_init_ratio, g_snapshot_total_num, init_nodes,
                                       g, node_map_orig2evo, node_map_evo2orig)
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
        model_always_retrain = GCN_delta(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                                         dropout).to(device)
    elif model_name == 'graphsage':
        model = SAGE(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu, dropout).to(device)
        model_delta_all_ngh = SAGE_delta(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                                         dropout).to(device)
        model_always_retrain = SAGE_delta(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                                          dropout).to(device)
    elif model_name == 'gat':
        model = GAT(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu, feat_dropout,
                    attn_dropout, heads).to(device)
        model_delta_all_ngh = GAT_delta(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                                        feat_dropout, attn_dropout, heads).to(device)
        model_always_retrain = GAT_delta(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                                         feat_dropout, attn_dropout, heads).to(device)
    elif model_name == 'gin':
        model = GIN(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu, dropout).to(device)
        model_delta_all_ngh = GIN_delta(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                                        dropout).to(device)
        model_always_retrain = GIN_delta(g_evo, in_feats, n_hidden, n_classes, n_layers, F.relu,
                                         dropout).to(device)

    # Train the initial graph (timestamp = 0)
    print("\n>>> Accuracy on initial graph (timestamp=0):")
    test_mask = model.g.ndata['test_mask']
    if model_name == 'gcn':
        gcn.train_delta_update(args, model_delta_all_ngh, device, lr, weight_decay)
        acc = gcn.evaluate_delta_update(model_delta_all_ngh, test_mask, device)
        gcn.train_delta_update(args, model_always_retrain, device, lr, weight_decay)
        acc_always_retrain = gcn.evaluate_delta_update(model_always_retrain, test_mask, device)
    elif model_name == 'graphsage':
        graphsage.train_delta_update(args, model_delta_all_ngh, device, fan_out, batch_size, lr,
                                     weight_decay)
        acc = graphsage.evaluate_delta_update(device, model_delta_all_ngh, test_mask, batch_size)
        graphsage.train_delta_update(args, model_always_retrain, device, fan_out, batch_size, lr,
                                     weight_decay)
        acc_always_retrain = graphsage.evaluate_delta_update(device, model_always_retrain,
                                                             test_mask, batch_size)
    elif model_name == 'gat':
        gat.train_delta_update(args, model_delta_all_ngh, device, lr, weight_decay)
        acc = gat.evaluate_delta_update(model_delta_all_ngh, test_mask, device)
        gat.train_delta_update(args, model_always_retrain, device, lr, weight_decay)
        acc_always_retrain = gat.evaluate_delta_update(model_always_retrain, test_mask, device)
    elif model_name == 'gin':
        gin.train_delta_update(args, model_delta_all_ngh, device, lr, weight_decay)
        acc = gin.evaluate_delta_update(model_delta_all_ngh, test_mask, device)
        gin.train_delta_update(args, model_always_retrain, device, lr, weight_decay)
        acc_always_retrain = gin.evaluate_delta_update(model_always_retrain, test_mask, device)

    accuracy = []
    # accuracy.append(
    #     [model.g.number_of_nodes(),
    #      model.g.number_of_edges(), acc * 100, acc_always_retrain * 100])

    print("Test accuracy {:.2%}".format(acc))

    # Evolve graph
    print(">>> Evaluate evolved graph: ")
    ## Record memory trace
    mem_access_q_delta_ngh = []  # For gen mem trace
    if dump_mem_trace_flag:
        trace_path_delta_ngh = '../../../results/mem_trace/' + args.dataset + '_delta_ngh_deg_' + str(
            deg_th) + '.txt'
        os.system('rm ' + trace_path_delta_ngh)  # Reset mem trace

    # while len(node_q) > 0:
    #     print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    #     print('Add nodes @ iter = {:d}'.format(i))
    #     print('node_q size: {:d}'.format(len(node_q)))
    #     # add_node_num = i * node_batch
    #     if node_batch < len(node_q):
    #         inserted_nodes = node_q[:node_batch]
    #         node_q = node_q[node_batch:]
    #     else:
    #         inserted_nodes = node_q
    #         node_q.clear()

    deg_th = int(args.deg_threshold)
    delta_neighbor = []

    acc_last = 0
    retrain_num = 0
    # Add new nodes
    for i in range(len(node_seq[1:])):
        print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('Add nodes @ iter = {:d}'.format(i + 1))
        inserted_nodes = node_q[node_seq[i]:node_seq[i + 1] - 1]
        print('Add nodes: {:d}, Total nodes: {:d}'.format(len(inserted_nodes), node_seq[i + 1]))

        # # Execute full retraining at the beginning
        # if i <= 0:
        #     model_delta_all_ngh.g = g_update.graph_evolve(args, inserted_nodes, g,
        #                                                   node_map_orig2evo, node_map_evo2orig,
        #                                                   model_delta_all_ngh.g)
        # else:
        #     model_delta_all_ngh.g = g_update.graph_evolve_delta_all_ngh(
        #         inserted_nodes, g_csr, g, node_map_orig2evo, node_map_evo2orig, n_layers,
        #         model_delta_all_ngh.g)

        model_delta_all_ngh.g = g_update.graph_evolve(args, g_struct_init_ratio,
                                                      g_snapshot_total_num, i + 1, inserted_nodes,
                                                      g, node_map_orig2evo, node_map_evo2orig,
                                                      model_delta_all_ngh.g)
        model_always_retrain.g = g_update.graph_evolve(args, g_struct_init_ratio,
                                                       g_snapshot_total_num, i + 1, inserted_nodes,
                                                       g, node_map_orig2evo, node_map_evo2orig,
                                                       model_always_retrain.g)

        # Get node index of added_nodes in evolve graph
        inserted_nodes_evo = g_update.get_nodes_reindex(node_map_orig2evo, inserted_nodes)
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

            if args.model == 'gcn':
                # graph_adj_t, v_sensitive, v_insensitive = util.gen_graph_adj_t(
                #     model_delta_all_ngh.g, inserted_nodes_evo, deg_th, n_layers)

                graph_adj_t_affected, v_sensitive_affected, v_insensitive_affected = util.gen_graph_adj_t_affected_by_trace(
                    args, g_struct_init_ratio, g_snapshot_total_num, i + 1, model_delta_all_ngh.g,
                    inserted_nodes_evo, deg_th, n_layers)

                # graph_adj_t_all_sensitive, v_sensitive_all_sensitive, v_insensitive_all_sensitive = util.gen_graph_adj_t_affected(
                #     model_delta_all_ngh.g, inserted_nodes_evo, 0, n_layers)

            if args.model == 'graphsage' or args.model == 'gin' or args.model == 'gat':
                edge_mask, nodes_high_deg, nodes_low_deg = util.gen_edge_mask(
                    model_delta_all_ngh.g, inserted_nodes_evo, deg_th, n_layers)
                model_delta_all_ngh.g.edata['edge_mask'] = edge_mask

            time_start = time.perf_counter()
            test_mask = model_delta_all_ngh.g.ndata['test_mask']
            retrain_epoch = 3
            # model_delta_all_ngh.reset_parameters()
            if i <= 0:
                model_delta_all_ngh.reset_parameters()
                if model_name == 'gcn':
                    # gcn.train_delta_update(args, model_delta_all_ngh, device, lr, weight_decay,
                    #                             nodes_high_deg, nodes_low_deg)
                    gcn.train_delta_update(args, model_delta_all_ngh, device, lr, weight_decay)
                    # acc = gcn.evaluate_delta_update(model_delta_all_ngh, test_mask, device,
                    #                                 graph_adj_t)
                elif model_name == 'graphsage':
                    graphsage.train_delta_edge_masked(args, model_delta_all_ngh, device, fan_out,
                                                      batch_size, lr, weight_decay)
                    acc = graphsage.evaluate_delta_edge_masked(device, model_delta_all_ngh,
                                                               test_mask, batch_size,
                                                               nodes_high_deg, nodes_low_deg)
                elif model_name == 'gat':
                    gat.train_delta_edge_masked(args, model_delta_all_ngh, device, lr, weight_decay,
                                                nodes_high_deg, nodes_low_deg)
                    acc = gat.evaluate_delta_edge_masked(model_delta_all_ngh, test_mask, device,
                                                         nodes_high_deg, nodes_low_deg)
                elif model_name == 'gin':
                    gin.train_delta_edge_masked(args, model_delta_all_ngh, device, lr, weight_decay)
                    acc = gin.evaluate_delta_edge_masked(model_delta_all_ngh, test_mask, device)
            else:
                if model_name == 'gcn':
                    model_always_retrain.reset_parameters()
                    # gcn.train_delta_update(args, model_always_retrain, device, lr, weight_decay,
                    #                        graph_adj_t_affected, v_sensitive_affected,
                    #                        v_insensitive_affected)
                    # acc_always_retrain = gcn.evaluate_delta_update(model_always_retrain, test_mask,
                    #                                                device, graph_adj_t_affected,
                    #                                                v_sensitive_affected,
                    #                                                v_insensitive_affected)
                    gcn.train_delta_update(args, model_always_retrain, device, lr, weight_decay)
                    acc_always_retrain = gcn.evaluate_delta_update(model_always_retrain, test_mask,
                                                                   device)

                    if args.periodic_retrain:
                        period = 3
                        if i % period == 0:
                            retrain_num += 1
                            print('\n>> Evoke a periodic retraining')
                            model_delta_all_ngh.reset_parameters()
                            gcn.train_delta_update(args, model_delta_all_ngh, device, lr,
                                                   weight_decay, graph_adj_t_affected,
                                                   v_sensitive_affected, v_insensitive_affected)
                    elif args.adaptive_retrain:
                        acc_pre = gcn.evaluate_delta_update(model_delta_all_ngh, test_mask, device,
                                                            graph_adj_t_affected,
                                                            v_sensitive_affected,
                                                            v_insensitive_affected)
                        acc_th = 0.03
                        # if acc_last - acc_pre > acc_th:
                        if acc_always_retrain - acc_pre > acc_th:
                            retrain_num += 1
                            print('\n>> Evoke an adaptive retraining')
                            model_delta_all_ngh.reset_parameters()
                            gcn.train_delta_update(args, model_delta_all_ngh, device, lr,
                                                   weight_decay, graph_adj_t_affected,
                                                   v_sensitive_affected, v_insensitive_affected)

                    # if (i + 1) % retrain_epoch == 0 or i == 0:
                    # gcn.train_delta_update(args, model_delta_all_ngh, device, lr, weight_decay,
                    #                        graph_adj_t_affected, v_sensitive, v_insensitive)
                    # gcn.train_delta_update(args, model_delta_all_ngh, device, lr, weight_decay,
                    #                        graph_adj_t_affected, v_sensitive_affected,
                    #                        v_insensitive_affected)
                    # acc = gcn.evaluate_delta_update(model_delta_all_ngh, test_mask, device,
                    #                                 graph_adj_t_affected, v_sensitive,
                    #                                 v_insensitive)
                    # acc = gcn.evaluate_delta_update(model_delta_all_ngh, test_mask, device)

                    # acc = gcn.evaluate_delta_update(model_delta_all_ngh, test_mask, device)
                    # print('>> &*&*&* Golden: {:.6f}'.format(acc))

                    # acc = gcn.evaluate_delta_update(model_delta_all_ngh, test_mask, device,
                    #                                 graph_adj_t, v_sensitive, v_insensitive)
                    # print('>> &*&*&* All_adj: {:.6f}'.format(acc))

                    # acc = gcn.evaluate_delta_update(model_delta_all_ngh, test_mask, device,
                    #                                 graph_adj_t_all_sensitive,
                    #                                 v_sensitive_all_sensitive,
                    #                                 v_insensitive_all_sensitive)
                    # print('>> Acc_all_sensitive: {:.6f}, v_sen: {:d}, v_insen: {:d}'.format(
                    #     acc, len(v_sensitive_all_sensitive), len(v_insensitive_all_sensitive)))
                    # acc = gcn.evaluate_delta_update(model_delta_all_ngh, test_mask, device,
                    #                                 graph_adj_t_affected, v_sensitive_affected,
                    #                                 v_insensitive_affected)
                    # print('>> Acc_affected: {:.6f}, v_sen: {:d}, v_insen: {:d}'.format(
                    #     acc, len(v_sensitive_affected), len(v_insensitive_affected)))

                    acc = gcn.evaluate_delta_update(model_delta_all_ngh, test_mask, device,
                                                    graph_adj_t_affected, v_sensitive_affected,
                                                    v_insensitive_affected)
                    # print('>> Acc: {:.2%}, v_sen: {:d}, v_insen: {:d}'.format(
                    #     acc, len(v_sensitive_affected), len(v_insensitive_affected)))

                    # acc_last = acc
                    # os.system("pause")

                elif model_name == 'graphsage':
                    # if (i + 1) % retrain_epoch == 0 or i == 0:
                    graphsage.train_delta_edge_masked(args, model_delta_all_ngh, device, fan_out,
                                                      batch_size, lr, weight_decay, nodes_high_deg,
                                                      nodes_low_deg)
                    acc = graphsage.evaluate_delta_edge_masked(device, model_delta_all_ngh,
                                                               test_mask, batch_size,
                                                               nodes_high_deg, nodes_low_deg)
                elif model_name == 'gat':
                    # if (i + 1) % retrain_epoch == 0 or i == 0:
                    gat.train_delta_edge_masked(args, model_delta_all_ngh, device, lr, weight_decay,
                                                nodes_high_deg, nodes_low_deg)
                    acc = gat.evaluate_delta_edge_masked(model_delta_all_ngh, test_mask, device,
                                                         nodes_high_deg, nodes_low_deg)
                elif model_name == 'gin':
                    # if (i + 1) % retrain_epoch == 0 or i == 0:
                    gin.train_delta_edge_masked(args, model_delta_all_ngh, device, lr, weight_decay,
                                                nodes_high_deg, nodes_low_deg)
                    acc = gin.evaluate_delta_edge_masked(model_delta_all_ngh, test_mask, device,
                                                         nodes_high_deg, nodes_low_deg)

            time_snapshot = time.perf_counter() - time_start
            print('>> Snapshot {:d} execute time: {}'.format(i + 1,
                                                             util.time_format(time_snapshot)))

            print('Acc_retrain: {:.2%}; Acc_always_retrain: {:.2%}'.format(acc, acc_always_retrain))

            accuracy.append([
                model.g.number_of_nodes(),
                model.g.number_of_edges(), acc * 100, acc_always_retrain * 100
            ])

        i += 1

        deg_th = str(args.deg_threshold)
        # Dump log
        if dump_accuracy_flag:
            np.savetxt('../../../results/accuracy/' + args.dataset + '_' + args.model +
                       '_evo_delta_' + deg_th + '.txt',
                       accuracy,
                       fmt='%d, %d, %.2f, %.2f')
        if dump_node_access_flag:
            np.savetxt('../../../results/node_access/' + args.dataset + '_' + args.model +
                       '_evo_delta_deg_' + deg_th + '.txt',
                       delta_neighbor,
                       fmt='%d, %d')

    # plot.plt_edge_epoch()
    # plot.plt_edge_epoch(edge_epoch, result)

    print('\n>> Task {:s} execution time: {}'.format(
        args.dataset, util.time_format(time.perf_counter() - Task_time_start)))

    for i in range(len(accuracy)):
        print('{:d}\t{:.2f}  {:.2f}'.format(i, accuracy[i][2], accuracy[i][3]))

    if args.adaptive_retrain:
        print('Adaptive retrain: {:d} times'.format(retrain_num))
    elif args.periodic_retrain:
        print('Periodic retrain: {:d} times'.format(retrain_num))


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
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--n-epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--self-loop",
                        default=True,
                        action='store_true',
                        help="graph self-loop (default=False)")
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
    parser.add_argument("--adaptive-retrain",
                        type=bool,
                        default=0,
                        help="Adaptive retrain based on acc degradation")
    parser.add_argument("--periodic-retrain",
                        type=bool,
                        default=0,
                        help="Periodic retrain with snapshots")
    args = parser.parse_args()

    args.model = 'gcn'
    # args.model = 'graphsage'
    # args.model = 'gat'
    # args.model = 'gin'

    # args.dataset = 'cora'
    # args.dataset = 'citeseer'
    args.dataset = 'ogbn-arxiv'
    # args.dataset = 'ogbn-mag'

    args.n_epochs = 100
    # args.deg_threshold = [100, 0]
    args.deg_threshold = 20
    args.gpu = 0

    args.adaptive_retrain = 1
    args.periodic_retrain = 0

    dump_accuracy_flag = 1
    dump_mem_trace_flag = 0
    dump_node_access_flag = 0

    print('\n************ {:s} ************'.format(args.dataset))
    print(args)

    main(args)
