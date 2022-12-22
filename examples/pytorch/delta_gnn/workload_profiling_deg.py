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

import matplotlib.pyplot as plt


def gen_degree_dist(g):
    """
    Gen node distribution with degree
    """

    deg_list = []
    max_deg = 0
    degs = g.in_degrees(g.nodes())
    deg_list = degs.numpy().tolist()
    max_deg = max(deg_list)

    deg_distribution = [0 for i in range(max_deg + 1)]
    for i in deg_list:
        deg_distribution[i] += 1

    return deg_distribution


def gen_workload_accumulate_by_deg(deg_dist, op_per_vertice):
    workload_sum = 0
    workload_accumulated_dist = [0 for i in range(len(deg_dist))]
    for i in range(len(deg_dist)):
        workload = i * deg_dist[i] * op_per_vertice
        workload_sum += workload
        workload_accumulated_dist[i] = workload_sum

    workload_accumulated_dist_norm = [
        round((i * 100 / workload_sum), 2) for i in workload_accumulated_dist
    ]
    return workload_accumulated_dist_norm


def gen_deg_accumulated_dist(deg_dist):
    accmulated_node_num = 0
    node_accumulated_dist = []
    for node_num in deg_dist:
        accmulated_node_num += node_num
        node_accumulated_dist.append(accmulated_node_num)

    node_accumulated_dist_norm = [
        round((i * 100 / accmulated_node_num), 2) for i in node_accumulated_dist
    ]
    return node_accumulated_dist_norm


def plot(workload_accmulated_dist, dataset_name):
    # for i in range(len(workload_accmulated_dist)):
    #     workload_accmulated_dist[i] = min(workload_accmulated_dist[i], 1000)

    plt.bar(range(len(workload_accmulated_dist)), workload_accmulated_dist)
    plt.show()

    plt.savefig('../../../figure/workload_accmulated_dist_' + dataset_name + '.pdf',
                dpi=600,
                bbox_inches="tight",
                pad_inches=0)


def main(args):
    # Overall task execution time
    Task_time_start = time.perf_counter()

    # Load GNN model parameter
    model_name = args.model
    if model_name == 'gcn':
        path = os.getcwd()
        print(path)
        with open('./gcn_para.json', 'r') as f:
            para = json.load(f)
            n_hidden = para['--n-hidden']
            n_layers = para['--n-layers']
            lr = para['--lr']
            weight_decay = para['--weight-decay']
            dropout = para['--dropout']
    elif model_name == 'graphsage':
        with open('./graphsage_para.json', 'r') as f:
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
        with open('./gat_para.json', 'r') as f:
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
    elif args.dataset == 'amazon_comp':
        dataset = AmazonCoBuyComputerDataset(raw_dir='../../../dataset')
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.dataset == 'ogbn-mag':
        g = preprocess.ogbn_mag_preprocess(dataset)
    else:
        g = dataset[0]

    deg_dist = gen_degree_dist(g)
    workload_accumulated_dist_norm = gen_workload_accumulate_by_deg(deg_dist, 10)
    deg_accumulated_dist = gen_deg_accumulated_dist(deg_dist)
    if args.dataset == 'cora' or args.dataset == 'citeseer':
        # deg_list = [1, 2, 5, 10, 15, 20, 30, 50, 99]
        deg_list = [i for i in range(20)]
    elif args.dataset == 'ogbn-arxiv' or args.dataset == 'ogbn-mag':
        deg_list = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200, 300, 500, 800, 1000, 2000]
        # deg_list = [i for i in range(20)]

    for deg in deg_list:
        print(deg, workload_accumulated_dist_norm[deg + 1], '   ', deg_accumulated_dist[deg + 1])

    # plot(workload_accumulated_dist_norm, args.dataset)

    # plot.plt_edge_epoch()
    # plot.plt_edge_epoch(edge_epoch, result)

    print('\n>> Task {:s} execution time: {}'.format(
        args.dataset, util.time_format(time.perf_counter() - Task_time_start)))


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
    # args.model = 'gat'

    # args.dataset = 'cora'
    # args.dataset = 'citeseer'
    # args.dataset = 'ogbn-arxiv'
    args.dataset = 'ogbn-mag'

    # args.n_epochs = 200
    # args.gpu = 0
    # args.mode = 'mixed'

    # dump_accuracy_flag = 1
    # dump_mem_trace_flag = 0
    # dump_node_access_flag = 0

    print('\n************ {:s} ************'.format(args.dataset))
    print(args)

    main(args)
