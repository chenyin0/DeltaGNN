import argparse
from fcntl import DN_DELETE
import time
import xxlimited
import numpy as np
import torch as th
import torch.nn.functional as F

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset, AmazonCoBuyComputerDataset
from dgl.data import AsNodePredDataset
from dgl import AddSelfLoop
from ogb.nodeproppred import DglNodePropPredDataset
import preprocess

import util
import g_update
import time
import datetime
import random
import copy as cp

import os
import json
import pathlib


def main(args):
    # Overall task execution time
    Task_time_start = time.perf_counter()

    # # Load GNN model parameter
    model_name = args.model

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
    # elif args.dataset == 'ogbn-mag':
    #     dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-mag', root='./dataset'))
    elif args.dataset == 'amazon_comp':
        dataset = AmazonCoBuyComputerDataset(raw_dir='./dataset')
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.dataset == 'ogbn-mag':
        g = preprocess.ogbn_mag_preprocess(dataset)
    else:
        g = dataset[0]

    arch = args.arch
    if arch == 'i-gcn':
        pass  # Reorder g

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        gpu_id = args.gpu

    device = th.device("cuda:" + str(gpu_id) if cuda else "cpu")
    mode = args.mode

    ##
    """ Construct evolve graph """
    g_csr = g.adj_sparse('csr')
    """ Traverse to get graph evolving snapshot """
    nodes_q = []
    file_ = pathlib.Path('./dataset/' + args.dataset + '_evo_delta_seq.txt')
    if file_.exists():
        f = open(file_, "r")
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')  # Delete '\n'
            nodes_q.append(int(line))
    else:
        if args.dataset == 'cora' or args.dataset == 'citeseer':
            # root_node_q = util.gen_root_node_queue(g)
            # node_q = util.bfs_traverse(g_csr, root_node_q)
            nodes_q = g.nodes().numpy().tolist()
        elif args.dataset == 'ogbn-arxiv' or args.dataset == 'ogbn-mag':
            nodes_q = util.sort_node_by_timestamp('./dataset/' + args.dataset + '_node_year.csv')

        with open(file_, 'w') as f:
            for i in nodes_q:
                f.write(str(i) + '\n')

    # Gen node_seq
    g_struct_init_ratio = 0.5
    node_seq = util.gen_snapshot(g_struct_init_ratio, 10, g.number_of_nodes())

    # Gen node_mapping from g_orig to g_evo, for DGL compels consecutive node id
    node_map_orig2evo = dict()
    node_map_evo2orig = dict()

    init_nodes = nodes_q[:node_seq[0]]
    g_evo = g_update.graph_struct_init(args, g_struct_init_ratio, init_nodes, g, node_map_orig2evo,
                                       node_map_evo2orig)

    # Evolve graph
    print(">>> Accuracy on evolove graph: ")
    ## Record memory trace
    """
    Memory trace format
    root_node_id  #ngh  ngh_node_id  #ngh-in-degree  #ngh-out-degree
    """
    mem_trace = []
    deg_th = args.deg_threshold
    ISOTIMEFORMAT = '%m%d_%H%M'
    theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    theTime = str(theTime)
    file_path = './results/mem_trace/' + args.dataset + '_' + arch + '_' + str(
        deg_th) + '_' + theTime + '.txt'
    os.system('rm ' + file_path)  # Reset mem trace
    # with open("file_path", mode='w') as f:
    #     print('New file!')

    n_layer = 2
    # Add new nodes
    for i in range(len(node_seq[1:])):
        print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('Add nodes @ iter = {:d}'.format(i + 1))
        inserted_nodes = nodes_q[node_seq[i]:node_seq[i + 1] - 1]
        print('Add nodes: {:d}, Total nodes: {:d}'.format(len(inserted_nodes),
                                                          g_evo.number_of_nodes()))
        iter_time_start = time.perf_counter()

        g_evo = g_update.graph_evolve(inserted_nodes, g_csr, g, node_map_orig2evo,
                                      node_map_evo2orig, n_layer, g_evo)

        # Get node index of added_nodes in evolve graph
        inserted_nodes_evo = g_update.get_nodes_reindex(node_map_orig2evo, inserted_nodes)
        visited = [0 for i in range(g_evo.number_of_nodes())]
        # affected_nghs = util.get_dst_nghs_multi_layers(g_evo, inserted_nodes_evo, n_layer)

        if arch == 'delta-gnn-opt':
            affected_nodes, dict_map = util.get_dst_nghs_multi_layers_with_mapping(
                g_evo, inserted_nodes_evo, n_layer)
            nodes_q_pair = []  # Format: [node_id, layer_id]
            for i in affected_nodes:
                nodes_q_pair.append([i, 0])  # For layer-0
            while len(nodes_q_pair) != 0:
                v = nodes_q_pair.pop(0)
                v_layer = v[1]
                if v_layer < n_layer:
                    v_id = v[0]
                    if g_evo.in_degrees(v_id) > deg_th:
                        nghs = g_evo.predecessors(v_id).numpy().tolist()
                        ngh_num = len(nghs)
                        for ngh in nghs and visited[i] != 1:
                            nodes_q_pair.insert(0, [i, v_layer + 1])  # Insert stack front
                            trace_item = [
                                v_id, ngh_num, ngh,
                                g_evo.in_degrees(ngh),
                                g_evo.out_degrees(ngh)
                            ]
                            mem_trace.append(trace_item)
                            visited[i] = 1
                    else:
                        root_v = dict_map[v_id]
                        trace_item = [
                            root_v,
                            g_evo.out_degrees(root_v), v_id,
                            g_evo.in_degrees(v_id),
                            g_evo.out_degrees(v_id)
                        ]
                        mem_trace.append(trace_item)
        elif arch == 'delta-gnn':
            affected_nodes, dict_map = util.get_dst_nghs_multi_layers_with_mapping(
                g_evo, inserted_nodes_evo, n_layer)
            vertex_q = affected_nodes
            ngh_per_layer = []
            for i in range(n_layer):
                for v in vertex_q:
                    if g_evo.out_degree(v) > deg_th:
                        nghs = g_evo.predecessors(v).numpy().tolist()
                        ngh_per_layer.extend(nghs)
                        ngh_num = len(nghs)
                        for ngh in nghs and visited[ngh] != 1:
                            trace_item = [
                                v, ngh_num, ngh,
                                g_evo.in_degrees(ngh),
                                g_evo.out_degrees(ngh)
                            ]
                            mem_trace.append(trace_item)
                            visited[ngh] = 1
                    else:
                        root_v = dict_map[v]
                        trace_item = [
                            root_v,
                            g_evo.out_degrees(root_v), v,
                            g_evo.in_degrees(v),
                            g_evo.out_degrees(v)
                        ]
                        mem_trace.append(trace_item)
                vertex_q.clear()
                vertex_q.extend(ngh_per_layer)
                ngh_per_layer.clear()
        elif arch == 'regnn':
            # Reduce redundant access
            affected_nodes = util.get_dst_nghs_multi_layers(g_evo, inserted_nodes_evo, n_layer)
            vertex_q = affected_nodes
            ngh_per_layer = []
            # nghs_total = []
            for i in range(n_layer):
                for v in vertex_q:
                    nghs = g_evo.predecessors(v).numpy().tolist()
                    ngh_per_layer.extend(nghs)
                    # nghs_total.extend(nghs)
                    ngh_num = len(nghs)
                    for ngh in nghs and visited[ngh] != 1:
                        trace_item = [
                            v, ngh_num, ngh,
                            g_evo.in_degrees(ngh),
                            g_evo.out_degrees(ngh)
                        ]
                        mem_trace.append(trace_item)
                        visited[ngh] = 1
                vertex_q.clear()
                vertex_q.extend(ngh_per_layer)
                ngh_per_layer.clear()
        else:
            affected_nodes = util.get_dst_nghs_multi_layers(g_evo, inserted_nodes_evo, n_layer)
            vertex_q = affected_nodes
            ngh_per_layer = []
            for i in range(n_layer):
                for v in vertex_q:
                    nghs = g_evo.predecessors(v).numpy().tolist()
                    ngh_per_layer.extend(nghs)
                    ngh_num = len(nghs)
                    for ngh in nghs:
                        trace_item = [
                            v, ngh_num, ngh,
                            g_evo.in_degrees(ngh),
                            g_evo.out_degrees(ngh)
                        ]
                        mem_trace.append(trace_item)
                vertex_q.clear()
                vertex_q.extend(ngh_per_layer)
                ngh_per_layer.clear()

        with open(file_path, mode='a+') as f:
            for line in mem_trace:
                for item in line:
                    f.write(str(item) + ' ')
                f.write('\n')

        mem_trace.clear()
        i += 1

        print('\n>> Iter: {:s} exe time: {}'.format(
            util.time_format(time.perf_counter() - iter_time_start)))

    print('\n>> Task {:s} on Arch {:s} execution time: {}'.format(
        args.dataset, args.arch, util.time_format(time.perf_counter() - Task_time_start)))


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
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Arch name ('hygcn', 'awb-gcn', 'i-gcn', 'regnn', 'delta-gnn', 'delta-gnn-opt').")
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

    args.arch = 'hygcn'
    # args.arch = 'awb-gcn'
    # args.arch = 'i-gcn'
    # args.arch = 'regnn'
    # args.arch = 'delta-gnn'
    # args.arch = 'delta-gnn-opt'

    args.dataset = 'cora'
    # args.dataset = 'citeseer'
    # args.dataset = 'ogbn-arxiv'
    # args.dataset = 'ogbn-mag'

    args.n_epochs = 200
    args.deg_threshold = 0
    args.gpu = 0

    dump_accuracy_flag = 1
    dump_mem_trace_flag = 0
    dump_node_access_flag = 0

    print('\n************ {:s} ************'.format(args.dataset))
    print(args)

    main(args)
