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
import random

from gcn import GCN
from gcn import GCN_delta

# from torchviz import make_dot

# from torch.utils.tensorboard import SummaryWriter

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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        # acc = evaluate(model, features, labels, val_mask)
        # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
        #                                     acc, n_edges / np.mean(dur) / 1000))


def train_delta_edge_masked(model,
                            features,
                            n_edges,
                            train_mask,
                            val_mask,
                            labels,
                            loss_fcn,
                            optimizer,
                            ngh_high_deg=None,
                            ngh_low_deg=None):

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features, ngh_high_deg, ngh_low_deg)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        # print('\n>> Grad', [x.grad for x in optimizer.param_groups[0]['params']])

        # print()
        # for name, parameters in model.named_parameters():
        #     print('\n' + name, ':', parameters)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate_delta_edge_masked(model, features, labels, val_mask, ngh_high_deg,
                                         ngh_low_deg)
        # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(), acc,
        #                                     n_edges / np.mean(dur) / 1000))

    # Update embedding
    model.embedding = torch.nn.Parameter(logits)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def evaluate_delta_edge_masked(model,
                               features,
                               labels,
                               mask,
                               nodes_high_deg=None,
                               nodes_low_deg=None):
    r"""
    Update feature of updated nodes according to node degree

    "model" should be GCN_delta
    """
    model.eval()
    with torch.no_grad():
        logits = model(features, nodes_high_deg, nodes_low_deg)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
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
        # g = g.int().to(args.gpu)

    # device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    device = th.device("cuda:0" if cuda else "cpu")

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
    n_classes = data.num_classes

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
        g_evo = g_evo.to(args.gpu)
    g_evo.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g_evo, in_feats, args.n_hidden, n_classes, args.n_layers, F.relu,
                args.dropout).to(device)
    model_delta_all_ngh = GCN_delta(g_evo, in_feats, args.n_hidden, n_classes, args.n_layers,
                                    F.relu, args.dropout).to(device)

    # for param in model.parameters():
    #     print(param)

    if cuda:
        model.cuda()
    else:
        model.cpu()

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_fcn_delta_all_ngh = torch.nn.CrossEntropyLoss()
    optimizer_delta_all_ngh = torch.optim.Adam(model_delta_all_ngh.parameters(),
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
    print("\n>>> Evaluate on evolved graph: ")

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
        inserted_nodes_evo = util.nodes_reindex(node_map_orig2evo, inserted_nodes)
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
            mem_access_q_delta_ngh = util.nodes_reindex(node_map_evo2orig, mem_access_q_delta_ngh)
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
                if i <= -1:
                    train_delta_edge_masked(model_delta_all_ngh, features,
                                            model_delta_all_ngh.g.number_of_edges(), train_mask,
                                            val_mask, labels, loss_fcn_delta_all_ngh,
                                            optimizer_delta_all_ngh)
                else:
                    train_delta_edge_masked(model_delta_all_ngh, features,
                                            model_delta_all_ngh.g.number_of_edges(), train_mask,
                                            val_mask, labels, loss_fcn_delta_all_ngh,
                                            optimizer_delta_all_ngh, list(ngh_high_deg.keys()),
                                            list(ngh_low_deg.keys()))

                time_delta_all_ngh_retrain = time.perf_counter() - time_start
                print('>> Epoch training time in delta ngh: {:.4}s'.format(
                    time_delta_all_ngh_retrain))

            if i <= 8:
                acc = evaluate(model_delta_all_ngh, features, labels, test_mask)
            else:
                acc = evaluate_delta_edge_masked(model_delta_all_ngh, features, labels, test_mask,
                                                 list(ngh_high_deg.keys()),
                                                 list(ngh_low_deg.keys()))

            # acc = evaluate(model_delta_all_ngh, features, labels, test_mask)
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
                        default=0,
                        help="degree threshold of neighbors nodes")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()

    args.dataset = 'cora'
    args.n_epochs = 200
    args.deg_threshold = 8
    args.gpu = -1
    args.n_layers = 0

    dump_accuracy_flag = 0
    dump_mem_trace_flag = 0
    dump_node_access_flag = 1

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
