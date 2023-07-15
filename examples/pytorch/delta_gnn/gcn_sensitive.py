import argparse
# import time
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
# import dgl
# from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

# from model.gcn import GCN
#from gcn_mp import GCN
#from gcn_spmv import GCN

# import plt_graph
import random
# import math
# import util

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, WikiCS, GitHub, FacebookPagePage, Actor
import copy as cp

import plt.plt_graph as plt_graph


class GCN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True, normalize=not args.use_gdc)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True, normalize=not args.use_gdc)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def train(model, data):
    model.train()
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ],
                                 lr=args.lr)  # Only perform weight-decay on first convolution.
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)

    # train_mask_index = th.LongTensor(np.arange(0, out.shape[-1])).to(device)
    # train_mask = th.index_select(data.train_mask, 1, train_mask_index)
    dim = data.train_mask.dim()
    if dim > 1:
        train_mask = data.train_mask[:, 0]
    else:
        train_mask = data.train_mask
    # train_mask = th.chunk(data.train_mask, data.train_mask.shape[0], dim=dim)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)

    # accs = []
    # for mask in [data.train_mask, data.val_mask, data.test_mask]:
    #     accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    # return accs

    dim = data.test_mask.dim()
    if dim > 1:
        test_mask = data.test_mask[:, 0]
    else:
        test_mask = data.test_mask

    acc = int((pred[test_mask] == data.y[test_mask]).sum()) / int(test_mask.sum())
    return acc


def count_vertex_indegree(edge_index, vertex_num):
    v_indegree = np.zeros(vertex_num)
    in_degree = edge_index[0].numpy()
    # for v in range(vertex_num):
    #     v_indegree[v] = np.sum(in_degree == v)
    for v in in_degree:
        v_indegree[v] += 1

    v_indegree = th.Tensor(v_indegree).long()
    return v_indegree


def degree_distribute(in_degree):
    # Statistic graph typology
    # in_degree = g.in_degrees()
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


def add_noise_out_deg(data, degree_list, deg_begin, deg_end, intensity, node_ratio, vertex_num,
                      device):
    """
    node ratio: the proportion of the total nodes which added noise
    intensity: the proportion of the noise amplitude with the original feature value
    """
    print()
    # in_feats = g.ndata['feat'].shape[1]
    in_feats = data.x.shape[1]
    deg_end = min(deg_end, len(degree_list))  # Avoid over boundary
    num = round(vertex_num * node_ratio)

    node_list = degree_list[deg_begin:deg_end]
    node_list = [i for bin in node_list for i in bin]
    loc_list = range(len(node_list))
    print('>> Deg: [',
          deg_begin,
          ',',
          deg_end,
          '] ',
          ' Node num: ',
          len(loc_list),
          ' Selet node num: ',
          num,
          flush=True)

    noise = torch.rand(in_feats)
    # device = g.device
    noise = noise.to(device)
    noise = th.mul(noise, intensity)
    loc = random.sample(loc_list, min(num, len(loc_list)))
    # loc = random.sample(loc_list, num)
    for i in loc:
        node_id = node_list[i]
        data.x[node_id] = th.add(noise, data.x[node_id])


def add_noise_in_deg(data, degree_list, deg_begin, deg_end, intensity, node_ratio, vertex_num,
                     device):
    """
    node ratio: the proportion of the total nodes which added noise
    intensity: the proportion of the noise amplitude with the original feature value

    Note: intensity should be amortized by degree
    """
    print()
    # in_feats = g.ndata['feat'].shape[1]
    in_feats = data.x.shape[1]
    deg_end = min(deg_end, len(degree_list))  # Avoid over boundary
    num = round(vertex_num * node_ratio)

    node_list = degree_list[deg_begin:deg_end]
    node_list = [i for bin in node_list for i in bin]
    loc_list = range(len(node_list))
    print('>> Deg: [',
          deg_begin,
          ',',
          deg_end,
          '] ',
          ' Node num: ',
          len(loc_list),
          ' Selet node num: ',
          num,
          flush=True)

    noise = torch.rand(in_feats)
    # device = g.device
    noise = noise.to(device)
    noise = th.mul(noise, intensity / max(round(
        (deg_begin + deg_end) / 2), 1))  # Intensity is amortized by the indegree
    loc = random.sample(loc_list, min(num, len(loc_list)))
    # loc = random.sample(loc_list, num)
    for i in loc:
        node_id = node_list[i]
        data.x[node_id] = th.add(noise, data.x[node_id])


def main(args):
    # Load and preprocess dataset
    if args.dataset == 'Cora' or args.dataset == 'CiteSeer':
        dataset = Planetoid('./dataset', args.dataset)
    if args.dataset == 'WikiCS':
        dataset = WikiCS('./dataset')
    if args.dataset == 'GitHub':
        dataset = GitHub('./dataset' + args.dataset)
    if args.dataset == 'FacebookPagePage':
        dataset = FacebookPagePage('./dataset' + args.dataset)
    if args.dataset == 'Actor':
        dataset = Actor('./dataset/' + args.dataset)

    data = dataset[0]

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        gpu_id = args.gpu
    device = torch.device("cuda:" + str(gpu_id) if cuda else "cpu")

    # degree_list = degree_distribute(g)
    vertex_num = data.num_nodes
    in_degree = count_vertex_indegree(data.edge_index, vertex_num)
    in_degree_max = th.max(in_degree)
    degree_list = degree_distribute(in_degree)

    # # Plot vertex degree distribution
    # plt_graph.plot_degree_distribution(vertex_num, degree_list, args.dataset)

    task_round = args.n_round
    acc_task = [0] * task_round
    for task_round_id in range(task_round):

        data_t = cp.deepcopy(data)
        model = GCN(dataset.num_features, args.n_hidden, dataset.num_classes)
        model, data_t = model.to(device), data_t.to(device)

        for epoch in range(1, args.n_epochs + 1):
            train(model, data_t)

        deg_begin = args.deg_begin
        deg_end = args.deg_end
        noise_intensity = args.noise_intensity
        node_ratio = args.node_ratio
        if args.degree_type == 'indeg':
            add_noise_in_deg(data_t, degree_list, deg_begin, deg_end, noise_intensity * 4,
                             node_ratio, vertex_num, device)
        elif args.degree_type == 'outdeg':
            add_noise_out_deg(data_t, degree_list, deg_begin, deg_end, noise_intensity * 2,
                              node_ratio, vertex_num, device)

        print()
        acc = test(model, data_t)
        print("Test accuracy {:.2%}".format(acc))
        acc_task[task_round_id] = acc

    acc_total = 0
    for i in acc_task:
        acc_total += i

    # print()
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
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=20, help="number of training epochs")
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
    parser.add_argument("--n-round",
                        type=int,
                        default=5,
                        help="output the average accuracy of n-round execution")
    parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
    parser.add_argument('--degree-type',
                        type=str,
                        default='indeg',
                        help='Indentify indegree or outdegree')
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()

    args.dataset = 'Cora'
    # args.dataset = 'CiteSeer'
    # args.dataset = 'WikiCS'
    # args.dataset = 'Actor'
    # args.dataset = 'FacebookPagePage'
    # args.dataset = 'GitHub'

    args.degree_type = 'outdeg'
    args.gpu = 1
    args.noise_intensity = 0.2
    args.node_ratio = 0.02
    args.deg_begin = 2
    args.deg_end = 3

    # print(args)
    main(args)
