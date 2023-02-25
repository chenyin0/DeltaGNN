import time
import uuid
import random
import argparse
import gc
import torch
import resource
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ogb.nodeproppred import Evaluator
from utils import SimpleDataset
# from model import ClassMLP
from model.gcn_t import GCN_t
import model.gcn_t as gcn_t
from utils import *
from glob import glob
import copy as cp
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


def main():
    # import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    parser = argparse.ArgumentParser()
    # Dataset and Algorithom
    parser.add_argument('--seed', type=int, default=20159, help='random seed..')
    parser.add_argument('--dataset', default='papers100M', help='dateset.')
    # Algorithm parameters
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha.')
    parser.add_argument('--rmax', type=float, default=1e-7, help='threshold.')
    # Learining parameters
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay.')
    parser.add_argument('--layer', type=int, default=3, help='number of layers.')
    parser.add_argument('--hidden', type=int, default=2048, help='hidden dimensions.')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate.')
    parser.add_argument('--bias', default='none', help='bias.')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs.')
    parser.add_argument('--batch_size', type=int, default=10000, help='batch size.')
    parser.add_argument('--patience', type=int, default=50, help='patience.')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu')
    args = parser.parse_args()

    args.dataset = 'arxiv'
    args.layer = 2
    args.hidden = 256
    args.alpha = 0.1
    args.dropout = 0.3
    args.epochs = 2
    args.gpu = 0
    args.batch_size = 128

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        gpu_id = args.gpu

    device = torch.device("cuda:" + str(gpu_id) if cuda else "cpu")
    # device = args.gpu
    # device = torch.cuda.set_device(0)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print("--------------------------")
    print(args)
    checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'

    # features, train_labels, val_labels, test_labels, train_idx, val_idx, test_idx, edge_idx_init, labels = load_ogb_init(
    #     args.dataset, args.alpha, args.rmax)  ##
    features, labels, train_idx, val_idx, test_idx, n_classes, edge_idx_init = load_ogb_init(
        args.dataset, args.alpha, args.rmax)  ##

    in_feats = features.shape[-1]
    n_hidden = args.hidden
    # n_classes = int(max(train_labels.max(), val_labels.max(), test_labels.max())) + 1
    n_layers = args.layer
    dropout = args.dropout

    # model_wo_retrain = GCN_t(in_feats, n_hidden, n_classes, n_layers, dropout).to(device)
    # model_retrain = GCN_t(in_feats, n_hidden, n_classes, n_layers, dropout).to(device)
    # model_delta = GCN_t(in_feats, n_hidden, n_classes, n_layers, dropout).to(device)

    # adj_t = adj.to_symmetric().to(device)
    # adj_t_wo_retrain = cp.deepcopy(adj_t).to(device)
    # adj_t_retrain = cp.deepcopy(adj_t).to(device)
    # adj_t_delta = cp.deepcopy(adj_t).to(device)

    model_wo_retrain = GCN_t(in_feats, n_hidden, n_classes, n_layers, dropout).cuda(device)
    model_retrain = GCN_t(in_feats, n_hidden, n_classes, n_layers, dropout).cuda(device)
    model_delta = GCN_t(in_feats, n_hidden, n_classes, n_layers, dropout).cuda(device)

    # adj_t = adj.to_symmetric().cuda(device)
    # adj_t_wo_retrain = cp.deepcopy(adj_t).cuda(device)
    # adj_t_retrain = cp.deepcopy(adj_t).cuda(device)
    # adj_t_delta = cp.deepcopy(adj_t).cuda(device)

    # Training for the initial snapshot
    num_sampling = [-1, -1]
    prepare_to_train(model_wo_retrain, edge_idx_init, features, train_idx, val_idx, test_idx,
                     labels, num_sampling, args, device, checkpt_file)
    # prepare_to_train(model_retrain, edge_idx_init, features, train_idx, val_idx, test_idx,
    #                  train_labels, val_labels, test_labels, args, device, checkpt_file)
    # prepare_to_train(model_delta, edge_idx_init, features, train_idx, val_idx, test_idx,
    #                  train_labels, val_labels, test_labels, args, device, checkpt_file)

    print('------------------ update -------------------')
    snapList = [f for f in glob('./data/' + args.dataset + '/*Edgeupdate_snap*.txt')]
    print('number of snapshots: ', len(snapList))
    for i in range(len(snapList)):
        # py_alg.snapshot_operation(
        #     'data/' + args.dataset + '/' + args.dataset + '_Edgeupdate_snap' + str(i + 1) + '.txt',
        #     args.rmax, args.alpha, features)

        # Update edge_index according to inserted edges

        # prepare_to_train(features, train_idx, val_idx, test_idx, train_labels, val_labels,
        #                  test_labels, args, checkpt_file)
        prepare_to_train(model_wo_retrain, adj_t_wo_retrain, features, train_idx, val_idx, test_idx,
                         train_labels, val_labels, test_labels, args, device, checkpt_file)
        prepare_to_train(model_retrain, adj_t_retrain, features, train_idx, val_idx, test_idx,
                         train_labels, val_labels, test_labels, args, device, checkpt_file)
        prepare_to_train(model_delta, adj_t_delta, features, train_idx, val_idx, test_idx,
                         train_labels, val_labels, test_labels, args, device, checkpt_file)


def train(model, device, train_loader, optimizer):
    model.train()

    time_epoch = 0
    loss_list = []
    for step, (x, y) in enumerate(train_loader):
        t_st = time.time()
        x, y = x.cuda(device), y.cuda(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y.squeeze(1))
        loss.backward()
        optimizer.step()
        time_epoch += (time.time() - t_st)
        loss_list.append(loss.item())
    return np.mean(loss_list), time_epoch


@torch.no_grad()
def validate(model, device, loader, evaluator):
    model.eval()
    y_pred, y_true = [], []
    for step, (x, y) in enumerate(loader):
        x = x.cuda(device)
        out = model(x)
        y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_true.append(y)
    return evaluator.eval({
        "y_true": torch.cat(y_true, dim=0),
        "y_pred": torch.cat(y_pred, dim=0),
    })['acc']


@torch.no_grad()
def test(model, device, loader, evaluator, checkpt_file):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    y_pred, y_true = [], []
    for step, (x, y) in enumerate(loader):
        x = x.cuda(device)
        out = model(x)
        y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_true.append(y)
    return evaluator.eval({
        "y_true": torch.cat(y_true, dim=0),
        "y_pred": torch.cat(y_pred, dim=0),
    })['acc']


def prepare_to_train(model, edge_index, features, train_idx, val_idx, test_idx, labels,
                     num_sampling, args, device, checkpt_file):
    features = torch.FloatTensor(features)
    features_train = features[train_idx]
    features_val = features[val_idx]
    features_test = features[test_idx]
    data = Data(x=features, edge_index=edge_index, y=labels)
    del features
    gc.collect()

    # label_dim = int(max(train_labels.max(), val_labels.max(), test_labels.max())) + 1
    # train_dataset = SimpleDataset(features_train, train_labels)
    # valid_dataset = SimpleDataset(features_val, val_labels)
    # test_dataset = SimpleDataset(features_test, test_labels)

    # train_dataset = SimpleDataset(x=features_train, edge_index=edge_index, y=train_labels)
    # valid_dataset = SimpleDataset(x=features_val, edge_index=edge_index, y=val_labels)
    # test_dataset = SimpleDataset(x=features_test, edge_index=edge_index, y=test_labels)

    # train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # train_idx = torch.arange(1, data.x.shape[0])
    train_loader = NeighborLoader(data,
                                  input_nodes=train_idx,
                                  num_neighbors=num_sampling,
                                  shuffle=True,
                                  batch_size=args.batch_size)
    valid_loader = NeighborLoader(data,
                                  input_nodes=val_idx,
                                  num_neighbors=num_sampling,
                                  shuffle=False,
                                  batch_size=args.batch_size)
    test_loader = NeighborLoader(data,
                                 input_nodes=test_idx,
                                 num_neighbors=num_sampling,
                                 shuffle=False,
                                 batch_size=args.batch_size)

    # model = ClassMLP(features_train.size(-1), args.hidden, label_dim, args.layer,
    #                  args.dropout).cuda(args.dev)
    # evaluator = Evaluator(name='ogbn-papers100M')
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    bad_counter = 0
    best = 0
    best_epoch = 0
    train_time = 0
    model.reset_parameters()
    print("--------------------------")
    print("Training...")
    for epoch in range(args.epochs):
        loss_tra, train_ep = gcn_t.train(model, device, train_loader, args.lr, args.weight_decay)
        t_st = time.time()
        f1_val = gcn_t.validate(model, device, valid_loader)
        train_time += train_ep
        if (epoch + 1) % 20 == 0:
            print('Epoch:{:02d}, Train_loss:{:.3f}, Valid_acc:{:.2f}%, Time_cost:{:.3f}/{:.3f}'.
                  format(epoch + 1, loss_tra, 100 * f1_val, train_ep, train_time))
            print('Remove print')
        if f1_val > best:
            best = f1_val
            best_epoch = epoch + 1
            t_st = time.time()
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == args.patience:
            break

    test_acc = gcn_t.test(model, device, test_loader, checkpt_file)
    print('Train cost: {:.2f}s'.format(train_time))
    print('Load {}th epoch'.format(best_epoch))
    print('Test accuracy:{:.2f}%'.format(100 * test_acc))


if __name__ == '__main__':
    main()
