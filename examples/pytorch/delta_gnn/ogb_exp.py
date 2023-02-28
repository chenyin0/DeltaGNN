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
from model.gcn_t import GCN, GCN_delta
import model.gcn_t as gcn
from utils import *
from glob import glob
import copy as cp
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import util


def main(args):
    # import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    Task_time_start = time.time()

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        gpu_id = args.gpu

    device = torch.device("cuda:" + str(gpu_id) if cuda else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print("--------------------------")
    print(args)
    print("--------------------------")
    print('Initial graph')
    checkpt_file_wo_retrain = 'pretrained/wo_retrain/' + uuid.uuid4().hex + '.pt'
    checkpt_file_retrain = 'pretrained/retrain/' + uuid.uuid4().hex + '.pt'
    checkpt_file_delta = 'pretrained/delta/' + uuid.uuid4().hex + '.pt'

    features, labels, train_idx, val_idx, test_idx, n_classes, edge_index_init = load_dataset_init(
        args.dataset)  ##

    in_feats = features.shape[-1]
    n_hidden = args.hidden
    n_layers = args.layer
    dropout = args.dropout

    model_wo_retrain = GCN(in_feats, n_hidden, n_classes, n_layers, dropout).cuda(device)
    model_retrain = GCN(in_feats, n_hidden, n_classes, n_layers, dropout).cuda(device)
    model_delta = GCN_delta(in_feats, n_hidden, n_classes, n_layers, dropout,
                            features.shape[0]).cuda(device)

    accuracy = []
    print('Edges_init: ', edge_index_init.shape)
    # Training for the initial snapshot
    num_nghs = [-1, -1]

    edge_index_evolved = edge_index_init.clone().detach()
    train_loader, valid_loader, test_loader = gen_dataloader(args, edge_index_evolved, train_idx,
                                                             val_idx, test_idx, features, labels,
                                                             num_nghs)
    # Gen edge_dict
    edge_dict = gen_edge_dict(edge_index_evolved)

    # Without retrain
    print('--- Model_wo_retrain:')
    # edge_index_wo_retrain = edge_idx_init.clone().detach()
    train(args, model_wo_retrain, train_loader, valid_loader, device, checkpt_file_wo_retrain)
    acc_wo_retrain = test(model_wo_retrain, test_loader, device, checkpt_file_wo_retrain)

    # Retrain
    print('--- Model_retrain:')
    # edge_index_retrain = edge_idx_init.clone().detach()
    train(args, model_retrain, train_loader, valid_loader, device, checkpt_file_retrain)
    acc_retrain = test(model_retrain, test_loader, device, checkpt_file_retrain)

    # Delta-retrain
    print('--- Model_retrain_delta:')
    edge_index_delta = edge_index_init.clone().detach()
    train_delta(args, model_delta, train_loader, valid_loader, device, checkpt_file_delta)
    acc_delta = test_delta(model_delta, test_loader, device, checkpt_file_delta)

    accuracy.append([0, acc_wo_retrain, acc_retrain, acc_delta])

    print('------------------ update -------------------')
    snapList = [f for f in glob('./data/' + args.dataset + '/*Edgeupdate_snap*.txt')]
    print('number of snapshots: ', len(snapList))
    for i in range(len(snapList)):
        print("\n--------------------------")
        print('Snapshot: {:d}'.format(i + 1))
        # Update edge_index according to inserted edges
        inserted_edge_index = load_updated_edges(args.dataset, i + 1)
        edge_index_evolved = insert_edges(edge_index_evolved, inserted_edge_index)
        train_loader, valid_loader, test_loader = gen_dataloader(args, edge_index_evolved,
                                                                 train_idx, val_idx, test_idx,
                                                                 features, labels, num_nghs)
        print('--- Model_wo_retrain:')
        # edge_index_wo_retrain = insert_edges(edge_index_wo_retrain, inserted_edge_index)
        print('Edges_wo_retrain: ', edge_index_evolved.shape)
        acc_wo_retrain = test(model_wo_retrain, test_loader, device, checkpt_file_wo_retrain)

        print('--- Model_retrain:')
        # edge_index_retrain = insert_edges(edge_index_retrain, inserted_edge_index)
        print('Edges_retrain: ', edge_index_evolved.shape)
        train(args, model_retrain, train_loader, valid_loader, device, checkpt_file_retrain)
        acc_retrain = test(model_retrain, test_loader, device, checkpt_file_retrain)

        print('--- Model_delta:')
        threshold = 200
        edge_dict, edge_index_delta, v_sen, v_insen = insert_edges_delta(
            edge_dict, inserted_edge_index, threshold, args.layer)
        print('fjapgwajhgi', len(v_sen), len(v_insen))
        train_loader_delta, valid_loader_delta, test_loader_delta = gen_dataloader_delta(
            args, edge_index_evolved, train_idx, val_idx, test_idx, features, labels, num_nghs)
        print('Edges_delta: ', edge_index_delta.shape)
        train_delta(args, model_delta, train_loader_delta, valid_loader_delta, device,
                    checkpt_file_delta, v_sen, v_insen)
        acc_delta = test_delta(model_delta, test_loader_delta, device, checkpt_file_delta, v_sen,
                               v_insen)
        # acc_delta = 0

        accuracy.append([i + 1, acc_wo_retrain, acc_retrain, acc_delta])

    for i in range(len(accuracy)):
        print('{:d}\t{:.2f}  {:.2f}  {:.2f}'.format(i, accuracy[i][1] * 100, accuracy[i][2] * 100,
                                                    accuracy[i][3] * 100))

    print('\n>> Task {:s} execution time: {}'.format(
        args.dataset, util.time_format(time.perf_counter() - Task_time_start)))


def gen_dataloader(args, edge_index, train_idx, val_idx, test_idx, features, labels, num_nghs):
    features = torch.FloatTensor(features)
    data = Data(x=features, edge_index=edge_index, y=labels)
    del features
    gc.collect()

    train_loader = NeighborLoader(data,
                                  input_nodes=train_idx,
                                  num_neighbors=num_nghs,
                                  shuffle=False,
                                  batch_size=args.batch_size)
    valid_loader = NeighborLoader(data,
                                  input_nodes=val_idx,
                                  num_neighbors=num_nghs,
                                  shuffle=False,
                                  batch_size=args.batch_size)
    test_loader = NeighborLoader(data,
                                 input_nodes=test_idx,
                                 num_neighbors=num_nghs,
                                 shuffle=False,
                                 batch_size=args.batch_size)
    return train_loader, valid_loader, test_loader


def gen_dataloader_delta(args, edge_index, train_idx, val_idx, test_idx, features, labels,
                         num_nghs):
    features = torch.FloatTensor(features)
    index = torch.LongTensor([[i] for i in range(labels.shape[0])])
    labels = torch.cat((labels, index), 1)
    data = Data(x=features, edge_index=edge_index, y=labels)
    del features
    gc.collect()

    train_loader = NeighborLoader(data,
                                  input_nodes=train_idx,
                                  num_neighbors=num_nghs,
                                  shuffle=False,
                                  batch_size=args.batch_size)
    valid_loader = NeighborLoader(data,
                                  input_nodes=val_idx,
                                  num_neighbors=num_nghs,
                                  shuffle=False,
                                  batch_size=args.batch_size)
    test_loader = NeighborLoader(data,
                                 input_nodes=test_idx,
                                 num_neighbors=num_nghs,
                                 shuffle=False,
                                 batch_size=args.batch_size)
    return train_loader, valid_loader, test_loader


def train(args, model, train_loader, valid_loader, device, checkpt_file):
    bad_counter = 0
    best = 0
    best_epoch = 0
    train_time = 0
    model.reset_parameters()
    print("--------------------------")
    print("Training...")
    for epoch in range(args.epochs):
        loss_tra, train_ep = gcn.train(model, device, train_loader, args.lr, args.weight_decay)
        t_st = time.time()
        f1_val = gcn.validate(model, device, valid_loader)
        train_time += train_ep
        if (epoch + 1) % 20 == 0:
            print('Epoch:{:02d}, Train_loss:{:.3f}, Valid_acc:{:.2f}%, Time_cost:{:.3f}/{:.3f}'.
                  format(epoch + 1, loss_tra, 100 * f1_val, train_ep, train_time))
            # print('Remove print')
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

    print('Train cost: {:.2f}s'.format(train_time))
    print('The best epoch: {}th'.format(best_epoch))


def test(model, test_loader, device, checkpt_file):
    test_acc = gcn.test(model, device, test_loader, checkpt_file)
    print('Test accuracy:{:.2f}%'.format(100 * test_acc))
    return test_acc


def train_delta(args,
                model,
                train_loader,
                valid_loader,
                device,
                checkpt_file,
                v_sen=None,
                v_insen=None):
    bad_counter = 0
    best = 0
    best_epoch = 0
    train_time = 0
    model.reset_parameters()
    print("--------------------------")
    print("Training...")
    for epoch in range(args.epochs):
        loss_tra, train_ep = gcn.train_delta(model, device, train_loader, args.lr,
                                             args.weight_decay, v_sen, v_insen)
        t_st = time.time()
        f1_val = gcn.validate_delta(model, device, valid_loader, v_sen, v_insen)
        train_time += train_ep
        if (epoch + 1) % 20 == 0:
            print('Epoch:{:02d}, Train_loss:{:.3f}, Valid_acc:{:.2f}%, Time_cost:{:.3f}/{:.3f}'.
                  format(epoch + 1, loss_tra, 100 * f1_val, train_ep, train_time))
            # print('Remove print')
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

    print('Train cost: {:.2f}s'.format(train_time))
    print('The best epoch: {}th'.format(best_epoch))


def test_delta(model, test_loader, device, checkpt_file, v_sen=None, v_insen=None):
    test_acc = gcn.test_delta(model, device, test_loader, checkpt_file, v_sen, v_insen)
    print('Test accuracy:{:.2f}%'.format(100 * test_acc))
    return test_acc


# def train(model, edge_index, train_idx, val_idx, features, labels, num_nghs, args, device,
#           checkpt_file):
#     model.reset_parameters()
#     features = torch.FloatTensor(features)
#     data = Data(x=features, edge_index=edge_index, y=labels)
#     del features
#     gc.collect()

#     train_loader = NeighborLoader(data,
#                                   input_nodes=train_idx,
#                                   num_neighbors=num_nghs,
#                                   shuffle=True,
#                                   batch_size=args.batch_size)
#     valid_loader = NeighborLoader(data,
#                                   input_nodes=val_idx,
#                                   num_neighbors=num_nghs,
#                                   shuffle=False,
#                                   batch_size=args.batch_size)

#     bad_counter = 0
#     best = 0
#     best_epoch = 0
#     train_time = 0
#     model.reset_parameters()
#     print("--------------------------")
#     print("Training...")
#     for epoch in range(args.epochs):
#         loss_tra, train_ep = gcn.train(model, device, train_loader, args.lr, args.weight_decay)
#         t_st = time.time()
#         f1_val = gcn.validate(model, device, valid_loader)
#         train_time += train_ep
#         if (epoch + 1) % 20 == 0:
#             print('Epoch:{:02d}, Train_loss:{:.3f}, Valid_acc:{:.2f}%, Time_cost:{:.3f}/{:.3f}'.
#                   format(epoch + 1, loss_tra, 100 * f1_val, train_ep, train_time))
#             # print('Remove print')
#         if f1_val > best:
#             best = f1_val
#             best_epoch = epoch + 1
#             t_st = time.time()
#             torch.save(model.state_dict(), checkpt_file)
#             bad_counter = 0
#         else:
#             bad_counter += 1
#         if bad_counter == args.patience:
#             break

#     print('Train cost: {:.2f}s'.format(train_time))
#     print('The best epoch: {}th'.format(best_epoch))

# def test(model, edge_index, test_idx, features, labels, num_nghs, args, device, checkpt_file):
#     features = torch.FloatTensor(features)
#     data = Data(x=features, edge_index=edge_index, y=labels)
#     del features
#     gc.collect()

#     test_loader = NeighborLoader(data,
#                                  input_nodes=test_idx,
#                                  num_neighbors=num_nghs,
#                                  shuffle=False,
#                                  batch_size=args.batch_size)

#     test_acc = gcn.test(model, device, test_loader, checkpt_file)
#     print('Test accuracy:{:.2f}%'.format(100 * test_acc))
#     return test_acc

if __name__ == '__main__':

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

    # args.dataset = 'Cora'
    args.dataset = 'CiteSeer'
    # args.dataset = 'PubMed'
    # args.dataset = 'arxiv'
    # args.dataset = 'products'

    args.layer = 2
    args.hidden = 128
    args.lr = 1e-2
    args.weight_decay = 0
    args.dropout = 0.5
    # args.epochs = 1
    args.gpu = 0
    args.batch_size = pow(2, 13)

    print('\n************ {:s} ************'.format(args.dataset))
    print(args)

    main(args)
