import time
import uuid
import random
import argparse
import gc
# import torch
import numpy as np
# from model import ClassMLP
from model.gcn_t import GCN, GCN_delta
import model.gcn_t as gcn
from model.graphsage_t import GraphSAGE, GraphSAGE_delta
import model.graphsage_t as graphsage
from model.gat_t import GAT, GAT_delta
import model.gat_t as gat
from model.gin_t import GIN, GIN_delta
import model.gin_t as gin
from model.deepergcn_t import DeeperGCN, DeeperGCN_delta
import model.deepergcn_t as deepergcn

from utils import *
from glob import glob
import copy as cp
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import util
import json
import torch as th

import faulthandler


def main(args):
    faulthandler.enable()
    # import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    Task_time_start = time.perf_counter()

    # Load GNN model parameter
    model_name = args.model
    dataset_name = args.dataset
    if model_name == 'gcn':
        # path = os.getcwd()
        # print(path)
        # with open('./examples/pytorch/delta_gnn/gcn_para.json', 'r') as f:
        with open('./gcn_para.json', 'r') as f:
            para = json.load(f)
            n_hidden = para[dataset_name]['--n-hidden']
            n_layers = para[dataset_name]['--n-layers']
            lr = para[dataset_name]['--lr']
            weight_decay = para[dataset_name]['--weight-decay']
            dropout = para[dataset_name]['--dropout']
    elif model_name == 'graphsage':
        # with open('./examples/pytorch/delta_gnn/graphsage_para.json', 'r') as f:
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
        # with open('./examples/pytorch/delta_gnn/gat_para.json', 'r') as f:
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
        # path = os.getcwd()
        # print(path)
        # with open('./examples/pytorch/delta_gnn/gin_para.json', 'r') as f:
        with open('./gin_para.json', 'r') as f:
            para = json.load(f)
            n_hidden = para[dataset_name]['--n-hidden']
            n_layers = para[dataset_name]['--n-layers']
            lr = para[dataset_name]['--lr']
            weight_decay = para[dataset_name]['--weight-decay']
            dropout = para[dataset_name]['--dropout']
    elif model_name == 'deepergcn':
        # path = os.getcwd()
        # print(path)
        # with open('./examples/pytorch/delta_gnn/gin_para.json', 'r') as f:
        with open('./deepergcn_para.json', 'r') as f:
            para = json.load(f)
            n_hidden = para[dataset_name]['--n-hidden']
            n_layers = para[dataset_name]['--n-layers']
            lr = para[dataset_name]['--lr']
            weight_decay = para[dataset_name]['--weight-decay']
            dropout = para[dataset_name]['--dropout']
    else:
        assert ('Not define GNN model')

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        gpu_id = args.gpu

    device = torch.device("cuda:" + str(gpu_id) if cuda else "cpu")
    batch_size = args.batch_size

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
    # n_hidden = args.hidden
    # n_layers = args.layer
    # dropout = args.dropout

    # create GNN model
    if model_name == 'gcn':
        model_wo_retrain = GCN(in_feats, n_hidden, n_classes, n_layers, dropout).to(device)
        model_retrain = GCN(in_feats, n_hidden, n_classes, n_layers, dropout).to(device)
        model_delta = GCN_delta(in_feats, n_hidden, n_classes, n_layers, dropout,
                                features.shape[0]).to(device)
    elif model_name == 'graphsage':
        model_wo_retrain = GraphSAGE(in_feats, n_hidden, n_classes, n_layers, dropout).to(device)
        model_retrain = GraphSAGE(in_feats, n_hidden, n_classes, n_layers, dropout).to(device)
        model_delta = GraphSAGE_delta(in_feats, n_hidden, n_classes, n_layers, dropout,
                                      features.shape[0]).to(device)
    elif model_name == 'gat':
        model_wo_retrain = GAT(in_feats, n_hidden, n_classes, n_layers, feat_dropout,
                               heads).to(device)
        model_retrain = GAT(in_feats, n_hidden, n_classes, n_layers, feat_dropout, heads).to(device)
        model_delta = GAT_delta(in_feats, n_hidden, n_classes, n_layers, feat_dropout, heads,
                                features.shape[0]).to(device)
    elif model_name == 'gin':
        model_wo_retrain = GIN(in_feats, n_hidden, n_classes, n_layers, dropout).to(device)
        model_retrain = GIN(in_feats, n_hidden, n_classes, n_layers, dropout).to(device)
        model_delta = GIN_delta(in_feats, n_hidden, n_classes, n_layers, dropout,
                                features.shape[0]).to(device)
    elif model_name == 'deepergcn':
        model_wo_retrain = DeeperGCN(in_feats, n_hidden, n_classes, n_layers, dropout).to(device)
        model_retrain = DeeperGCN(in_feats, n_hidden, n_classes, n_layers, dropout).to(device)
        model_delta = DeeperGCN_delta(in_feats, n_hidden, n_classes, n_layers, dropout,
                                      features.shape[0]).to(device)

    # model_wo_retrain = GCN(in_feats, n_hidden, n_classes, n_layers, dropout).cuda(device)
    # model_retrain = GCN(in_feats, n_hidden, n_classes, n_layers, dropout).cuda(device)
    # model_delta = GCN_delta(in_feats, n_hidden, n_classes, n_layers, dropout,
    #                         features.shape[0]).cuda(device)

    accuracy = []
    print('Edges_init: ', edge_index_init.shape)
    # Training for the initial snapshot
    if model_name == 'graphsage':
        num_nghs = [int(fanout_) for fanout_ in fan_out.split(',')]
    else:
        num_nghs = [-1 for layer in range(n_layers)]

    edge_index_evo = edge_index_init.clone().detach()
    train_loader, valid_loader, test_loader = gen_dataloader(edge_index_evo, train_idx, val_idx,
                                                             test_idx, features, labels, num_nghs,
                                                             batch_size)
    # Gen edge_dict
    edge_dict = gen_edge_dict(edge_index_evo)
    edge_dict_delta = cp.deepcopy(edge_dict)

    # Without retrain
    print('### Model_wo_retrain:')
    # edge_index_wo_retrain = edge_idx_init.clone().detach()
    train(args, model_wo_retrain, train_loader, valid_loader, device, checkpt_file_wo_retrain, lr,
          weight_decay)
    acc_wo_retrain = test(model_wo_retrain, test_loader, device, checkpt_file_wo_retrain)

    # Retrain
    print('### Model_retrain:')
    # edge_index_retrain = edge_idx_init.clone().detach()
    train(args, model_retrain, train_loader, valid_loader, device, checkpt_file_retrain, lr,
          weight_decay)
    acc_retrain = test(model_retrain, test_loader, device, checkpt_file_retrain)

    # Delta-retrain
    print('### Model_retrain_delta:')
    edge_index_evo_delta = edge_index_init.clone().detach()
    train_delta(args, model_delta, train_loader, valid_loader, device, checkpt_file_delta, lr,
                weight_decay)
    acc_delta = test_delta(model_delta, test_loader, device, checkpt_file_delta)
    # acc_delta = 0

    accuracy.append([0, acc_wo_retrain * 100, acc_retrain * 100, acc_delta * 100])

    access_delta_accumulate = 0
    access_total_accumulate = 0
    comp_delta_accumulate = 0
    comp_total_accumulate = 0
    access_reduct_list = []
    comp_reduct_list = []
    sensitive_ratio = [100]  # The sensitivity in the inital snapshot is 0

    print('------------------ update -------------------')
    snapList = [f for f in glob('./data/' + args.dataset + '/*Edgeupdate_snap*.txt')]
    print('number of snapshots: ', len(snapList))
    for i in range(len(snapList)):
        print("\n--------------------------")
        print('Snapshot: {:d}'.format(i + 1))
        # Update edge_index according to inserted edges
        inserted_edge_index = load_updated_edges(args.dataset, i + 1)
        # edge_index_evo = insert_edges(edge_index_evo, inserted_edge_index)
        edge_dict, edge_index_evo = insert_edges_evo(edge_index_evo, edge_dict, inserted_edge_index,
                                                     0, n_layers)
        train_loader, valid_loader, test_loader = gen_dataloader(edge_index_evo, train_idx, val_idx,
                                                                 test_idx, features, labels,
                                                                 num_nghs, batch_size)

        print('### Model_wo_retrain:')
        # edge_index_wo_retrain = insert_edges(edge_index_wo_retrain, inserted_edge_index)
        print('Edges_wo_retrain: ', edge_index_evo.shape)
        acc_wo_retrain = test(model_wo_retrain, test_loader, device, checkpt_file_wo_retrain)

        print('### Model_retrain:')
        # edge_index_retrain = insert_edges(edge_index_retrain, inserted_edge_index)
        print('Edges_retrain: ', edge_index_evo.shape)
        if i <= 100:
            train(args, model_retrain, train_loader, valid_loader, device, checkpt_file_retrain, lr,
                  weight_decay)
        acc_retrain = test(model_retrain, test_loader, device, checkpt_file_retrain)

        print('### Model_delta:')
        threshold = args.threshold
        edge_dict_delta, edge_index_evo_delta, v_sen, v_insen, comp_total, comp_delta, access_total, access_delta = insert_edges_delta(
            edge_index_evo_delta, edge_dict_delta, inserted_edge_index, threshold, n_layers)

        print('Node_num: ', len(v_sen), len(v_insen))
        sensitive_ratio.append(round(len(v_sen) * 100 / (len(v_sen) + len(v_insen)), 2))

        comp_delta_accumulate += comp_delta
        comp_total_accumulate += comp_total
        access_delta_accumulate += access_delta
        access_total_accumulate += access_total
        comp_reduction = round(
            (comp_total_accumulate - comp_delta_accumulate) / comp_total_accumulate * 100, 2)
        access_reduction = round(
            (access_total_accumulate - access_delta_accumulate) / access_total_accumulate * 100, 2)
        access_reduct_list.append(access_reduction)
        comp_reduct_list.append(comp_reduction)
        print('Access reduction: {:d}/{:d} = {:.2f}%'.format(access_delta, access_total,
                                                             access_reduction))
        print('Computation reduction: {:d}/{:d} = {:.2f}%'.format(comp_delta, comp_total,
                                                                  comp_reduction))

        train_loader_delta, valid_loader_delta, test_loader_delta = gen_dataloader(
            edge_index_evo_delta, train_idx, val_idx, test_idx, features, labels, num_nghs,
            batch_size)
        print('Edges_delta: ', edge_index_evo_delta.shape)
        # train_delta(args, model_delta, train_loader_delta, valid_loader_delta, device,
        #                 checkpt_file_delta, lr, weight_decay, v_sen, v_insen)
        if i <= 100:
            train_delta(args, model_delta, train_loader_delta, valid_loader_delta, device,
                        checkpt_file_delta, lr, weight_decay, v_sen, v_insen)
        acc_delta = test_delta(model_delta, test_loader_delta, device, checkpt_file_delta, v_sen,
                               v_insen)

        accuracy.append([i + 1, acc_wo_retrain * 100, acc_retrain * 100, acc_delta * 100])

    acc_degrad = []
    for i in range(len(accuracy)):
        if i == 0:
            print('{:d}\t{:.2f}  {:.2f}  {:.2f}'.format(i, accuracy[i][1], accuracy[i][2],
                                                        accuracy[i][3]))
        else:
            # print('{:d}\t{:.2f}  {:.2f}  {:.2f}  aggr: {:.2f}%  comb: {:.2f}%'.format(
            #     i, accuracy[i][1], accuracy[i][2], accuracy[i][3], aggr_reduct_list[i - 1],
            #     comb_reduct_list[i - 1]))
            delta_acc = accuracy[i][2] - accuracy[i][3]
            print('{:d} {:.2f}  {:.2f}  {:.2f} {:.2f} mem: {:.2f}%  comp: {:.2f}%  sen:{:.2f}%'.
                  format(i, accuracy[i][1], accuracy[i][2], accuracy[i][3], delta_acc,
                         access_reduct_list[i - 1], comp_reduct_list[i - 1],
                         sensitive_ratio[i - 1]))
            acc_degrad.append(delta_acc)

    # aggr_reduct_avg = aggr_reduct_list[-1]
    # comb_reduct_avg = comb_reduct_list[-1]
    access_reduct_avg = access_reduct_list[-1]
    comp_reduct_avg = comp_reduct_list[-1]
    # print('Aggr_reduct_avg: {:.2f}%, Comb_reduct_avg: {:.2f}%'.format(aggr_reduct_avg,
    #                                                                   comb_reduct_avg))
    print()
    print('Avg degrad: Acc: {:.2f}, Mem: {:.1f}%, Comp: {:.1f}%'.format(
        np.mean(acc_degrad), access_reduct_avg, comp_reduct_avg))

    print('\n>> Task {:s} execution time: {}'.format(
        args.dataset, util.time_format(time.perf_counter() - Task_time_start)))

    acc_dump = [accuracy[i][1:] for i in range(len(accuracy))]
    acc_dump = [acc_dump[i] + [sensitive_ratio[i]] for i in range(len(acc_dump))]
    np.savetxt('./results/' + args.dataset + '_' + args.model + '_th' + str(args.threshold) +
               '_acc.txt',
               acc_dump,
               fmt='%.2f, %.2f, %.2f, %.2f')

    # reduction = [[aggr_reduct_list[i], comb_reduct_list[i]] for i in range(len(aggr_reduct_list))]
    reduction = [[access_reduct_list[i], comp_reduct_list[i]]
                 for i in range(len(access_reduct_list))]
    np.savetxt('./results/' + args.dataset + '_' + args.model + '_th' + str(args.threshold) +
               '_reduction.txt',
               reduction,
               fmt='%.2f, %.2f')


def gen_dataloader(edge_index, train_idx, val_idx, test_idx, features, labels, num_nghs,
                   batch_size):
    features = torch.FloatTensor(features)
    data = Data(x=features, edge_index=edge_index, y=labels)
    data.n_id = torch.arange(data.num_nodes)  # Keep the original vertex ID of global
    del features
    gc.collect()

    vertex_num_total = labels.size()[0]
    train_loader = NeighborLoader(
        data,
        input_nodes=train_idx,
        num_neighbors=num_nghs,
        shuffle=False,
        #   batch_size=vertex_num_total)
        batch_size=pow(2, 10))

    valid_loader = NeighborLoader(
        data,
        input_nodes=val_idx,
        num_neighbors=num_nghs,
        shuffle=False,
        #   batch_size=vertex_num_total)
        batch_size=pow(2, 10))

    test_loader = NeighborLoader(
        data,
        input_nodes=test_idx,
        num_neighbors=num_nghs,
        shuffle=False,
        #  batch_size=vertex_num_total)
        batch_size=pow(2, 10))

    return train_loader, valid_loader, test_loader


def train(args, model, train_loader, valid_loader, device, checkpt_file, lr, weight_decay):
    bad_counter = 0
    best = 0
    best_epoch = 0
    train_time = 0
    model.reset_parameters()
    print("--------------------------")
    print("Training...")

    args.epochs = 200

    for epoch in range(args.epochs):
        if args.model == 'gcn':
            loss_tra, train_ep = gcn.train(model, device, train_loader, lr, weight_decay)
        elif args.model == 'graphsage':
            loss_tra, train_ep = graphsage.train(model, device, train_loader, lr, weight_decay)
        elif args.model == 'gat':
            loss_tra, train_ep = gat.train(model, device, train_loader, lr, weight_decay)
        elif args.model == 'gin':
            loss_tra, train_ep = gin.train(model, device, train_loader, lr, weight_decay)
        elif args.model == 'deepergcn':
            loss_tra, train_ep = deepergcn.train(model, device, train_loader, lr)

        t_st = time.time()

        if args.model == 'gcn':
            f1_val = gcn.validate(model, device, valid_loader)
        elif args.model == 'graphsage':
            f1_val = graphsage.validate(model, device, valid_loader)
        elif args.model == 'gat':
            f1_val = gat.validate(model, device, valid_loader)
        elif args.model == 'gin':
            f1_val = gin.validate(model, device, valid_loader)
        elif args.model == 'deepergcn':
            f1_val = deepergcn.validate(model, device, valid_loader)

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
    if args.model == 'gcn':
        test_acc = gcn.test(model, device, test_loader, checkpt_file)
    elif args.model == 'graphsage':
        test_acc = graphsage.test(model, device, test_loader, checkpt_file)
    elif args.model == 'gat':
        test_acc = gat.test(model, device, test_loader, checkpt_file)
    elif args.model == 'gin':
        test_acc = gin.test(model, device, test_loader, checkpt_file)
    elif args.model == 'deepergcn':
        test_acc = deepergcn.test(model, device, test_loader, checkpt_file)
    print('Test accuracy:{:.2f}%'.format(100 * test_acc))
    return test_acc


def train_delta(args,
                model,
                train_loader,
                valid_loader,
                device,
                checkpt_file,
                lr,
                weight_decay,
                v_sen=None,
                v_insen=None):
    bad_counter = 0
    best = 0
    best_epoch = 0
    train_time = 0
    model.reset_parameters()
    print("--------------------------")
    print("Training...")

    args.epochs = 200

    v_sen_feat_loc_train, v_insen_feat_loc_train, v_sen_train, v_insen_train = util.feature_merge_preprocess(
        train_loader, device, v_sen, v_insen)
    v_sen_feat_loc_val, v_insen_feat_loc_val, v_sen_val, v_insen_val = util.feature_merge_preprocess(
        valid_loader, device, v_sen, v_insen)

    for epoch in range(args.epochs):
        if args.model == 'gcn':
            loss_tra, train_ep = gcn.train_delta(model, device, train_loader, lr, weight_decay,
                                                 v_sen_feat_loc_train, v_insen_feat_loc_train,
                                                 v_sen_train, v_insen_train)
        elif args.model == 'graphsage':
            # Remapping feat_loc during each iteration of training, due to the random sampling in graphSAGE
            loss_tra, train_ep = graphsage.train_delta(model, device, train_loader, lr,
                                                       weight_decay, v_sen, v_insen)
        elif args.model == 'gat':
            loss_tra, train_ep = gat.train_delta(model, device, train_loader, lr, weight_decay,
                                                 v_sen_feat_loc_train, v_insen_feat_loc_train,
                                                 v_sen_train, v_insen_train)
        elif args.model == 'gin':
            loss_tra, train_ep = gin.train_delta(model, device, train_loader, lr, weight_decay,
                                                 v_sen_feat_loc_train, v_insen_feat_loc_train,
                                                 v_sen_train, v_insen_train)
        elif args.model == 'deepergcn':
            loss_tra, train_ep = deepergcn.train_delta(model, device, train_loader, lr,
                                                       v_sen_feat_loc_train, v_insen_feat_loc_train,
                                                       v_sen_train, v_insen_train)

        t_st = time.time()

        if args.model == 'gcn':
            f1_val = gcn.validate_delta(model, device, valid_loader, v_sen_feat_loc_val,
                                        v_insen_feat_loc_val, v_sen_val, v_insen_val)
        elif args.model == 'graphsage':
            # Remapping feat_loc during each iteration of training, due to the random sampling in graphSAGE
            f1_val = graphsage.validate_delta(model, device, valid_loader, v_sen, v_insen)
        elif args.model == 'gat':
            f1_val = gat.validate_delta(model, device, valid_loader, v_sen_feat_loc_val,
                                        v_insen_feat_loc_val, v_sen_val, v_insen_val)
        elif args.model == 'gin':
            f1_val = gin.validate_delta(model, device, valid_loader, v_sen_feat_loc_val,
                                        v_insen_feat_loc_val, v_sen_val, v_insen_val)
        elif args.model == 'deepergcn':
            f1_val = deepergcn.validate_delta(model, device, valid_loader, v_sen_feat_loc_val,
                                              v_insen_feat_loc_val, v_sen_val, v_insen_val)

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
    if args.model == 'gcn':
        test_acc = gcn.test_delta(model, device, test_loader, checkpt_file, v_sen, v_insen)
    elif args.model == 'graphsage':
        test_acc = graphsage.test_delta(model, device, test_loader, checkpt_file, v_sen, v_insen)
    elif args.model == 'gat':
        test_acc = gat.test_delta(model, device, test_loader, checkpt_file, v_sen, v_insen)
    elif args.model == 'gin':
        test_acc = gin.test_delta(model, device, test_loader, checkpt_file, v_sen, v_insen)
    elif args.model == 'deepergcn':
        test_acc = deepergcn.test_delta(model, device, test_loader, checkpt_file, v_sen, v_insen)
    print('Test accuracy:{:.2f}%'.format(100 * test_acc))
    return test_acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Dataset and Algorithom
    parser.add_argument('--seed', type=int, default=20159, help='random seed..')
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="Model name ('gcn', 'graphsage', 'gin', 'gat').")
    parser.add_argument('--dataset', default='Cora', help='dateset.')
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
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs.')
    parser.add_argument('--batch_size', type=int, default=10000, help='batch size.')
    parser.add_argument('--patience', type=int, default=50, help='patience.')
    parser.add_argument('--threshold', type=int, default=0, help='Sensitivity threshold')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu')
    args = parser.parse_args()

    # args.model = 'gcn'
    # args.model = 'graphsage'
    # args.model = 'gat'
    # args.model = 'gin'
    args.model = 'deepergcn'

    # args.dataset = 'Cora'
    # args.dataset = 'CiteSeer'
    args.dataset = 'Twitch'
    # args.dataset = 'Facebook'
    # args.dataset = 'WikiCS'
    # args.dataset = 'arxiv'
    # args.dataset = 'mag'

    # args.dataset = 'PubMed'
    # args.dataset = 'products'

    args.threshold = 50

    # args.epochs = 1
    args.gpu = 0
    # args.batch_size = pow(2, 13)

    print('\n************ {:s} ************'.format(args.dataset))
    print('>> Task start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # print(args)

    main(args)
