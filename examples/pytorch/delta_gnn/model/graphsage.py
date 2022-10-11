import torch as th
import torch.nn as nn
import torch.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import tqdm
from model.sageconv_delta import SAGEConv_delta
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
import numpy as np
import time
import dgl.function as fn


class SAGE(nn.Module):

    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(g, in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.g = g
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, features):
        h = features
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    # def inference(self, g, device, batch_size):
    #     """Conduct layer-wise inference to get all the node embeddings."""
    #     feat = g.ndata['feat']
    #     sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
    #     dataloader = DataLoader(g,
    #                             th.arange(g.num_nodes()).to(g.device),
    #                             sampler,
    #                             device=device,
    #                             batch_size=batch_size,
    #                             shuffle=False,
    #                             drop_last=False,
    #                             num_workers=0)
    #     buffer_device = th.device('cpu')
    #     pin_memory = (buffer_device != device)

    #     for l, layer in enumerate(self.layers):
    #         y = th.empty(g.num_nodes(),
    #                      self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
    #                      device=buffer_device,
    #                      pin_memory=pin_memory)
    #         feat = feat.to(device)
    #         for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
    #             x = feat[input_nodes]
    #             h = layer(blocks[0], x)  # len(blocks) = 1
    #             if l != len(self.layers) - 1:
    #                 h = th.nn.functional.relu(h)
    #                 h = self.dropout(h)
    #             # by design, our output nodes are contiguous
    #             y[output_nodes[0]:output_nodes[-1] + 1] = h.to(buffer_device)
    #         feat = y
    #     return y

    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(),
                         self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(g,
                                                        th.arange(g.num_nodes()).to(g.device),
                                                        sampler,
                                                        device=device if num_workers == 0 else None,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        drop_last=False,
                                                        num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y


class NegativeSampler(object):

    def __init__(self, g, k, neg_share=False, device=None):
        if device is None:
            device = g.device
        self.weights = g.in_degrees().float().to(device)**0.75
        self.k = k
        self.neg_share = neg_share

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        n = len(src)
        if self.neg_share and n % self.k == 0:
            dst = self.weights.multinomial(n, replacement=True)
            dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
        else:
            dst = self.weights.multinomial(n * self.k, replacement=True)
        src = src.repeat_interleave(self.k)
        return src, dst


# def train(args, g, model, device, fanout, loss_fcn, optimizer):
#     batch_size = 10000
#     # create sampler & dataloader
#     # train_idx = dataset.train_idx.to(device)
#     # val_idx = dataset.val_idx.to(device)
#     nfeat = g.ndata['feat']
#     train_idx = g.ndata['train_mask'].to(device)
#     val_idx = g.ndata['val_mask'].to(device)
#     # sampler = NeighborSampler([int(fanout_) for fanout_ in fanout.split(',')],
#     #                           prefetch_node_feats=['feat'],
#     #                           prefetch_labels=['label'])
#     sampler = dgl.dataloading.MultiLayerNeighborSampler(
#         [int(fanout_) for fanout_ in fanout.split(',')])
#     use_uva = (args.mode == 'mixed')
#     # train_dataloader = DataLoader(g,
#     #                               train_idx,
#     #                               sampler,
#     #                               device=device,
#     #                               batch_size=1024,
#     #                               shuffle=True,
#     #                               drop_last=False,
#     #                               num_workers=0,
#     #                               use_uva=use_uva)

#     n_edges = g.num_edges()
#     train_seeds = train_idx
#     dataloader = dgl.dataloading.EdgeDataLoader(
#         g,
#         train_idx,
#         sampler,
#         exclude='reverse_id',
#         # For each edge with ID e in Reddit dataset, the reverse edge is e ± |E|/2.
#         reverse_eids=th.cat([th.arange(n_edges // 2, n_edges),
#                              th.arange(0, n_edges // 2)]).to(train_seeds),
#         negative_sampler=NegativeSampler(g, 1),
#         device=device,
#         # use_ddp=n_gpus > 1,
#         batch_size=batch_size,
#         shuffle=True,
#         drop_last=False,
#         # num_workers=args.num_workers
#     )

#     # val_dataloader = DataLoader(g,
#     #                             val_idx,
#     #                             sampler,
#     #                             device=device,
#     #                             batch_size=1024,
#     #                             shuffle=True,
#     #                             drop_last=False,
#     #                             num_workers=0,
#     #                             use_uva=use_uva)

#     # val_dataloader = dgl.dataloading.EdgeDataLoader(
#     #     g,
#     #     val_idx,
#     #     sampler,
#     #     exclude='reverse_id',
#     #     # For each edge with ID e in Reddit dataset, the reverse edge is e ± |E|/2.
#     #     reverse_eids=th.cat(
#     #         [th.arange(n_edges // 2, n_edges),
#     #          th.arange(0, n_edges // 2)]).to(train_seeds),
#     #     # negative_sampler=NegativeSampler(
#     #     #     g, args.num_negs, args.neg_share,
#     #     #     device if args.graph_device == 'uva' else None),
#     #     device=device,
#     #     # use_ddp=n_gpus > 1,
#     #     batch_size=10000,
#     #     shuffle=True,
#     #     drop_last=False,
#     #     # num_workers=args.num_workers
#     #     )

#     # opt = th.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

#     # for epoch in range(10):
#     #     model.train()
#     #     total_loss = 0
#     #     for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
#     #         x = blocks[0].srcdata['feat']
#     #         y = blocks[-1].dstdata['label']
#     #         y_hat = model(blocks, x)
#     #         # loss = F.cross_entropy(y_hat, y)
#     #         loss = loss_fcn(y_hat, y)
#     #         optimizer.zero_grad()
#     #         loss.backward()
#     #         optimizer.step()
#     #         total_loss += loss.item()
#     #     # acc = evaluate_with_sample(model, g, val_dataloader)
#     #     acc = evaluate(device, model, g, val_idx, batch_size=10000)
#     #     print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, total_loss / (it + 1),
#     #                                                                  acc.item()))

#     # Training loop
#     proc_id = 0
#     avg = 0
#     iter_pos = []
#     iter_neg = []
#     iter_d = []
#     iter_t = []
#     best_eval_acc = 0
#     # best_test_acc = 0
#     for epoch in range(args.n_epochs):
#         tic = time.time()

#         # Loop over the dataloader to sample the computation dependency graph as a list of
#         # blocks.
#         tic_step = time.time()
#         for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
#             batch_inputs = nfeat[input_nodes].to(device)
#             pos_graph = pos_graph.to(device)
#             neg_graph = neg_graph.to(device)
#             blocks = [block.int().to(device) for block in blocks]
#             d_step = time.time()

#             # Compute loss and prediction
#             batch_pred = model(blocks, batch_inputs)
#             loss = loss_fcn(batch_pred, pos_graph, neg_graph)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             t = time.time()
#             pos_edges = pos_graph.num_edges()
#             neg_edges = neg_graph.num_edges()
#             iter_pos.append(pos_edges / (t - tic_step))
#             iter_neg.append(neg_edges / (t - tic_step))
#             iter_d.append(d_step - tic_step)
#             iter_t.append(t - d_step)
#             if step % args.log_every == 0 and proc_id == 0:
#                 gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available(
#                 ) else 0
#                 print(
#                     '[{}]Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f} | GPU {:.1f} MB'
#                     .format(proc_id, epoch, step, loss.item(), np.mean(iter_pos[3:]),
#                             np.mean(iter_neg[3:]), np.mean(iter_d[3:]), np.mean(iter_t[3:]),
#                             gpu_mem_alloc))
#             tic_step = time.time()

#             if step % args.eval_every == 0 and proc_id == 0:
#                 eval_acc = evaluate(device, model, g, val_idx, batch_size)
#                 print('Eval Acc {:.4f} Test Acc {:.4f}'.format(eval_acc))
#                 if eval_acc > best_eval_acc:
#                     best_eval_acc = eval_acc
#                 print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc))
#         toc = time.time()
#         if proc_id == 0:
#             print('Epoch Time(s): {:.4f}'.format(toc - tic))
#         if epoch >= 5:
#             avg += toc - tic
#         # if n_gpus > 1:
#         #     th.distributed.barrier()

#     if proc_id == 0:
#         print('Avg epoch time: {}'.format(avg / (epoch - 4)))


class CrossEntropyLoss(nn.Module):

    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = th.cat([pos_score, neg_score])
        label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
        loss = th.nn.functional.binary_cross_entropy_with_logits(score, label.float())
        return loss


def train(args, g, model, device, fanout, batch_size, loss_fcn, optimizer):
    # Unpack data
    # device = th.device(devices[proc_id])
    # if n_gpus > 0:
    #     th.cuda.set_device(device)
    # if n_gpus > 1:
    #     dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
    #         master_ip='127.0.0.1', master_port='12345')
    #     world_size = n_gpus
    #     th.distributed.init_process_group(backend="nccl",
    #                                       init_method=dist_init_method,
    #                                       world_size=world_size,
    #                                       rank=proc_id)
    # train_nid, val_nid, test_nid, n_classes, g, nfeat, labels = data

    # batch_size = 10000

    train_nid = g.ndata['train_mask'].to(device)
    val_mask = g.ndata['val_mask'].to(device)
    test_nid = g.ndata['test_mask'].to(device)
    nfeat = g.ndata['feat']
    labels = g.ndata['label']

    # if args.data_device == 'gpu':
    #     nfeat = nfeat.to(device)
    #     labels = labels.to(device)
    # elif args.data_device == 'uva':
    #     nfeat = dgl.contrib.UnifiedTensor(nfeat, device=device)
    #     labels = dgl.contrib.UnifiedTensor(labels, device=device)
    in_feats = nfeat.shape[1]

    # Create PyTorch DataLoader for constructing blocks
    n_edges = g.num_edges()
    train_seeds = th.arange(n_edges)

    # if args.graph_device == 'gpu':
    #     train_seeds = train_seeds.to(device)
    #     g = g.to(device)
    #     args.num_workers = 0
    # elif args.graph_device == 'uva':
    #     train_seeds = train_seeds.to(device)
    #     g.pin_memory_()
    #     args.num_workers = 0

    # Create sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout_) for fanout_ in fanout.split(',')])
    dataloader = dgl.dataloading.EdgeDataLoader(
        g,
        train_seeds,
        sampler,
        exclude='reverse_id',
        # For each edge with ID e in Reddit dataset, the reverse edge is e ± |E|/2.
        reverse_eids=th.cat([th.arange(n_edges // 2, n_edges),
                             th.arange(0, n_edges // 2)]).to(train_seeds),
        negative_sampler=NegativeSampler(g, 1),
        device=device,
        # use_ddp=n_gpus > 1,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        # num_workers=args.num_workers
    )

    # Define model and optimizer
    # model = SAGE(in_feats, args.n_hidden, args.n_hidden, args.n_layers, F.relu, args.dropout)
    model = model.to(device)
    # if n_gpus > 1:
    #     model = DistributedDataParallel(model,
    #                                     device_ids=[device],
    #                                     output_device=device)
    # loss_fcn = CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    proc_id = 0
    # Training loop
    avg = 0
    iter_pos = []
    iter_neg = []
    iter_d = []
    iter_t = []
    best_acc = 0
    for epoch in range(args.n_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
            batch_inputs = nfeat[input_nodes].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            blocks = [block.int().to(device) for block in blocks]
            d_step = time.time()

            loss_fcn = CrossEntropyLoss()

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, pos_graph, neg_graph)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## args log_every; eval_eval
            log_every = 20
            eval_every = 1000

            t = time.time()
            pos_edges = pos_graph.num_edges()
            neg_edges = neg_graph.num_edges()
            iter_pos.append(pos_edges / (t - tic_step))
            iter_neg.append(neg_edges / (t - tic_step))
            iter_d.append(d_step - tic_step)
            iter_t.append(t - d_step)
            if step % log_every == 0 and proc_id == 0:
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available(
                ) else 0
                print(
                    '[{}]Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f} | GPU {:.1f} MB'
                    .format(proc_id, epoch, step, loss.item(), np.mean(iter_pos[3:]),
                            np.mean(iter_neg[3:]), np.mean(iter_d[3:]), np.mean(iter_t[3:]),
                            gpu_mem_alloc))
            tic_step = time.time()

            if step % eval_every == 0 and proc_id == 0:
                acc = evaluate(model, g, labels, val_mask, batch_size, device)
                print('Eval Acc {:.4f}'.format(acc))
                if acc > best_acc:
                    best_acc = acc
                print('Best Eval Acc {:.4f}'.format(best_acc))
        toc = time.time()
        if proc_id == 0:
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        # if n_gpus > 1:
        #     th.distributed.barrier()

    if proc_id == 0:
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))


# def evaluate_with_sample(model, g, dataloader):
#     r"""
#     Evaluate in training, include neighbor sampling (Used in training)
#     """

#     model.eval()
#     ys = []
#     y_hats = []
#     for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
#         with th.no_grad():
#             x = blocks[0].srcdata['feat']
#             ys.append(blocks[-1].dstdata['label'])
#             y_hats.append(model(blocks, x))
#     return MF.accuracy(th.cat(y_hats), th.cat(ys))

# def evaluate(device, model, g, mask, batch_size):
#     r"""
#     Evaluate with all neighbor aggregation (Used in test)
#     """

#     model.eval()
#     with th.no_grad():
#         # pred = model.inference(g, device, batch_size)  # pred in buffer_device
#         pred = model.inference(g, g.ndata['feat'], device, batch_size, num_workers = 0)  # pred in buffer_device
#         pred = pred[mask]
#         label = g.ndata['label'][mask].to(pred.device)
#         return MF.accuracy(pred, label)


def compute_acc(emb, labels, train_nids, acc_mask):
    """
    Compute the accuracy of prediction given the labels.

    acc_mask: The ids of nodes for accuracy computing
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    acc_mask = acc_mask.cpu().numpy()
    acc_labels = labels[acc_mask]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(acc_labels, pred[acc_mask], average='micro')
    return f1_micro_eval


def evaluate(model, g, labels, mask, batch_size, device, num_workers=0):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    nfeat = g.ndata['feat']
    labels = g.ndata['label']
    train_nids = g.ndata['train_mask']

    model.eval()
    with th.no_grad():
        # single gpu
        if isinstance(model, SAGE):
            pred = model.inference(g, nfeat, device, batch_size, num_workers)
        # multi gpu
        else:
            pred = model.module.inference(g, nfeat, device, batch_size, num_workers)
    model.train()
    return compute_acc(pred, labels, train_nids, mask)


class SAGE_delta(nn.Module):
    r"""
    Initial para:
        g: subgraph of original graph
        in_feats: feature matrix of current subgraph
    """

    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(g, in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.g = g
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(SAGEConv_delta(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(SAGEConv_delta(n_hidden, n_hidden, 'mean'))
            self.layers.append(SAGEConv_delta(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(SAGEConv_delta(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # Record previous embedding
        self.embedding = th.Tensor([[0 for i in range(n_classes)]
                                    for j in range(g.number_of_nodes())]).requires_grad_(True)

    def forward(self, blocks, features, ngh_high_deg=None, ngh_low_deg=None):
        h = features
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)

        # for name, parameters in self.named_parameters():
        #     print(name, ':', parameters.size())
        #     print(parameters.detach().is_leaf)
        #     print(parameters.detach().grad)
        #     print(parameters.detach().grad_fn)

        if (ngh_high_deg is not None) or (ngh_low_deg is not None):
            # Combine delta-inferenced embedding and previous embedding
            h = self.combine_embedding(self.embedding, h, ngh_high_deg, ngh_low_deg)

        return h

    def combine_embedding(self, embedding_prev, feat, ngh_high_deg, ngh_low_deg):
        # Compulsorily execute in CPU (GPU not suits for scalar execution)
        device = feat.device
        feat = feat.to('cpu')
        embedding_prev = embedding_prev.to('cpu')

        # Combine delta rst with feat_prev
        feat_prev_ind = list(i for i in range(embedding_prev.shape[0]))
        feat_prev_keep_ind = list(set(feat_prev_ind) - set(ngh_high_deg) - set(ngh_low_deg))

        feat_prev_keep_ind = th.tensor(feat_prev_keep_ind, dtype=th.long)
        # ngh_high_deg_ind = th.tensor(ngh_high_deg, dtype=th.long)
        ngh_low_deg_ind = th.tensor(ngh_low_deg, dtype=th.long)

        feat_prev = th.index_select(embedding_prev, 0, feat_prev_keep_ind)
        # feat_high_deg = th.index_select(feat, 0, ngh_high_deg_ind)
        feat_low_deg = th.index_select(feat, 0, ngh_low_deg_ind)

        # Gen index for scatter
        index_feat_prev = [[feat_prev_keep_ind[row].item() for col in range(feat_prev.shape[1])]
                           for row in range(feat_prev_keep_ind.shape[0])]
        # index_high_deg = [[ngh_high_deg_ind[row].item() for col in range(feat_high_deg.shape[1])]
        #                   for row in range(ngh_high_deg_ind.shape[0])]
        index_low_deg = [[ngh_low_deg_ind[row].item() for col in range(feat_low_deg.shape[1])]
                         for row in range(ngh_low_deg_ind.shape[0])]

        index_feat_prev = th.tensor(index_feat_prev)
        # index_high_deg = th.tensor(index_high_deg)
        index_low_deg = th.tensor(index_low_deg)

        # Update feat of the nodes in the high and low degree
        feat.scatter(0, index_feat_prev, feat_prev)
        # embedding_prev.scatter(0, index_high_deg, feat_high_deg)
        feat.scatter(0, index_low_deg, feat_low_deg, reduce='add')

        # Transfer 'feat' to its previous device
        feat = feat.to(device)

        return feat
