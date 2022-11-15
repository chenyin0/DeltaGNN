import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset, CiteseerGraphDataset
import tqdm
import argparse
import numpy as np
import time

class SAGE(nn.Module):

    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(g,
                                torch.arange(g.num_nodes()).to(g.device),
                                sampler,
                                device=device,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(g.num_nodes(),
                            self.hid_size if l != len(self.layers) - 1 else self.out_size,
                            device=buffer_device,
                            pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, graph, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(blocks[-1].dstdata['label'])
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys))


def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size)  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata['label'][nid].to(pred.device)
        return MF.accuracy(pred, label)


def train(args, device, g, dataset, model):
    # create sampler & dataloader
    # train_idx = dataset.train_idx.to(device)
    train_mask = g.ndata['train_mask']
    train_idx = torch.Tensor(np.nonzero(train_mask.cpu().numpy())[0]).long()
    # val_idx = dataset.val_idx.to(device)
    val_mask = g.ndata['val_mask']
    val_idx = torch.Tensor(np.nonzero(val_mask.cpu().numpy())[0]).long()
    sampler = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=['feat'],
        prefetch_labels=['label'])
    use_uva = (args.mode == 'mixed')
    train_dataloader = DataLoader(g,
                                  train_idx,
                                  sampler,
                                  device=device,
                                  batch_size=1024,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=0,
                                  use_uva=use_uva)

    val_dataloader = DataLoader(g,
                                val_idx,
                                sampler,
                                device=device,
                                batch_size=1024,
                                shuffle=True,
                                drop_last=False,
                                num_workers=0,
                                use_uva=use_uva)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(1):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, g, val_dataloader)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, total_loss / (it + 1),
                                                                     acc.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default='cpu',
        choices=['cpu', 'mixed', 'puregpu'],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')

    # load and preprocess dataset
    print('Loading data')
    # dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-arxiv', root='../../../../dataset'))
    dataset = CiteseerGraphDataset(raw_dir='../../../../dataset')
    # dataset = CoraGraphDataset(raw_dir='../../../../dataset', transform=transform)
    g = dataset[0]
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')
    g = g.int().to(device)

    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 128, out_size).to(device)

    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    # print(train_mask, torch.count_nonzero(train_mask).item())
    # print(val_mask, torch.count_nonzero(val_mask).item())
    # print(test_mask, torch.count_nonzero(test_mask).item())
    # print(g.number_of_nodes())

    # model training
    print('Training...')
    train(args, device, g, dataset, model)

    # test the model
    print('Testing...')
    test_mask = g.ndata['test_mask']
    test_idx = torch.Tensor(np.nonzero(test_mask.numpy())[0]).long()
    time_start = time.perf_counter()
    acc = layerwise_infer(device, g, test_idx, model, batch_size=4096)
    print("Test Accuracy {:.4f}".format(acc.item()))
    print('Task time: ', time.perf_counter() - time_start)