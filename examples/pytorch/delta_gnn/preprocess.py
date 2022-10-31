import dgl
import torch

def ogbn_mag_preprocess(dataset):
    """
    Generate subgraph (paper, cites, paper) from ogbn-mag
    """

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']['paper']
    valid_idx = split_idx['valid']['paper']
    test_idx = split_idx['test']['paper']

    g_original, labels = dataset[0]
    labels = labels['paper'].squeeze()
    sub_g = dgl.edge_type_subgraph(g_original, [('paper', 'cites', 'paper')])
    h_sub_g = dgl.to_homogeneous(sub_g)
    h_sub_g.ndata['feat'] = g_original.nodes['paper'].data['feat']
    h_sub_g.ndata['label'] = labels

    # Initialize mask
    train_mask = torch.zeros((h_sub_g.num_nodes(), ), dtype=torch.bool)
    train_mask[train_idx] = True
    valid_mask = torch.zeros((h_sub_g.num_nodes(), ), dtype=torch.bool)
    valid_mask[valid_idx] = True
    test_mask = torch.zeros((h_sub_g.num_nodes(), ), dtype=torch.bool)
    test_mask[test_idx] = True

    h_sub_g.ndata['train_mask'] = train_mask
    h_sub_g.ndata['val_mask'] = valid_mask
    h_sub_g.ndata['test_mask'] = test_mask

    return h_sub_g
