import torch
from torch._C import device
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

import torch as th
from torch import nn
from torch.nn import init

# from dgl import function as fn
# from dgl import DGLError
# # from dgl.transform import reverse
# from dgl.convert import block_to_graph
# from dgl.heterograph import DGLBlock


class GraphConv_delta(GraphConv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GraphConv_delta, self).__init__(in_feats, out_feats, norm, weight, bias, activation,
                                              allow_zero_in_degree)

        # feat_prev size is variable with graph evolving (size = g_nodes_number * out_feats)
        # self.feat = nn.Parameter(th.Tensor(1, out_feats))

    def forward(self, graph, feat, ngh_high_deg=None, ngh_low_deg=None):
        r"""    
        The data type of "ngh_high_deg" and "ngh_low_deg" is List
        """
        if (ngh_high_deg is not None) or (ngh_low_deg is not None):
            edge_mask = graph.edata['edge_mask']
            rst_delta = super().forward(graph, feat, edge_weight=edge_mask)
            return rst_delta
        else:
            rst = super().forward(graph, feat)
            # self.feat = nn.Parameter(rst)
            return rst