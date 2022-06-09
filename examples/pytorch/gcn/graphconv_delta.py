import torch
from torch._C import device
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

import torch as th
from torch import nn
from torch.nn import init

from dgl import function as fn
from dgl import DGLError
# from dgl.transform import reverse
from dgl.convert import block_to_graph
from dgl.heterograph import DGLBlock


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
        self.feat = nn.Parameter(th.Tensor(1, out_feats))

    def forward(self, graph, feat, ngh_high_deg=None, ngh_low_deg=None):
        r"""    
        The data type of "ngh_high_deg" and "ngh_low_deg" is List
        """
        if ngh_high_deg is not None and ngh_low_deg is not None:
            rst_golden = super().forward(graph, feat)

            edge_mask = graph.edata['edge_mask']
            rst_delta = super().forward(graph, feat, edge_weight=edge_mask)

            # Resize feat_prev
            feat_num = self.feat.shape[0]
            if self.feat.shape[0] < rst_delta.shape[0]:
                new_feat_num = rst_delta.shape[0] - self.feat.shape[0]
                self.feat = nn.Parameter(
                    th.cat((self.feat, th.zeros([new_feat_num, self.feat.shape[1]])), 0))

            # Combine delta rst with feat_prev
            ngh_high_deg_ind = th.tensor(ngh_high_deg, dtype=th.long)
            ngh_low_deg_ind = th.tensor(ngh_low_deg, dtype=th.long)

            feat_high_deg = th.index_select(rst_delta, 0, ngh_high_deg_ind)
            feat_high_deg_golden = th.index_select(rst_golden, 0, ngh_high_deg_ind)

            feat_low_deg = th.index_select(rst_delta, 0, ngh_low_deg_ind)
            feat_low_deg_golden = th.index_select(rst_golden, 0, ngh_low_deg_ind)
            feat_low_deg_updated_golden = feat_low_deg_golden + feat_low_deg

            # Gen index for scatter
            index_high_deg = [[
                ngh_high_deg_ind[row].item() for col in range(feat_high_deg.shape[1])
            ] for row in range(ngh_high_deg_ind.shape[0])]

            index_low_deg = [[ngh_low_deg_ind[row].item() for col in range(feat_low_deg.shape[1])]
                             for row in range(ngh_low_deg_ind.shape[0])]

            index_high_deg = th.tensor(index_high_deg)
            index_low_deg = th.tensor(index_low_deg)

            # Update feat of the nodes in the high and low degree
            # self.feat.scatter(0, index_high_deg, feat_high_deg)
            self.feat.scatter(0, index_high_deg, feat_high_deg_golden)

            self.feat.scatter(0, index_low_deg, feat_low_deg, reduce='add')
            # self.feat.scatter(0, index_low_deg, feat_low_deg_updated_golden)

            return self.feat
        else:
            rst = super().forward(graph, feat)
            self.feat = nn.Parameter(rst)
            return rst