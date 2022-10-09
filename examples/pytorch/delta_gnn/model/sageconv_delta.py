"""Torch Module for GraphSAGE-delta layer"""
from dgl.nn.pytorch import SAGEConv


class SAGEConv_delta(SAGEConv):

    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv_delta, self).__init__(in_feats, out_feats, aggregator_type, feat_drop, norm,
                                             bias, norm, activation)

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
