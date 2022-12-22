"""Torch Module for Graph Isomorphism Network layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from dgl.nn.pytorch import GINConv


class GINConv_delta(GINConv):

    def __init__(self,
                 apply_func=None,
                 aggregator_type='sum',
                 init_eps=0,
                 learn_eps=False,
                 activation=None):
        super(GINConv_delta, self).__init__(apply_func, aggregator_type, init_eps, learn_eps,
                                            activation)

    def forward(self, graph, feat, edge_mask=None):
        r"""    
        The data type of "ngh_high_deg" and "ngh_low_deg" is List
        """
        if edge_mask is not None:
            rst_delta = super().forward(graph, feat, edge_weight=edge_mask)
            return rst_delta
        else:
            rst = super().forward(graph, feat)
            return rst