import torch
from torch._C import device
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

import torch as th
from torch import nn
from torch.nn import init

from dgl import function as fn
from dgl import DGLError
from dgl.utils import expand_as_pair
# from dgl.transform import reverse
from dgl.convert import block_to_graph
from dgl.heterograph import DGLBlock

from pyinstrument import Profiler


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
        # self.feat_prev = nn.Parameter(th.Tensor(in_feats, out_feats), requires_grad=True)
        self.feat_prev = th.Tensor(1, out_feats)

    def forward(self, graph, feat, ngh_high_deg=None, ngh_low_deg=None):
        r"""    
        The data type of "ngh_high_deg" and "ngh_low_deg" is List
        """
        if ngh_high_deg is not None and ngh_low_deg is not None:
            edge_mask = graph.edata['edge_mask']
            rst_delta = super().forward(graph, feat, edge_weight=edge_mask)

            # Resize rst
            if self.feat_prev.size()[0] < graph.number_of_nodes():
                new_feat_num = graph.number_of_nodes() - self.feat_prev.size()[0]
                rst = th.cat((self.feat_prev, th.zeros(new_feat_num, self.feat_prev.size()[1])), 0)
            else:
                rst = self.feat_prev.clone()

            # Combine delta rst with feat_prev
            for node in ngh_high_deg:
                rst[node] = rst_delta[node]

            for node in ngh_low_deg:
                rst[node] = rst[node] + rst_delta[node]
                # rst[node] = rst_delta[node]

            self.feat_prev = rst  # Update feat_prev

            return rst
        else:
            rst = super().forward(graph, feat)
            return rst

    # def forward(self,
    #             graph,
    #             feat,
    #             nodes_high_deg=None,
    #             nodes_low_deg=None,
    #             weight=None,
    #             edge_weight=None):
    #     r"""

    #     Description
    #     -----------
    #     Compute graph convolution.

    #     Parameters
    #     ----------
    #     graph : DGLGraph
    #         The graph.
    #     feat : torch.Tensor or pair of torch.Tensor
    #         If a torch.Tensor is given, it represents the input feature of shape
    #         :math:`(N, D_{in})`
    #         where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
    #         If a pair of torch.Tensor is given, which is the case for bipartite graph, the pair
    #         must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
    #         :math:`(N_{out}, D_{in_{dst}})`.
    #     weight : torch.Tensor, optional
    #         Optional external weight tensor.
    #     edge_weight : torch.Tensor, optional
    #         Optional tensor on the edge. If given, the convolution will weight
    #         with regard to the message.

    #     Returns
    #     -------
    #     torch.Tensor
    #         The output feature

    #     Raises
    #     ------
    #     DGLError
    #         Case 1:
    #         If there are 0-in-degree nodes in the input graph, it will raise DGLError
    #         since no message will be passed to those nodes. This will cause invalid output.
    #         The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

    #         Case 2:
    #         External weight is provided while at the same time the module
    #         has defined its own weight parameter.

    #     Note
    #     ----
    #     * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
    #       dimensions, :math:`N` is the number of nodes.
    #     * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
    #       the same shape as the input.
    #     * Weight shape: :math:`(\text{in_feats}, \text{out_feats})`.
    #     """

    #     # profiler = Profiler()
    #     # profiler.start()

    #     if nodes_high_deg == None and nodes_low_deg == None:
    #         return super().forward(graph, feat, weight, edge_weight)

    #     with graph.local_scope():
    #         if not self._allow_zero_in_degree:
    #             if (graph.in_degrees() == 0).any():
    #                 raise DGLError('There are 0-in-degree nodes in the graph, '
    #                                'output for those nodes will be invalid. '
    #                                'This is harmful for some applications, '
    #                                'causing silent performance regression. '
    #                                'Adding self-loop on the input graph by '
    #                                'calling `g = dgl.add_self_loop(g)` will resolve '
    #                                'the issue. Setting ``allow_zero_in_degree`` '
    #                                'to be `True` when constructing this module will '
    #                                'suppress the check and let the code run.')
    #         aggregate_fn = fn.copy_src('h', 'm')
    #         if edge_weight is not None:
    #             assert edge_weight.shape[0] == graph.number_of_edges()
    #             graph.edata['_edge_weight'] = edge_weight
    #             aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

    #         # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
    #         feat_src, feat_dst = expand_as_pair(feat, graph)
    #         if self._norm in ['left', 'both']:
    #             degs = graph.out_degrees().float().clamp(min=1)
    #             if self._norm == 'both':
    #                 norm = th.pow(degs, -0.5)
    #             else:
    #                 norm = 1.0 / degs
    #             shp = norm.shape + (1, ) * (feat_src.dim() - 1)
    #             norm = th.reshape(norm, shp)
    #             feat_src = feat_src * norm

    #         if weight is not None:
    #             if self.weight is not None:
    #                 raise DGLError('External weight is provided while at the same time the'
    #                                ' module has defined its own weight parameter. Please'
    #                                ' create the module with flag weight=False.')
    #         else:
    #             weight = self.weight

    #         # if self._in_feats > self._out_feats:
    #         #     # mult W first to reduce the feature size for aggregation.
    #         #     if weight is not None:
    #         #         feat_src = th.matmul(feat_src, weight)
    #         #     graph.srcdata['h'] = feat_src
    #         #     graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
    #         #     rst = graph.dstdata['h']
    #         # else:
    #         #     # aggregate first then mult W
    #         #     graph.srcdata['h'] = feat_src
    #         #     graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
    #         #     rst = graph.dstdata['h']
    #         #     if weight is not None:
    #         #         rst = th.matmul(rst, weight)

    #         # # Expand feat_prev to evolved graph size
    #         # if self.feat_prev.size()[0] < graph.number_of_nodes():
    #         #     new_feat_num = graph.number_of_nodes() - self.feat_prev.size()[0]
    #         #     self.feat_prev = th.cat(
    #         #         (self.feat_prev, th.Tensor(new_feat_num,
    #         #                                    self.feat_prev.size()[1])), 0)

    #         # Delta aggregates
    #         feat_src = feat_src
    #         feat_aggr = th.zeros([feat.size()[0], feat.size()[1]])
    #         # Merge nodes_high_deg and nodes_low_deg
    #         ngh_dict = nodes_high_deg.copy()
    #         ngh_dict.update(nodes_low_deg)
    #         for root_node, ngh_nodes in ngh_dict.items():
    #             node_feat = th.Tensor(feat.size()[1])
    #             for node in ngh_nodes:
    #                 node_feat += feat_src[node]
    #             feat_aggr[root_node] = node_feat

    #         if weight is not None:
    #             rst_delta = th.matmul(feat_aggr, weight)

    #         if self._norm in ['right', 'both']:
    #             degs = graph.in_degrees().float().clamp(min=1)
    #             if self._norm == 'both':
    #                 norm = th.pow(degs, -0.5)
    #             else:
    #                 norm = 1.0 / degs
    #             shp = norm.shape + (1, ) * (feat_dst.dim() - 1)
    #             norm = th.reshape(norm, shp)
    #             rst_delta = rst_delta * norm

    #         if self.bias is not None:
    #             rst_delta = rst_delta + self.bias

    #         if self._activation is not None:
    #             rst_delta = self._activation(rst_delta)

    #         # Resize rst
    #         if self.feat_prev.size()[0] < graph.number_of_nodes():
    #             new_feat_num = graph.number_of_nodes() - self.feat_prev.size()[0]
    #             rst = th.cat((self.feat_prev, th.zeros(new_feat_num, self.feat_prev.size()[1])), 0)
    #         else:
    #             rst = self.feat_prev.clone()

    #         # Combine delta rst with feat_prev
    #         for node in nodes_high_deg.keys():
    #             rst[node] = rst_delta[node]

    #         for node in nodes_low_deg.keys():
    #             rst[node] = rst[node] + rst_delta[node]

    #         self.feat_prev = rst  # Update feat_prev

    #         # profiler.stop()
    #         # profiler.print()

    #         return rst