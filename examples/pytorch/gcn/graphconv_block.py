import torch
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


class GraphConv_block(GraphConv):
    def forward(self, block, feat, weight=None, edge_weight=None):
        with block.local_scope():
            if not self._allow_zero_in_degree:
                if (block.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the block, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input block by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == block.number_of_edges()
                block.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, block)
            if self._norm in ['left', 'both']:
                degs = block.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1, ) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                block.srcdata['h'] = feat_src
                # block.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                block.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = block.dstdata['h']
            else:
                # aggregate first then mult W
                block.srcdata['h'] = feat_src
                # block.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                block.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = block.dstdata['h']
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm in ['right', 'both']:
                degs = block.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1, ) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst