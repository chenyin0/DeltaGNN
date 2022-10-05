"""Package for NN model."""
from .gcn import GCN
from .gcn import GCN_delta

# from .graphconv_delta import *

__all__ = ["GCN", "GCN_delta"]
