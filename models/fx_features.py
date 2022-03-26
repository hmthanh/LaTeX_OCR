from typing import Callable, List, Dict, Union

import torch
from torch import nn
from models.layers.cond_conv2d import CondConv2d
from models.layers.conv2d_same import Conv2dSame


from .features import _get_feature_info

try:
    from torchvision.models.feature_extraction import create_feature_extractor as _create_feature_extractor
    has_fx_feature_extraction = True
except ImportError:
    has_fx_feature_extraction = False

# Layers we went to treat as leaf modules
from models.layers.pool2d_same import AvgPool2dSame, MaxPool2dSame
from models.layers.non_local_attn import BilinearAttnTransform
from models.layers.std_conv import ScaledStdConv2dSame, StdConv2dSame

# NOTE: By default, any modules from timm.models.layers that we want to treat as leaf modules go here
# BUT modules from timm.models should use the registration mechanism below
_leaf_modules = {
    BilinearAttnTransform,  # reason: flow control t <= 1
    # Reason: get_same_padding has a max which raises a control flow error
    Conv2dSame, MaxPool2dSame, ScaledStdConv2dSame, StdConv2dSame, AvgPool2dSame,
    # reason: TypeError: F.conv2d received Proxy in groups=self.groups * B (because B = x.shape[0])
    CondConv2d,
}

try:
    from layers import InplaceAbn
    _leaf_modules.add(InplaceAbn)
except ImportError:
    pass


def register_notrace_module(module: nn.Module):
    """
    Any module not under timm.models.layers should get this decorator if we don't want to trace through it.
    """
    _leaf_modules.add(module)
    return module


# Functions we want to autowrap (treat them as leaves)
_autowrap_functions = set()


def register_notrace_function(func: Callable):
    """
    Decorator for functions which ought not to be traced through
    """
    _autowrap_functions.add(func)
    return func


def create_feature_extractor(model: nn.Module, return_nodes: Union[Dict[str, str], List[str]]):
    assert has_fx_feature_extraction, 'Please update to PyTorch 1.10+, torchvision 0.11+ for FX feature extraction'
    return _create_feature_extractor(
        model, return_nodes,
        tracer_kwargs={'leaf_modules': list(
            _leaf_modules), 'autowrap_functions': list(_autowrap_functions)}
    )


class FeatureGraphNet(nn.Module):
    """ A FX Graph based feature extractor that works with the model feature_info metadata
    """

    def __init__(self, model, out_indices, out_map=None):
        super().__init__()
        assert has_fx_feature_extraction, 'Please update to PyTorch 1.10+, torchvision 0.11+ for FX feature extraction'
        self.feature_info = _get_feature_info(model, out_indices)
        if out_map is not None:
            assert len(out_map) == len(out_indices)
        return_nodes = {
            info['module']: out_map[i] if out_map is not None else info['module']
            for i, info in enumerate(self.feature_info) if i in out_indices}
        self.graph_module = create_feature_extractor(model, return_nodes)

    def forward(self, x):
        return list(self.graph_module(x).values())
