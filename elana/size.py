import os
import math
import logging

import torch
import torch.nn as nn

from transformers import DynamicCache
# from modeling.model_helper_registry import ModelHelperRegistry

logger = logging.getLogger(os.path.basename(__file__))


def tensor_tree_nbytes(obj) -> int:
    """
    Recursively walk an object tree and sum the size (in bytes)
    of all unique torch.Tensors found inside.

    Works for:
      - DynamicCache
      - HybridMambaAttentionDynamicCache
      - any custom cache/layer objects that store tensors in
        attributes / lists / dicts.

    Avoids double-counting the same tensor referenced multiple times.
    """
    visited = set()
    total = 0

    def _walk(o):
        nonlocal total
        if isinstance(o, torch.Tensor):
            oid = id(o)
            if oid in visited:
                return
            visited.add(oid)
            total += o.numel() * o.element_size()
        elif isinstance(o, (list, tuple, set)):
            for x in o:
                _walk(x)
        elif isinstance(o, dict):
            for x in o.values():
                _walk(x)
        else:
            # For cache/layer objects: inspect their attributes
            if hasattr(o, "__dict__"):
                for v in o.__dict__.values():
                    _walk(v)

    _walk(obj)
    return total


def dynamic_cache_nbytes(cache) -> int:
    """
    Wrapper for cache-like objects.
    """
    return tensor_tree_nbytes(cache)
