"""
Best discovered architecture from automated search.

Architecture ID: arch_2334
Trial: 2334
Parameters: 103500
d_model: 50
Average accuracy (across 3 tasks): 0.9772

Per-task results:
  nested_depth: loss=0.0352, acc=0.9971
  multiscale_copy: loss=0.1212, acc=0.9980
  hierarchical_parity: loss=0.1219, acc=0.9365

Mechanism composition:
  Block 0: propagation=['lsh_exchange', 'hierarchical_pool_s2'], update=conv_gru
  Block 1: propagation=['hierarchical_pool_s2', 'dilated_conv_d8', 'dilated_conv_d32', 'depthwise_conv_3'], update=squeeze_excite
  Block 2: propagation=['depthwise_conv_3', 'depthwise_conv_15'], update=polynomial_activation
  Block 3: propagation=['depthwise_conv_5', 'spectral_filter', 'dilated_conv_d4'], update=per_channel_scale

This architecture was discovered via random search over 3000 trials.
All mechanisms are attention-free and softmax-free (except final classification head).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.primitives import (
    PROPAGATION_REGISTRY, UPDATE_REGISTRY, build_module
)


class DiscoveredBlock(nn.Module):
    """One block of the discovered architecture."""
    def __init__(self, d_model, prop_names, update_name):
        super().__init__()
        self.props = nn.ModuleList([
            build_module(name, d_model, PROPAGATION_REGISTRY)
            for name in prop_names
        ])
        self.update = build_module(update_name, d_model, UPDATE_REGISTRY)

    def forward(self, x):
        for prop in self.props:
            x = prop(x)
        x = self.update(x)
        return x


class BestDiscovered(nn.Module):
    """Best architecture discovered by automated search."""

    BLOCKS_CONFIG = [{"props": ["lsh_exchange", "hierarchical_pool_s2"], "update": "conv_gru"}, {"props": ["hierarchical_pool_s2", "dilated_conv_d8", "dilated_conv_d32", "depthwise_conv_3"], "update": "squeeze_excite"}, {"props": ["depthwise_conv_3", "depthwise_conv_15"], "update": "polynomial_activation"}, {"props": ["depthwise_conv_5", "spectral_filter", "dilated_conv_d4"], "update": "per_channel_scale"}]
    D_MODEL = 50

    def __init__(self, vocab_size=16, d_model=50, max_len=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_norm = nn.LayerNorm(d_model)

        self.blocks = nn.ModuleList([
            DiscoveredBlock(d_model, b["props"], b["update"])
            for b in self.BLOCKS_CONFIG
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.emb_norm(self.tok_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_norm(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
