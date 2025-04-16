import torch
import torch.nn as nn
from typing import Literal, Tuple


class NdLinearGated(nn.Module):
    """A neural network module that performs N-dimensional linear transformations with gating mechanisms.
    
    This module applies linear transformations to N-dimensional input tensors while incorporating
    gating mechanisms to control the flow of information. The gating can be applied to all dimensions,
    only the first dimension, or the top-k most important dimensions based on their standard deviation.
    
    Args:
        input_dims (tuple): Dimensions of the input tensor for each transformation layer
        hidden_size (tuple): Target dimensions for each transformation layer
        transform_outer (bool, optional): If True, transforms from outer to inner dimensions.
            If False, transforms from inner to outer dimensions. Defaults to True.
        gating_mode (Literal["soft", "hard"], optional): Type of gating mechanism to use.
            "soft" uses continuous gating values, "hard" uses binary gating. Defaults to "soft".
        gating_hidden_dim (int, optional): Hidden dimension size for the gating networks.
            Defaults to 16.
        gated_modes (Literal["all", "first", "topk"], optional): Specifies which dimensions
            to apply gating to. "all" applies gating to all dimensions, "first" only to the
            first dimension, and "topk" to the top-k dimensions with highest standard deviation.
            Defaults to "all".
    
    Raises:
        Exception: If the length of input_dims does not match the length of hidden_size.
    
    Example:
        >>> module = NdLinearGated(input_dims=(32, 64), hidden_size=(64, 128))
        >>> x = torch.randn(10, 32, 64)
        >>> output = module(x)
    """
    def __init__(self, 
                 input_dims: tuple, 
                 hidden_size: tuple, 
                 transform_outer=True,
                 gating_mode: Literal["soft", "hard"] = "soft",
                 gating_hidden_dim: int = 16,
                 gated_modes: Literal["all", "first", "topk"] = "all"):
        super(NdLinearGated, self).__init__()

        if len(input_dims) != len(hidden_size):
            raise Exception("Input shape and hidden shape do not match.")

        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.num_layers = len(input_dims)
        self.transform_outer = transform_outer
        self.gating_mode = gating_mode
        self.gating_hidden_dim = gating_hidden_dim
        self.gated_modes = gated_modes
        
        self.align_layers = nn.ModuleList([
            nn.Linear(input_dims[i], hidden_size[i]) for i in range(self.num_layers)
        ])
        
        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dims[i], gating_hidden_dim),
                nn.ReLU(),
                nn.Linear(gating_hidden_dim, 1),
                nn.Sigmoid()
            ) for i in range(self.num_layers)
        ])
        
        self.identity_projections = nn.ModuleList([
            nn.Linear(input_dims[i], hidden_size[i]) if input_dims[i] != hidden_size[i] else nn.Identity()
            for i in range(self.num_layers)
        ])
        
        self.topk_modes = None
        self.first_batch_processed = False
        
    def _compute_topk_modes(self, X):
        mode_stds = []
        for i in range(self.num_layers):
            transpose_dim = i + 1 if self.transform_outer else self.num_layers - i
            X_transposed = torch.transpose(X, transpose_dim, self.num_layers)
            X_mean = X_transposed.mean(dim=tuple(range(len(X_transposed.shape) - 1)))
            mode_stds.append(X_mean.std().item())
        
        sorted_modes = sorted(range(len(mode_stds)), key=lambda i: mode_stds[i], reverse=True)
        return sorted_modes[:2] 
        
    def forward(self, X):
        num_transforms = self.num_layers
        
        if self.gated_modes == "topk" and not self.first_batch_processed:
            self.topk_modes = self._compute_topk_modes(X)
            self.first_batch_processed = True
            
        for i in range(num_transforms):
            if self.transform_outer:
                layer_idx = i
                transpose_dim = i + 1
            else:
                layer_idx = num_transforms - (i+1)
                transpose_dim = num_transforms - i
                
            apply_gating = False
            if self.gated_modes == "all":
                apply_gating = True
            elif self.gated_modes == "first" and i == 0:
                apply_gating = True
            elif self.gated_modes == "topk" and self.topk_modes and layer_idx in self.topk_modes:
                apply_gating = True
                
            X_original = X.clone()
            
            X = torch.transpose(X, transpose_dim, num_transforms).contiguous()
            
            X_size = X.shape[:-1]
            
            X_flat = X.view(-1, X.shape[-1])
            
            X_transformed = self.align_layers[layer_idx](X_flat)
            
            if apply_gating:
                X_mean = X_flat.mean(dim=0, keepdim=True)
                
                gate = self.gate_networks[layer_idx](X_mean)
                
                X_transformed = X_transformed.view(*X_size, X_transformed.shape[-1])
                
                X_identity = torch.transpose(X_original, transpose_dim, num_transforms).contiguous()
                
                X_identity_flat = X_identity.view(-1, X_identity.shape[-1])
                
                if X_transformed.shape[-1] != X_identity_flat.shape[-1]:
                    identity_flat = self.identity_projections[layer_idx](X_identity_flat)
                else:
                    identity_flat = X_identity_flat
                
                if self.gating_mode == "soft":
                    X_flat = gate * X_transformed.view(-1, X_transformed.shape[-1]) + (1 - gate) * identity_flat
                else: 
                    X_flat = torch.where(gate > 0.5, 
                                         X_transformed.view(-1, X_transformed.shape[-1]),
                                         identity_flat)
                    
                X = X_flat.view(*X_size, X_flat.shape[-1])
            else:
                X = X_transformed.view(*X_size, X_transformed.shape[-1])
            
            X = torch.transpose(X, transpose_dim, num_transforms).contiguous()
            
        return X 