import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MambaBlock(nn.Module):
    """
    Simplified Mamba block for EEG signals with bidirectional processing
    Based on concepts from EEGMamba paper
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand_factor=2, dt_rank="auto", 
                 bias=False, conv_bias=True, inner_layernorms=False):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = int(self.expand_factor * self.dim)
        
        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.dim / 16)
        else:
            self.dt_rank = dt_rank
            
        # Linear projections
        self.in_proj = nn.Linear(self.dim, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize special parameters
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, self.dim, bias=bias)
        
        if inner_layernorms:
            self.dt_layernorm = nn.LayerNorm(self.dt_rank)
            self.B_layernorm = nn.LayerNorm(self.d_state)
            self.C_layernorm = nn.LayerNorm(self.d_state)
        else:
            self.dt_layernorm = self.B_layernorm = self.C_layernorm = None

    def forward(self, x):
        """Forward pass for bidirectional processing"""
        B, L, D = x.shape
        
        # Forward direction
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_inner)
        x_forward, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        
        x_forward = rearrange(x_forward, 'b l d -> b d l')
        x_forward = self.conv1d(x_forward)[:, :, :L]  # depthwise convolution
        x_forward = rearrange(x_forward, 'b d l -> b l d')
        x_forward = F.silu(x_forward)
        
        y_forward = self.ssm(x_forward)
        
        # Backward direction  
        x_backward = torch.flip(x_forward, dims=[1])  # reverse sequence
        y_backward = self.ssm(x_backward)
        y_backward = torch.flip(y_backward, dims=[1])  # reverse back
        
        # Combine forward and backward
        y = y_forward + y_backward
        y = y * F.silu(res)
        output = self.out_proj(y)
        
        return output
    
    def ssm(self, x):
        """Selective State Space Model computation"""
        # x: (b, l, d_inner)
        B, L, D = x.shape
        
        # Get Δ, B, C from x
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*d_state)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B_ssm = self.B_layernorm(B_ssm)
        if self.C_layernorm is not None:
            C_ssm = self.C_layernorm(C_ssm)
            
        dt = self.dt_proj(dt)  # (b, l, d_inner)
        
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Discretize A and B
        dt = F.softplus(dt + self.dt_proj.bias)
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (b, l, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)  # (b, l, d_inner, d_state)
        
        # SSM step
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for i in range(L):
            h = h * dA[:, i] + dB[:, i] * x[:, i].unsqueeze(-1)
            y = torch.sum(h * C_ssm[:, i].unsqueeze(1), dim=-1)  # (b, d_inner)
            ys.append(y)
            
        y = torch.stack(ys, dim=1)  # (b, l, d_inner)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y


# Helper function for tensor rearrangement
def rearrange(tensor, pattern, **axes_lengths):
    """Simplified rearrange function"""
    if pattern == 'b l d -> b d l':
        return tensor.transpose(1, 2)
    elif pattern == 'b d l -> b l d':
        return tensor.transpose(1, 2)
    else:
        raise NotImplementedError(f"Pattern {pattern} not implemented")


class STMambaEncoderLayer(nn.Module):
    """
    Encoder layer combining Spatial-Temporal Mamba with residual connections
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand_factor=2, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = MambaBlock(
            dim=dim, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand_factor=expand_factor
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Mamba block with residual connection
        x = x + self.dropout(self.mamba(self.norm1(x)))
        
        # MLP block with residual connection  
        x = x + self.dropout(self.mlp(self.norm2(x)))
        
        return x


class SpatialTemporalAdaptiveModule(nn.Module):
    """
    Spatial-Temporal Adaptive module for EEG signals
    Handles different channel numbers and sequence lengths
    """
    def __init__(self, n_input_channels, dim, kernel_sizes=[3, 7], 
                 stride=1, padding=1):
        super().__init__()
        
        # Spatial adaptive convolution
        self.spatial_conv = nn.Conv2d(
            n_input_channels, dim // 2, 
            kernel_size=(1, kernel_sizes[0]), 
            stride=(1, stride), 
            padding=(0, padding)
        )
        
        # Temporal convolutions with different kernel sizes
        self.temporal_conv_small = nn.Conv2d(
            dim // 2, dim // 2, 
            kernel_size=(1, kernel_sizes[0]), 
            stride=(1, stride), 
            padding=(0, padding),
            groups=dim // 2
        )
        
        self.temporal_conv_large = nn.Conv2d(
            dim // 2, dim // 2, 
            kernel_size=(1, kernel_sizes[1]), 
            stride=(1, stride), 
            padding=(0, kernel_sizes[1] // 2),
            groups=dim // 2
        )
        
        self.channel_conv = nn.Conv2d(
            dim, dim, 
            kernel_size=(1, 1), 
            stride=(1, 1)
        )
        
        self.norm = nn.BatchNorm2d(dim)
        self.activation = nn.GELU()
        
        # Class token for temporal adaptation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        # x: (B, C, T) -> (B, C, 1, T) for 2D convolution
        B, C, T = x.shape
        x = x.unsqueeze(2)  # (B, C, 1, T)
        
        # Spatial processing - reduce channels first
        x = self.spatial_conv(x)  # (B, dim//2, 1, T)
        
        # Temporal processing with multi-scale kernels
        x_small = self.temporal_conv_small(x)  # (B, dim//2, 1, T)
        x_large = self.temporal_conv_large(x)  # (B, dim//2, 1, T)
        
        # Combine multi-scale features
        x = torch.cat([x_small, x_large], dim=1)  # (B, dim, 1, T)
        
        # Final channel processing
        x = self.channel_conv(x)  # (B, dim, 1, T)
        x = self.norm(x)
        x = self.activation(x)
        
        # Flatten to sequence format
        x = x.squeeze(2).transpose(1, 2)  # (B, T, dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, dim)
        
        return x


class STMambaCCT(nn.Module):
    """
    Spatial-Temporal Mamba Compact Convolutional Transformer for EEG
    Combines STMamba with CCT architecture
    """
    def __init__(self, 
                 n_input_channels=22,
                 sequence_length=1000,
                 dim=256,
                 num_layers=6,
                 num_classes=4,
                 d_state=16,
                 d_conv=4,
                 expand_factor=2,
                 dropout=0.1,
                 kernel_sizes=[3, 7]):
        super().__init__()
        
        self.dim = dim
        self.num_classes = num_classes
        
        # Spatial-Temporal Adaptive Module
        self.st_adaptive = SpatialTemporalAdaptiveModule(
            n_input_channels=n_input_channels,
            dim=dim,
            kernel_sizes=kernel_sizes
        )
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, sequence_length // 4 + 1, dim)  # +1 for cls token
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # STMamba encoder layers
        self.layers = nn.ModuleList([
            STMambaEncoderLayer(
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Final processing
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # x: (B, C, T) - EEG signal
        B = x.shape[0]
        
        # Spatial-temporal adaptive processing
        x = self.st_adaptive(x)  # (B, seq_len, dim)
        
        # Add positional embedding
        if x.size(1) <= self.pos_embedding.size(1):
            x = x + self.pos_embedding[:, :x.size(1), :]
        else:
            # Handle longer sequences by interpolating position embeddings
            pos_emb = F.interpolate(
                self.pos_embedding.transpose(1, 2), 
                size=x.size(1), 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            x = x + pos_emb
            
        x = self.dropout(x)
        
        # Pass through STMamba layers
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        
        # Use class token for classification
        cls_token = x[:, 0]  # (B, dim)
        output = self.classifier(cls_token)
        
        return output


def create_stmamba_cct(n_input_channels=22, sequence_length=1000, num_classes=4):
    """
    Factory function to create STMambaCCT model with default parameters
    """
    model = STMambaCCT(
        n_input_channels=n_input_channels,
        sequence_length=sequence_length,
        dim=256,
        num_layers=6,
        num_classes=num_classes,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        dropout=0.1,
        kernel_sizes=[3, 7]
    )
    return model


if __name__ == "__main__":
    # Test the model
    model = create_stmamba_cct(n_input_channels=22, sequence_length=1000, num_classes=4)
    
    # Create dummy input
    x = torch.randn(2, 22, 1000)  # (batch_size, channels, time)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print("Model created successfully!") 
