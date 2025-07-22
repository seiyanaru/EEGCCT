import torch
import torch.nn as nn
import torch.nn.functional as F

from .cct import Tokenizer
from .stmamba_cct import MambaBlock


class MambaEncoderLayer(nn.Module):
    """Encoder layer using Mamba block and MLP."""
    def __init__(self, dim, d_state=16, d_conv=4, expand_factor=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = MambaBlock(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.mamba(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class MambaTransformer(nn.Module):
    """Transformer style stack of Mamba encoder layers."""
    def __init__(self, dim, num_layers, num_classes, dropout=0.1,
                 positional_embedding='sine', sequence_length=None,
                 d_state=16, d_conv=4, expand_factor=2):
        super().__init__()

        positional_embedding = positional_embedding if positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        self.dim = dim
        self.sequence_length = sequence_length

        assert sequence_length is not None or positional_embedding == 'none', (
            f"Positional embedding is {positional_embedding} but sequence length not specified")

        self.attention_pool = nn.Linear(dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, dim), requires_grad=True)
            else:
                self.positional_emb = nn.Parameter(self.sinusoidal_embedding(sequence_length, dim), requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            MambaEncoderLayer(dim=dim, d_state=d_state, d_conv=d_conv,
                               expand_factor=expand_factor, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        if self.positional_emb is not None:
            x = x + self.positional_emb
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        x = self.fc(x)
        return x

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class MambaCCT(nn.Module):
    """Compact Convolutional Transformer using Mamba blocks."""
    def __init__(self, kernel_sizes, stride, padding,
                 pooling_kernel_size, pooling_stride, pooling_padding,
                 n_conv_layers, n_input_channels, in_planes, activation,
                 max_pool, conv_bias,
                 dim, num_layers, num_classes,
                 dropout=0.1, positional_emb='sine',
                 d_state=16, d_conv=4, expand_factor=2):
        super().__init__()

        self.tokenizer = Tokenizer(
            kernel_sizes=kernel_sizes, stride=stride, padding=padding,
            pooling_kernel_size=pooling_kernel_size, pooling_stride=pooling_stride, pooling_padding=pooling_padding,
            n_conv_layers=n_conv_layers, n_input_channels=n_input_channels, n_output_channels=dim,
            in_planes=in_planes, activation=activation,
            max_pool=max_pool, conv_bias=conv_bias
        )

        self.transformer = MambaTransformer(
            dim=dim, num_layers=num_layers, num_classes=num_classes,
            dropout=dropout, positional_embedding=positional_emb,
            sequence_length=self.tokenizer.sequence_length(n_channels=1, height=22, width=1000),
            d_state=d_state, d_conv=d_conv, expand_factor=expand_factor,
        )

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.transformer(x)
        return x

