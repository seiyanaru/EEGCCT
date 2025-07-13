import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------------------------------------------------------------------
# 1. Fixed FIR kernels (crude sinc-based band-pass)
# ---------------------------------------------------------------------
_BAND_K = 51  # kernel length (odd)
_DEF_COEFFS = {}
for name, (f1, f2) in {
        'alpha': (8, 14), 'beta': (15, 30), 'lowg': (31, 45), 'bb': (0, 45)}.items():
    t = torch.arange(_BAND_K) - (_BAND_K - 1) / 2
    sinc = lambda f: torch.where(t == 0,
                                 torch.tensor(math.tau * f),
                                 torch.sin(math.tau * f * t) / t)
    h = (sinc(f2/250) - sinc(f1/250)) if f1 > 0 else sinc(f2/250)
    _DEF_COEFFS[name] = (h / h.sum()).view(1, 1, 1, -1)  # shape=(1,1,1,_BAND_K)

# ---------------------------------------------------------------------
# 2. SpectralDepthwiseTokenizer
# ---------------------------------------------------------------------
class SpectralDepthwiseTokenizer(nn.Module):
    """Four fixed band-pass depthwise convs + pointwise → BN → GELU → AvgPool(¼) → Flatten"""
    def __init__(self, dim: int = 32):
        super().__init__()
        # register fixed FIR weights as buffers
        self.register_buffer('alpha', _DEF_COEFFS['alpha'])
        self.register_buffer('beta',  _DEF_COEFFS['beta'])
        self.register_buffer('lowg',  _DEF_COEFFS['lowg'])
        self.register_buffer('bb',    _DEF_COEFFS['bb'])
        # learnable pointwise to dim
        self.pw   = nn.Conv2d(4, dim, 1, bias=False)
        self.bn   = nn.BatchNorm2d(dim)
        self.act  = nn.GELU()
        # pool time axis: 1000→250
        self.pool = nn.AvgPool2d((1, 4), (1, 4))
        self.flatten = nn.Flatten(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,22,1000)
        x_cat = torch.cat([
            F.conv2d(x, self.alpha, groups=1, padding=(0,_BAND_K//2)),
            F.conv2d(x, self.beta,  groups=1, padding=(0,_BAND_K//2)),
            F.conv2d(x, self.lowg,  groups=1, padding=(0,_BAND_K//2)),
            F.conv2d(x, self.bb,    groups=1, padding=(0,_BAND_K//2))
        ], dim=1)                               # (B,4,22,1000)
        x = self.pool(self.act(self.bn(self.pw(x_cat))))  # (B,dim,22,250)
        x = self.flatten(x).transpose(1, 2)               # (B,22*250,dim)= (B,5500,dim)
        return x

# ---------------------------------------------------------------------
# 3. Performer Linear Attention (FAVOR+)  ------------------------------
# ---------------------------------------------------------------------
class PerformerAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, nb_features: int = 64):
        super().__init__()
        self.heads = heads
        self.nb_features = nb_features
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj   = nn.Linear(dim, dim)
        # random projection matrix for FAVOR+
        self.register_buffer('proj_matrix', torch.randn(heads, dim//heads, nb_features))

    def _favor(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) → map to (B, N, heads, features) via Gaussian kernel
        B, N, C = x.shape
        H = self.heads
        D = C // H
        x = x.view(B, N, H, D)
        # compute phi(x) = elu(x @ P) + 1
        x_proj = torch.einsum('bnhd,hdf->bnhf', x, self.proj_matrix)
        return F.elu(x_proj) + 1  # (B, N, heads, nb_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # FAVOR+ mapping for Q, K
        Q = self._favor(q)  # (B, N, H, F)
        K = self._favor(k)  # (B, N, H, F)
        # reshape V to heads dims but keep head_dim
        B, N, C = v.shape
        H = self.heads
        D = C // H
        V = v.view(B, N, H, D)  # (B, N, H, D)
        # compute KV: sum over tokens → (heads, features, head_dim)
        KV = torch.einsum('bnhf,bnhd->hfd', K, V)
        # normalization Z per token & head → (B, N, H)
        Z = 1.0 / torch.einsum('bnhf,hfd->bnh', Q, KV)
        # linear attention: out = (Q * Z) @ KV → (B, N, H, D)
        out = torch.einsum('bnhf,hfd,bnh->bnhd', Q, KV, Z)
        # merge heads
        out = out.contiguous().view(B, N, C)  # (B, N, C)
        return self.proj(out)

# ---------------------------------------------------------------------
# 4. Encoder Block -----------------------------------------------------
# ---------------------------------------------------------------------
class PerformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = PerformerAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(dim*mlp_ratio, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ---------------------------------------------------------------------
# 5. MB-Performer-EEG main class ---------------------------------------
# ---------------------------------------------------------------------
class MBPerformerEEG(nn.Module):
    def __init__(self, num_classes: int = 2, dim: int = 32, heads: int = 4, layers: int = 2):
        super().__init__()
        self.tokenizer = SpectralDepthwiseTokenizer(dim)
        enc = [PerformerBlock(dim, heads) for _ in range(layers)]
        self.encoder = nn.Sequential(*enc)
        self.norm    = nn.LayerNorm(dim)
        self.pool    = nn.Linear(dim, 1)  # attention pooling
        self.fc      = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 22, 1000)
        x = self.tokenizer(x)  # (B, N, dim)
        x = self.encoder(x)
        x = self.norm(x)
        w = F.softmax(self.pool(x), dim=1)      # (B, N, 1)
        x = (w.transpose(-1, -2) @ x).squeeze(1)
        return self.fc(x)
