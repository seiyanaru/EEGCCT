import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------------------------------------------------------------------
# 1. Depth‑wise separable conv block
# ---------------------------------------------------------------------
class DWConvBlock(nn.Module):
    """Depth‑wise → point‑wise → BN → SiLU"""
    def __init__(self, in_c: int, out_c: int, k_hw: tuple[int, int], *, dilation: int = 1):
        super().__init__()
        pad_h = (k_hw[0] - 1) // 2 * dilation
        pad_w = (k_hw[1] - 1) // 2 * dilation
        self.dw = nn.Conv2d(in_c, in_c, k_hw, stride=1,
                            padding=(pad_h, pad_w), groups=in_c,
                            dilation=dilation, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))

# ---------------------------------------------------------------------
# 2. Tokenizer :  空間→周波数多帯域→AvgPool(1/4)→拡張受容野
#                 出力 shape = (B, seq≈250, dim)
# ---------------------------------------------------------------------
class TinyTokenizer(nn.Module):

    def __init__(self, dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            # 空間 22×1 (depth‑wise + point‑wise)
            DWConvBlock(1, 16, (22, 1)),
            # 時系列 1×25 （β帯域中心）
            DWConvBlock(16, 32, (1, 25), dilation=1),
            # 時間方向 1/4 に間引く
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            # dilated conv で 50 サンプル RF 拡大
            DWConvBlock(32, dim, (1, 25), dilation=2)
        )
        self.flatten = nn.Flatten(2, 3)

    # 推論前に系列長を知りたいときに利用
    def seq_len(self, H: int = 22, W: int = 1000) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, H, W)
            return self.forward(x).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # B,1,22,1000
        x = self.conv(x)                                # B,dim,1,250
        x = self.flatten(x).transpose(1, 2)             # B,250,dim
        return x

# ---------------------------------------------------------------------
# 3. Window / Shifted‑Window Attention (Swin‑style)
# ---------------------------------------------------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, window_size: int = 32, *, shift: bool = False):
        super().__init__()
        self.h, self.w, self.shift = heads, window_size, shift
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj   = nn.Linear(dim, dim)
        self.scale  = (dim // heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B,N,C)
        B, N, C = x.shape
        if self.shift:
            x = torch.roll(x, -self.w // 2, dims=1)

        pad = (self.w - N % self.w) % self.w
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        n_win = x.shape[1] // self.w

        x = rearrange(x, 'b (nw w) c -> (b nw) w c', w=self.w)
        qkv = self.to_qkv(x).reshape(-1, self.w, 3, self.h, C // self.h)
        q, k, v = qkv.unbind(2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        out  = (attn @ v).transpose(1, 2).reshape(-1, self.w, C)
        out  = self.proj(out)
        out  = rearrange(out, '(b nw) w c -> b (nw w) c', nw=n_win)[:, :N]
        if self.shift:
            out = torch.roll(out,  self.w // 2, dims=1)
        return out

# ---------------------------------------------------------------------
# 4. Encoder Block (Attn + MLP)
# ---------------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, *, window: int = 32, shift: bool = False, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, heads, window_size=window, shift=shift)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ---------------------------------------------------------------------
# 5. Tiny‑EEGCCT‑DW main network
# ---------------------------------------------------------------------
class TinyEEGCCT_DW(nn.Module):
    def __init__(self, *, num_classes: int = 2, dim: int = 32, heads: int = 4,
                 layers: int = 2, window: int = 32):
        super().__init__()
        self.tokenizer = TinyTokenizer(dim)

        enc = []
        for l in range(layers):
            enc.append(EncoderBlock(dim, heads, window=window, shift=bool(l % 2)))
        self.encoder = nn.Sequential(*enc)

        self.norm = nn.LayerNorm(dim)
        self.pool = nn.Linear(dim, 1)   # attention pooling weight
        self.fc   = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,1,22,1000)
        x = self.tokenizer(x)            # (B,N,dim)
        x = self.encoder(x)
        x = self.norm(x)
        w = F.softmax(self.pool(x), dim=1)  # (B,N,1)
        x = (w.transpose(-1, -2) @ x).squeeze(1)
        return self.fc(x)
