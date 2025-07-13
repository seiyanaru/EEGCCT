import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ── ① Multi‐Scale Temporal Conv ──
class MultiScaleTemporal(nn.Module):
    def __init__(self, out_ch=8):
        super().__init__()
        # 異なる時間カーネル長で並列に畳み込み
        ks = [25, 51, 101]
        self.branches = nn.ModuleList([
            nn.Conv2d(1, out_ch, kernel_size=(1, k), padding=(0, k//2), bias=False)
            for k in ks
        ])
    def forward(self, x):
        # x: (B,1,22,1000)
        # 各枝から (B,out_ch,22,1000) が返るのでチャネル方向に concat → (B,24,22,1000)
        return torch.cat([b(x) for b in self.branches], dim=1)

# ── ② Depthwise Spatial Conv ──
class SpatialDW(nn.Module):
    def __init__(self, in_ch=24, token_dim=32):
        super().__init__()
        # 電極間の空間相関を捉える Depthwise → Pointwise
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=(22,1), groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, token_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(token_dim)
    def forward(self, x):
        # x: (B,24,22,1000)
        x = self.dw(x)      # → (B,24,1,1000)
        x = self.pw(x)      # → (B,32,1,1000)
        return self.bn(x)   # → (B,32,1,1000)

# ── ③ Windowed Attention ──
class WindowAttention(nn.Module):
    def __init__(self, dim, heads=4, window_size=50, shift=False):
        super().__init__()
        self.heads = heads
        self.window_size = window_size
        self.shift = shift
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj   = nn.Linear(dim, dim)
        self.scale  = (dim // heads) ** -0.5

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        # (1) シフト
        if self.shift:
            x = torch.roll(x, -self.window_size//2, dims=1)
        # (2) パディングして window_size の倍数に
        pad_len = (self.window_size - N % self.window_size) % self.window_size
        if pad_len:
            x = F.pad(x, (0,0,0,pad_len))
        # (3) 分割・畳み込み
        nw = x.shape[1] // self.window_size
        xw = rearrange(x, 'b (nw w) c -> (b nw) w c', nw=nw, w=self.window_size)
        qkv = self.to_qkv(xw) \
              .reshape(-1, self.window_size, 3, self.heads, C//self.heads) \
              .permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = (attn @ v) \
               .transpose(1,2) \
               .reshape(-1, self.window_size, C)
        out  = self.proj(out)
        # (4) 再結合・逆シフト・切り出し
        x = rearrange(out, '(b nw) w c -> b (nw w) c', b=B, nw=nw, w=self.window_size)
        if self.shift:
            x = torch.roll(x, self.window_size//2, dims=1)
        if pad_len:
            x = x[:, :N, :].contiguous()
        return x  # → (B, N, C)

# ── ④ Performer Encoder Block ──
class PerformerBlock(nn.Module):
    def __init__(self, dim, heads, window_size, shift):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, heads, window_size, shift)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ── ⑤ EEGHybridPerformer 本体 ──
class EEGHybridPerformer(nn.Module):
    def __init__(self,
                 num_classes:int = 2,
                 ms_out_ch:int  = 8,
                 token_dim:int  = 32,
                 heads:int      = 4,
                 layers:int     = 2,
                 window_size:int= 50,
                 patch_pool:int = 5):
        super().__init__()
        # ① Multi‐Scale Conv
        self.ms   = MultiScaleTemporal(out_ch=ms_out_ch)   # → (B,24,22,1000)
        # ② Depthwise Spatial
        self.spat = SpatialDW(in_ch=ms_out_ch*3, token_dim=token_dim)  
        # ③ Patch ダウンサンプリング
        self.pool = nn.AvgPool2d(kernel_size=(1,patch_pool),
                                 stride=(1,patch_pool))    # 1000→1000/5=200
        # ④ Positional Embedding + 正規化
        self.pos  = nn.Parameter(torch.zeros(1, 1000//patch_pool, token_dim))
        self.norm_tok = nn.LayerNorm(token_dim)
        # ⑤ Windowed Performer Blocks
        self.blocks = nn.Sequential(*[
            PerformerBlock(token_dim, heads, window_size, shift=bool(i%2))
            for i in range(layers)
        ])
        # ⑥ 最終 LayerNorm + Attention Pool + FC
        self.norm = nn.LayerNorm(token_dim)
        self.pool_attn = nn.Linear(token_dim, 1)
        self.fc  = nn.Linear(token_dim, num_classes)

    def forward(self, x):
        # x: (B,1,22,1000)
        x = self.ms(x)                       # → (B,24,22,1000)
        x = self.spat(x)                     # → (B,32,1,1000)
        x = self.pool(x)                     # → (B,32,1,200)
        x = x.squeeze(2).transpose(1,2)      # → (B,200,32)
        x = self.norm_tok(x + self.pos)      # +PosEmb
        x = self.blocks(x)                   # → (B,200,32)
        x = self.norm(x)                     # → (B,200,32)
        w = F.softmax(self.pool_attn(x), dim=1)    # → (B,200,1)
        x = (w.transpose(-1,-2) @ x).squeeze(1)     # → (B,32)
        return self.fc(x)                    # → (B,2)

# ── 動作確認 ──
if __name__ == "__main__":
    model = EEGHybridPerformer()
    dummy = torch.randn(4,1,22,1000)
    out = model(dummy)
    print("output shape:", out.shape)    # → (4,2)
