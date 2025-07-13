# patch_performer.py  ―― 修正版
import torch, torch.nn as nn, torch.nn.functional as F

# ---------- 1. PatchTokenizer ----------
class PatchTokenizer(nn.Module):
    def __init__(self, patch_size=10, dim=32):
        super().__init__()
        self.conv = nn.Conv2d(1, dim,
                              kernel_size=(22, patch_size),
                              stride=(22, patch_size),
                              bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):                  # x: (B,1,22,1000)
        x = self.conv(x)                  # (B,dim,1,100)  ※1000/10=100
        x = x.squeeze(2).transpose(1, 2)  # (B,100,dim)
        return self.norm(x)

# ---------- 2. Performer Attention (FAVOR+) ----------
class PerformerAttention(nn.Module):
    def __init__(self, dim, heads=4, nb_features=64):
        super().__init__()
        self.h, self.f = heads, nb_features
        self.to_qkv = nn.Linear(dim, dim*3, bias=False)
        self.proj    = nn.Linear(dim, dim)
        self.register_buffer('proj_mat',
            torch.randn(heads, dim//heads, nb_features))

    # φ(x) = ELU(x)+1  のランダム特徴写像だけを返す
    def _phi(self, x):                    # (B,N,H,D) → (B,N,H,F)
        return F.elu(torch.einsum('bnhd,hdf->bnhf', x, self.proj_mat)) + 1

    def forward(self, x):                 # x: (B,N,dim)
        B, N, _ = x.shape
        H, D = self.h, x.size(-1)//self.h

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)      # (B,N,dim)*3
        # split heads
        q = q.view(B, N, H, D)
        k = k.view(B, N, H, D)
        v = v.view(B, N, H, D)   # ★ V はそのまま D 次元

        q_phi = self._phi(q)     # (B,N,H,F)
        k_phi = self._phi(k)     # (B,N,H,F)

        # ① KV 集約 : (B,H,F,D)
        kv = torch.einsum('bnhf,bnhd->bhfd', k_phi, v)
        # ② 正規化項 : (B,N,H)
        z  = 1. / torch.einsum('bnhf,bhf->bnh', q_phi, k_phi.sum(dim=1))
        # ③ 出力 : (B,N,H,D)
        out = torch.einsum('bnhf,bhfd,bnh->bnhd', q_phi, kv, z)
        out = out.reshape(B, N, H*D)                  # (B,N,dim)
        return self.proj(out)

# ---------- 3. Encoder Block ----------
class PerformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = PerformerAttention(dim, heads)
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

# ---------- 4. EEGPatchPerformer ----------
class EEGPatchPerformer(nn.Module):
    def __init__(self, num_classes=2, dim=32, heads=4, layers=2, patch=10):
        super().__init__()
        self.tokenizer = PatchTokenizer(patch, dim)
        self.encoder   = nn.Sequential(*[PerformerBlock(dim, heads) for _ in range(layers)])
        self.norm      = nn.LayerNorm(dim)
        self.pool      = nn.Linear(dim, 1)   # attention pool
        self.fc        = nn.Linear(dim, num_classes)

    def forward(self, x):                    # x:(B,1,22,1000)
        x = self.tokenizer(x)                # (B,100,dim)
        x = self.encoder(x)
        x = self.norm(x)
        w = F.softmax(self.pool(x), 1)       # (B,100,1)
        x = (w.transpose(-1,-2) @ x).squeeze(1)
        return self.fc(x)
