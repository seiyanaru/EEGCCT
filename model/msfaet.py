import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce

class FrequencyAwareTokenizer(nn.Module):
    """
    周波数帯域を考慮したTokenizer
    複数の周波数帯域（α、β、γ波）を同時に処理
    """
    def __init__(self, dim=64, n_channels=22, sample_rate=250):
        super().__init__()
        self.dim = dim
        self.n_channels = n_channels
        
        # EEG周波数帯域定義
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        # 各周波数帯域用のFIRフィルタを生成
        self.register_buffers()
        
        # 空間畳み込み（チャネル間相互作用）
        self.spatial_conv = nn.Conv2d(1, 16, (n_channels, 1), bias=False)
        
        # 各周波数帯域の特徴を統合
        self.freq_fusion = nn.Conv2d(len(self.freq_bands), dim//2, 1, bias=False)
        
        # 時空間特徴統合
        self.spatiotemporal_conv = nn.Conv2d(dim//2 + 16, dim, (1, 25), 
                                           padding=(0, 12), bias=False)
        
        # バッチ正規化とアクティベーション
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        
        # 適応的プーリング（可変長シーケンス対応）
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        
    def register_buffers(self):
        """各周波数帯域のFIRフィルタを生成してバッファに登録"""
        kernel_size = 51
        for name, (low, high) in self.freq_bands.items():
            kernel = self.create_bandpass_filter(low, high, kernel_size)
            self.register_buffer(f'filter_{name}', kernel)
    
    def create_bandpass_filter(self, low_freq, high_freq, kernel_size, sample_rate=250):
        """バンドパスフィルタカーネルを生成"""
        nyquist = sample_rate / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        n = torch.arange(kernel_size) - (kernel_size - 1) / 2
        
        # Sinc関数を使用したバンドパスフィルタ
        if low_freq > 0:
            h = (torch.sin(math.pi * high_norm * n) - torch.sin(math.pi * low_norm * n)) / (math.pi * n)
            h[kernel_size // 2] = high_norm - low_norm
        else:
            h = torch.sin(math.pi * high_norm * n) / (math.pi * n)
            h[kernel_size // 2] = high_norm
        
        # ハミング窓を適用
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(kernel_size) / (kernel_size - 1))
        h = h * window
        h = h / h.sum()
        
        return h.view(1, 1, 1, -1)
    
    def forward(self, x):
        B, C, H, W = x.shape  # (batch, 1, 22, time_samples)
        
        # 空間特徴抽出
        spatial_feat = self.spatial_conv(x)  # (B, 16, 1, W)
        
        # 各周波数帯域でフィルタリング
        freq_features = []
        for name in self.freq_bands.keys():
            filt = getattr(self, f'filter_{name}')
            freq_feat = F.conv2d(x, filt, padding=(0, filt.size(-1)//2))
            freq_features.append(freq_feat)
        
        freq_stack = torch.cat(freq_features, dim=1)  # (B, n_bands, H, W)
        freq_feat = self.freq_fusion(freq_stack)  # (B, dim//2, H, W)
        
        # 空間と周波数特徴を結合
        freq_feat = F.adaptive_avg_pool2d(freq_feat, (1, W))  # (B, dim//2, 1, W)
        combined_feat = torch.cat([spatial_feat, freq_feat], dim=1)  # (B, dim//2+16, 1, W)
        
        # 時空間畳み込み
        x = self.spatiotemporal_conv(combined_feat)  # (B, dim, 1, W)
        x = self.act(self.bn(x))
        
        # 適応的プーリングでシーケンス長を調整
        x = F.adaptive_avg_pool2d(x, (1, W//4))  # 1/4にダウンサンプリング
        
        # Flatten and transpose for transformer
        x = x.flatten(2, 3).transpose(1, 2)  # (B, seq_len, dim)
        
        return x

class EfficientAttention(nn.Module):
    """
    Linear Attention with improved efficiency
    Performer-like attention with dynamic features
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Linear attention用のランダム特徴量
        self.nb_features = max(32, dim // 4)
        self.register_buffer('random_features', 
                           torch.randn(num_heads, head_dim, self.nb_features))
        
    def kernel_feature_map(self, x):
        """ReLU kernel feature map"""
        x_mapped = torch.einsum('bhnd,hdf->bhnf', x, self.random_features)
        return F.relu(x_mapped)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        # Linear attention computation
        q_prime = self.kernel_feature_map(q)  # (B, num_heads, N, nb_features)
        k_prime = self.kernel_feature_map(k)  # (B, num_heads, N, nb_features)
        
        # Compute attention efficiently
        kv = torch.einsum('bhnf,bhnd->bhfd', k_prime, v)  # (B, num_heads, nb_features, head_dim)
        qkv = torch.einsum('bhnf,bhfd->bhnd', q_prime, kv)  # (B, num_heads, N, head_dim)
        
        # Normalization
        z = torch.einsum('bhnf,bhf->bhn', q_prime, k_prime.sum(dim=2))  # (B, num_heads, N)
        z = z.unsqueeze(-1)  # (B, num_heads, N, 1)
        
        x = qkv / (z + 1e-6)  # Avoid division by zero
        x = x.transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class AdaptiveMLP(nn.Module):
    """
    Adaptive MLP with dynamic expansion ratio
    """
    def __init__(self, dim, hidden_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)
        
        # Gating mechanism for adaptive processing
        self.gate = nn.Linear(dim, hidden_dim)
        
    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        gate = torch.sigmoid(self.gate(shortcut))
        x = self.act(x) * gate  # Gated activation
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MSFAETBlock(nn.Module):
    """
    Multi-Scale Frequency-Aware EEG Transformer Block
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EfficientAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                     attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = AdaptiveMLP(dim=dim, hidden_dim=mlp_hidden_dim, 
                              act_layer=act_layer, drop=drop)
        
        # Multi-scale processing
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
    def forward(self, x):
        # Standard transformer block
        x = x + self.attn(self.norm1(x))
        
        # Add local convolution for multi-scale features
        x_conv = x.transpose(1, 2)  # (B, dim, seq_len)
        x_conv = self.local_conv(x_conv).transpose(1, 2)  # (B, seq_len, dim)
        
        x = x + self.mlp(self.norm2(x + x_conv))
        return x

class MSFAET(nn.Module):
    """
    Multi-Scale Frequency-Aware EEG Transformer
    CCTの改良版：効率的なAttention、周波数認識、マルチスケール処理
    """
    def __init__(self, n_channels=22, n_classes=2, dim=64, depth=4, num_heads=8, 
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        
        # Frequency-aware tokenizer
        self.tokenizer = FrequencyAwareTokenizer(dim=dim, n_channels=n_channels)
        
        # Positional embedding (learnable)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            MSFAETBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        
        # Final processing
        self.norm = nn.LayerNorm(dim)
        
        # Multi-head attention pooling for better feature aggregation
        self.attention_pool = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Classification head with residual connection
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(dim // 2, n_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # Tokenization with frequency awareness
        x = self.tokenizer(x)  # (B, seq_len, dim)
        B, seq_len, dim = x.shape
        
        # Add positional encoding
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Attention pooling with CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x_with_cls = torch.cat([cls_tokens, x], dim=1)  # (B, seq_len+1, dim)
        
        attn_out, _ = self.attention_pool(cls_tokens, x_with_cls, x_with_cls)
        pooled_feat = attn_out.squeeze(1)  # (B, dim)
        
        # Classification
        logits = self.head(pooled_feat)
        
        return logits 
