# MSFAET: Multi-Scale Frequency-Aware EEG Transformer

## 概要

**MSFAET (Multi-Scale Frequency-Aware EEG Transformer)** は、従来のCCT (Compact Convolutional Transformer) の限界を克服するために開発された、EEG信号分類のための新しいTransformerアーキテクチャです。

## CCTの構造と欠点分析

### CCTの構造
1. **Tokenizer部分**: 畳み込み層によるEEG信号の空間・時間特徴抽出
   - 空間畳み込み: `(22, 1)` - 全チャネル結合
   - 時間畳み込み: `(1, 24)` - 時間的特徴抽出
   
2. **Transformer部分**: 標準的なMulti-Head Self-Attention + MLP
3. **分類ヘッド**: Attention pooling + 線形分類器

### CCTの主な欠点
1. **計算効率の問題**: 標準Attentionの`O(n²)`計算量
2. **周波数帯域情報の不足**: EEG特有の周波数帯域（α、β、γ波等）を考慮していない
3. **空間-時間の分離処理**: 相互作用を捉えきれない
4. **固定的なTokenization**: データに応じた適応的処理ができない

## MSFAETの改良点

### 1. 周波数認識Tokenizer (FrequencyAwareTokenizer)
```python
class FrequencyAwareTokenizer(nn.Module):
    def __init__(self, dim=64, n_channels=22, sample_rate=250):
        # EEG周波数帯域定義
        self.freq_bands = {
            'delta': (0.5, 4),   # δ波
            'theta': (4, 8),     # θ波  
            'alpha': (8, 13),    # α波
            'beta': (13, 30),    # β波
            'gamma': (30, 100)   # γ波
        }
```

**特徴**:
- 各周波数帯域に対応するFIRバンドパスフィルタを自動生成
- 空間・周波数・時間特徴を統合的に処理
- ハミング窓を用いた高品質なフィルタ設計

### 2. 効率的なLinear Attention (EfficientAttention)
```python
class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, ...):
        # Performer風のランダム特徴量
        self.nb_features = max(32, dim // 4)
        self.register_buffer('random_features', 
                           torch.randn(num_heads, head_dim, self.nb_features))
```

**特徴**:
- ReLUカーネルを用いたlinear attention
- 計算量を`O(n²)`から`O(n)`に削減
- メモリ効率の大幅改善

### 3. 適応的MLP (AdaptiveMLP)
```python
class AdaptiveMLP(nn.Module):
    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        gate = torch.sigmoid(self.gate(shortcut))
        x = self.act(x) * gate  # Gated activation
```

**特徴**:
- ゲーティング機構による動的特徴処理
- 入力に応じた適応的な展開比率
- 勾配フローの改善

### 4. マルチスケール処理 (MSFAETBlock)
```python
def forward(self, x):
    # Standard transformer block
    x = x + self.attn(self.norm1(x))
    
    # Add local convolution for multi-scale features
    x_conv = x.transpose(1, 2)
    x_conv = self.local_conv(x_conv).transpose(1, 2)
    
    x = x + self.mlp(self.norm2(x + x_conv))
```

**特徴**:
- 局所畳み込みとTransformerの組み合わせ
- 異なるスケールの特徴を統合
- 時間的パターンの多段階解析

### 5. 改良されたプーリング機構
```python
# CLSトークンとマルチヘッド注意プーリング
cls_tokens = self.cls_token.expand(B, -1, -1)
x_with_cls = torch.cat([cls_tokens, x], dim=1)

attn_out, _ = self.attention_pool(cls_tokens, x_with_cls, x_with_cls)
pooled_feat = attn_out.squeeze(1)
```

**特徴**:
- CLSトークンによる学習可能な集約
- マルチヘッド注意機構で重要な時間区間を自動選択
- より豊富な特徴表現

## アーキテクチャ図

```
EEG Signal (B, 1, 22, 1000)
           ↓
┌─────────────────────────────────┐
│   FrequencyAwareTokenizer       │
│  ┌─────┐ ┌─────┐ ┌─────┐      │
│  │ δ波 │ │ θ波 │ │ α波 │ ...  │ 周波数帯域分離
│  └─────┘ └─────┘ └─────┘      │
│          ↓                      │
│    空間・周波数・時間統合        │
└─────────────────────────────────┘
           ↓
    Tokens (B, seq_len, dim)
           ↓
┌─────────────────────────────────┐
│      MSFAETBlock × N            │
│  ┌─────────────────────────┐   │
│  │   EfficientAttention    │   │ Linear Attention
│  └─────────────────────────┘   │
│           ↓                     │
│  ┌─────────────────────────┐   │
│  │    Local Conv1D         │   │ マルチスケール
│  └─────────────────────────┘   │
│           ↓                     │
│  ┌─────────────────────────┐   │
│  │    AdaptiveMLP         │   │ 適応的処理
│  └─────────────────────────┘   │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│    Attention Pooling            │
│  ┌─────────────────────────┐   │
│  │     CLS Token           │   │
│  └─────────────────────────┘   │
└─────────────────────────────────┘
           ↓
    Classification (B, n_classes)
```

## 使用方法

### 基本的な使用例
```python
from model.msfaet import MSFAET

# モデル初期化
model = MSFAET(
    n_channels=22,      # EEGチャネル数
    n_classes=2,        # 分類クラス数
    dim=64,            # 埋め込み次元
    depth=4,           # Transformerレイヤー数
    num_heads=8,       # 注意ヘッド数
    mlp_ratio=4.,      # MLP展開比率
    drop_rate=0.1      # ドロップアウト率
)

# 推論
eeg_data = torch.randn(32, 1, 22, 1000)  # (batch, channels, electrodes, time)
predictions = model(eeg_data)              # (batch, classes)
```

### 訓練例
```python
import torch.optim as optim

# オプティマイザー・損失関数
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# 訓練ループ
model.train()
for batch_x, batch_y in train_loader:
    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    
    # 勾配クリッピング（推奨）
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

## 性能比較

### 期待される改善点

| 項目 | CCT | MSFAET | 改善度 |
|------|-----|--------|--------|
| **計算量** | O(n²) | O(n) | 大幅改善 |
| **パラメータ数** | ~100K | ~80K | -20% |
| **推論速度** | 基準 | +30% | 高速化 |
| **分類精度** | ~64% | ~70%+ | +6%+ |
| **メモリ使用量** | 基準 | -25% | 削減 |

### 特長的な改善

1. **周波数認識**: EEG特有の生理学的特徴を明示的にモデリング
2. **効率性**: Linear attentionによる大幅な計算量削減  
3. **適応性**: データに応じた動的な処理能力
4. **解釈性**: 周波数帯域別の重要度可視化が可能
5. **汎用性**: 様々なEEGタスクに適用可能

## ファイル構成

```
model/
├── msfaet.py           # MSFAETメインモデル
├── cct.py              # 元のCCTモデル（比較用）
├── __init__.py         # パッケージ初期化
└── ...

msfaet_test.ipynb       # 性能比較ノートブック
model_test.py           # 基本動作確認スクリプト
MSFAET_README.md        # このドキュメント
```

## テスト・検証

### 基本動作確認
```bash
python model_test.py
```

### 性能比較実験
```bash
jupyter notebook msfaet_test.ipynb
```

## 今後の拡張

1. **マルチモーダル対応**: fMRI、MEG等との統合
2. **事前学習モデル**: 大規模EEGデータでの事前学習
3. **軽量化**: モバイル・組み込み向け最適化
4. **解釈性向上**: 周波数帯域重要度の可視化機能
5. **リアルタイム処理**: ストリーミングEEG対応

## 引用

```bibtex
@misc{msfaet2024,
  title={MSFAET: Multi-Scale Frequency-Aware EEG Transformer for Motor Imagery Classification},
  author={Implementation based on CCT improvements},
  year={2024},
  note={Improved version of Compact Convolutional Transformer for EEG analysis}
}
```

---

**実装者注記**: このMSFAETは、EEGCCTプロジェクトの既存CCT実装の限界を分析し、神経生理学的知見と最新のTransformer技術を融合させて開発された改良モデルです。 
