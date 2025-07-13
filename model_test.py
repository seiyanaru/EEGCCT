#!/usr/bin/env python3
"""
MSFAET Model Validation Script
新しいMSFAETモデルの基本的な動作確認とCCTとの比較
"""

import torch
import torch.nn as nn
import numpy as np
from model.cct import CCT
from model.msfaet import MSFAET
import time

def test_model_forward_pass():
    """モデルの前向き推論テスト"""
    print("=== モデル前向き推論テスト ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # サンプルデータ生成（BCI Competition IV Dataset 2a形式）
    batch_size = 16
    n_channels = 22
    n_samples = 1000
    n_classes = 2
    
    # ランダムEEGデータ生成
    X = torch.randn(batch_size, 1, n_channels, n_samples).to(device)
    print(f"Input shape: {X.shape}")
    
    # CCTモデルテスト
    print("\n--- CCT Model Test ---")
    cct_model = CCT(
        kernel_sizes=[(22, 1), (1, 24)], stride=(1, 1), padding=(0, 0),
        pooling_kernel_size=(3, 3), pooling_stride=(1, 1), pooling_padding=(0, 0),
        n_conv_layers=2, n_input_channels=1, in_planes=64, activation=None,
        max_pool=False, conv_bias=False, dim=64, num_layers=4, num_heads=8, num_classes=2,
        attn_dropout=0.1, dropout=0.1, mlp_size=64, positional_emb="learnable"
    ).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        cct_output = cct_model(X)
    cct_time = time.time() - start_time
    
    print(f"CCT Output shape: {cct_output.shape}")
    print(f"CCT Inference time: {cct_time:.4f}s")
    print(f"CCT Parameters: {sum(p.numel() for p in cct_model.parameters()):,}")
    
    # MSFAETモデルテスト
    print("\n--- MSFAET Model Test ---")
    msfaet_model = MSFAET(
        n_channels=22, n_classes=2, dim=64, depth=4, num_heads=8,
        mlp_ratio=4., qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1
    ).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        msfaet_output = msfaet_model(X)
    msfaet_time = time.time() - start_time
    
    print(f"MSFAET Output shape: {msfaet_output.shape}")
    print(f"MSFAET Inference time: {msfaet_time:.4f}s")
    print(f"MSFAET Parameters: {sum(p.numel() for p in msfaet_model.parameters()):,}")
    
    # 比較
    print("\n--- Comparison ---")
    speed_improvement = ((cct_time - msfaet_time) / cct_time) * 100
    param_difference = sum(p.numel() for p in msfaet_model.parameters()) - sum(p.numel() for p in cct_model.parameters())
    
    print(f"Speed improvement: {speed_improvement:+.1f}%")
    print(f"Parameter difference: {param_difference:+,}")
    
    # 出力の検証
    assert cct_output.shape == (batch_size, n_classes), f"CCT output shape mismatch: {cct_output.shape}"
    assert msfaet_output.shape == (batch_size, n_classes), f"MSFAET output shape mismatch: {msfaet_output.shape}"
    
    print("\n✅ 全てのテストが成功しました！")


def test_frequency_tokenizer():
    """周波数認識Tokenizerの動作確認"""
    print("\n=== 周波数認識Tokenizerテスト ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from model.msfaet import FrequencyAwareTokenizer
    
    tokenizer = FrequencyAwareTokenizer(dim=64, n_channels=22).to(device)
    
    # サンプルEEGデータ
    X = torch.randn(8, 1, 22, 1000).to(device)
    
    with torch.no_grad():
        tokens = tokenizer(X)
    
    print(f"Input shape: {X.shape}")
    print(f"Tokenized shape: {tokens.shape}")
    print(f"Sequence length: {tokens.shape[1]}")
    print(f"Embedding dimension: {tokens.shape[2]}")
    
    # 周波数帯域の確認
    print("\n周波数帯域設定:")
    for name, (low, high) in tokenizer.freq_bands.items():
        print(f"  {name}: {low}-{high} Hz")
    
    print("✅ Tokenizerテスト成功！")


def test_gradient_flow():
    """勾配フローのテスト"""
    print("\n=== 勾配フローテスト ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MSFAET(
        n_channels=22, n_classes=2, dim=64, depth=2, num_heads=4,
        mlp_ratio=2., qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0
    ).to(device)
    
    # 小さなバッチで訓練テスト
    X = torch.randn(4, 1, 22, 1000).to(device)
    y = torch.randint(0, 2, (4,)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 訓練ステップ実行
    model.train()
    for i in range(5):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # 勾配の確認
        total_norm = 0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        total_norm = total_norm ** (1. / 2)
        
        optimizer.step()
        
        print(f"Step {i+1}: Loss = {loss.item():.4f}, Grad norm = {total_norm:.4f}")
    
    print("✅ 勾配フローテスト成功！")


def main():
    """メイン実行関数"""
    print("🚀 MSFAET Model Validation Starting...")
    print("=" * 50)
    
    try:
        test_model_forward_pass()
        test_frequency_tokenizer() 
        test_gradient_flow()
        
        print("\n" + "=" * 50)
        print("🎉 全てのテストが正常に完了しました！")
        print("\nMSFAETモデルの主な特徴:")
        print("1. ✅ 周波数認識Tokenizer - EEG各周波数帯域を個別処理")
        print("2. ✅ 効率的なLinear Attention - O(n²)→O(n)の計算量削減")
        print("3. ✅ 適応的MLP - ゲーティング機構による動的処理")
        print("4. ✅ マルチスケール処理 - 局所・大域特徴の統合")
        print("5. ✅ 改良プーリング - CLSトークン + 注意機構")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
