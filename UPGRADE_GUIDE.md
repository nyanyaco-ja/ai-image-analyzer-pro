# PyTorch v2.6.0 アップグレードガイド

## 🚨 なぜアップグレードが必要か

**CLIP機能を使うには PyTorch v2.6.0 以上が必須です。**

### エラー例
```
CLIP類似度計算エラー: Due to a serious vulnerability issue in `torch.load`,
even with `weights_only=True`, we now require users to upgrade torch to at
least v2.6 in order to use the function.
```

**理由**: PyTorchのセキュリティ脆弱性（CVE-2025-32434）対応のため

---

## 📋 アップグレード手順（3ステップ）

### ステップ1: バッチファイルを実行

```bash
upgrade_torch.bat
```

**または、手動でコマンド実行**:

```bash
# 仮想環境をアクティベート
venv\Scripts\activate

# 古いバージョンをアンインストール
pip uninstall -y torch torchvision torchaudio

# PyTorch 2.6.0+ をインストール（CUDA 12.1対応）
pip install torch>=2.6.0 torchvision>=0.20.0 --index-url https://download.pytorch.org/whl/cu121
```

### ステップ2: インストール確認

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**期待される出力**:
```
PyTorch: 2.6.0
CUDA: True
```

### ステップ3: CLIP機能テスト

```bash
python test_clip.py
```

**期待される出力**:
```
================================================================================
✅ すべてのテストが正常に完了しました！
================================================================================

CLIP機能は正常に動作しています。
GUI (modern_gui.py) から使用できます。
```

---

## 🔧 トラブルシューティング

### Q1: "Could not find a version that satisfies the requirement torch>=2.6.0"

**原因**: PyPIにまだリリースされていない可能性

**対処法**: CUDA版を直接インストール
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Q2: CUDA版がインストールできない

**GPU非搭載PCの場合**: CPU版をインストール
```bash
pip install torch>=2.6.0 torchvision>=0.20.0 --index-url https://download.pytorch.org/whl/cpu
```

### Q3: アップグレード後にエラーが出る

**依存関係の再インストール**:
```bash
pip install -r requirements.txt --force-reinstall
```

### Q4: VRAMエラーが出る

**CLIP + LPIPS の同時実行でVRAM不足**

対処法:
1. 他のGPUアプリを終了
2. 画像サイズを小さくする
3. バッチ処理の枚数を減らす

---

## 📊 アップグレード前後の比較

| 項目 | v2.0.0（旧） | v2.6.0+（新） |
|-----|------------|--------------|
| CLIP動作 | ❌ エラー | ✅ 正常動作 |
| セキュリティ | ⚠️ 脆弱性あり | ✅ 修正済み |
| VRAM使用 | 同じ | 同じ |
| 処理速度 | 同じ | やや向上 |

---

## 🎯 アップグレード後にできること

### 1. CLIP意味的類似度評価
```
元画像: 胸部X線
超解像画像: 全く違う画像

→ CLIP 0.45 🚨 幻覚検出！
```

### 2. CLIP + LPIPS統合幻覚検出
```
CLIP < 0.70 & LPIPS > 0.3
→ 🚨 幻覚の可能性が極めて高い
```

### 3. より正確な品質評価
```
18項目フル稼働:
✅ SSIM, PSNR, LPIPS
✅ CLIP ← NEW!
✅ シャープネス、コントラスト...
```

---

## 📞 サポート

問題が解決しない場合:
1. **test_clip.py** の出力をコピー
2. **GitHub Issues** で報告
3. エラーメッセージ全文を添付

---

**最終更新**: 2025年10月25日
**対象バージョン**: AI Image Analyzer Pro v1.6+
