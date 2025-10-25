# AI Image Analyzer Pro v1.6 - Release Notes

**リリース日**: 2025年10月25日

## 🎉 主要な新機能

### 1. **CLIP Embeddings 意味的類似度評価** 🔬

OpenAIのCLIPモデルを統合し、画像の**意味的内容**を評価できるようになりました。

- **使用モデル**: `openai/clip-vit-base-patch32` (1.5億パラメータ)
- **評価内容**: 2つの画像が意味的に同じ内容を表しているか
- **GPU対応**: CUDA自動検出、高速処理

#### 従来メトリクスとの違い

| メトリクス | 評価対象 |
|----------|---------|
| SSIM/PSNR | ピクセル・構造の一致 |
| LPIPS | 知覚的（人間の見た目）の類似 |
| **CLIP (新)** | **意味的内容の一致** |

### 2. **CLIP + LPIPS 統合幻覚検出** 🚨

CLIP（意味）とLPIPS（知覚）を組み合わせた高精度な異常検出機能を実装。

#### 判定ロジック

```
CLIP < 0.70 & LPIPS > 0.3  → 🚨 幻覚の可能性極めて高い
CLIP > 0.85 & LPIPS < 0.2  → ✅ 幻覚なし・高品質
CLIP < 0.70                → ⚠️  意味的に異なる
LPIPS > 0.5                → ⚠️  知覚的に大きく異なる
```

## 📊 評価指標の拡張

### v1.5 → v1.6

- **17項目** → **18項目** に拡張
- 新規追加: **CLIP Similarity（意味的類似度）**

#### 評価基準

| CLIP類似度 | 評価 | 意味 |
|-----------|------|------|
| > 0.95 | ✅ ほぼ同一 | 意味的に完全一致 |
| 0.85-0.95 | ✅ 非常に類似 | 同じ内容の画像 |
| 0.70-0.85 | ⚠️ 類似 | 一部異なる可能性 |
| 0.50-0.70 | ⚠️ やや類似 | 内容が異なる可能性 |
| < 0.50 | 🚨 全く異なる | 幻覚の可能性 |

## 🛠️ 技術的改善

### 実装ファイル

1. **`advanced_image_analyzer.py`**
   - `calculate_clip_similarity()` 関数追加 (L188-247)
   - メイン分析関数にCLIP統合 (L1142-1169)

2. **`result_interpreter.py`**
   - CLIP評価ロジック追加 (L159-180)
   - CLIP + LPIPS統合判定 (L569-589)

3. **依存関係**
   - `transformers>=4.30.0` を requirements.txt に追加

### GPU最適化

- **初回実行**: CLIPモデルダウンロード（約350MB、1-2分）
- **2回目以降**: キャッシュから読み込み（数秒）
- **メモリ使用**: 約1.5GB VRAM

## 📝 ドキュメント

### 新規追加

1. **`docs/clip_feature_guide.md`**
   - CLIP機能の詳細ガイド
   - 使用方法、評価基準、トラブルシューティング

2. **`test_clip.py`**
   - CLIP機能の動作確認スクリプト
   - インストール後の検証に使用

### 更新

- **`README.md`**: 18項目に更新、CLIP機能を追加
- **`docs/references_similarity_thresholds.md`**: 既存の論文ベース閾値

## 🚀 使用方法

### インストール

```bash
# 仮想環境をアクティベート
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 依存関係をインストール
pip install -r requirements.txt
```

### GUIから使用

```bash
# GUIを起動
python modern_gui.py

# または
起動.bat  # Windows
```

1. **元画像比較モード**を選択
2. 元画像、画像1、画像2を選択
3. 「分析開始」をクリック

### テスト実行

```bash
# CLIP機能のテスト
python test_clip.py
```

## 📈 実験結果への影響

### v1.5データセット（300枚）への適用

今後、300枚データセットにCLIP分析を追加実行予定：

- **CLIP類似度の分布**
- **CLIP × SSIM × LPIPS 3次元分析**
- **幻覚検出精度の向上**

### v2.0データセット（15,000枚）

15,000枚分析時にCLIPを標準搭載：

- カテゴリ別CLIP統計
- 深層学習幻覚検出器の特徴量としてCLIP Embeddingsを活用

## ⚠️ 既知の制限事項

1. **初回実行が遅い**
   - CLIPモデルのダウンロード（約350MB）
   - キャッシュ後は高速化

2. **VRAM使用量**
   - 約1.5GB必要
   - 他のGPUプロセスと競合する可能性

3. **transformers必須**
   - `pip install transformers` が必要
   - インストールしない場合はCLIP計算をスキップ

## 🔄 後方互換性

- **完全互換**: v1.5以前のデータ・スクリプトがそのまま動作
- **オプション機能**: CLIPがなくても従来機能は利用可能
- **JSON出力**: `clip_similarity` フィールドが追加（Noneの場合あり）

## 📚 参考文献

### CLIP関連

1. **Learning Transferable Visual Models From Natural Language Supervision**
   - Radford, A., et al. (2021)
   - ICML 2021
   - https://arxiv.org/abs/2103.00020

2. **Hugging Face - OpenAI CLIP**
   - https://huggingface.co/openai/clip-vit-base-patch32

### 本プロジェクト

- **Zenodo DOI**: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17282715.svg)](https://doi.org/10.5281/zenodo.17282715)
- **300枚データセット**: `data/batch_analysis_300images.csv`

## 🙏 謝辞

- **OpenAI**: CLIP モデルの公開
- **Hugging Face**: transformersライブラリ
- **NIH**: ChestX-ray14データセット

## 📞 サポート

- **GitHub Issues**: https://github.com/yourusername/image_compare/issues
- **ドキュメント**: `docs/clip_feature_guide.md`
- **テストスクリプト**: `test_clip.py`

---

## 次期バージョン予定

### v2.0 (2026年2月予定)

- **15,000枚データセット**完全分析
- **深層学習幻覚検出器**（ResNet18ベースライン）
- **CLIP Embeddings**を特徴量として活用
- **カラー画像**クロスドメイン検証（女忍者実験）
- **arXiv論文**投稿 + Zenodo DOI取得

---

**AI Image Analyzer Pro v1.6**
2025年10月25日リリース
