# CLIP Embeddings機能ガイド

## 概要

v1.6から、**CLIP（Contrastive Language-Image Pre-training）Embeddings**による意味的類似度評価機能が追加されました。

OpenAIのCLIPモデルを使用して、2つの画像が**意味的に同じ内容**を表しているかを判定します。

## CLIP とは？

**CLIP (Contrastive Language-Image Pre-training)** は、OpenAIが開発した大規模Vision-Languageモデルです。

### 特徴
- **4億ペアの画像-テキストデータ**で学習
- 画像の**意味的内容**を理解できる
- ピクセル単位の比較（SSIM、PSNR）とは異なる視点で評価

### 既存メトリクスとの違い

| メトリクス | 評価内容 | 例 |
|----------|---------|---|
| **SSIM** | 構造的類似度（ピクセル配置） | 同じ構図の画像 |
| **PSNR** | 信号対雑音比（ピクセル値） | ノイズの少なさ |
| **LPIPS** | 知覚的類似度（人間の見た目） | 色味、質感の近さ |
| **CLIP** | **意味的類似度（画像の内容）** | **同じ物体/シーンか** |

## 使用例

### ケース1: AI超解像で内容が変わっていないか確認

```
元画像: 肺のX線画像
超解像画像: 肺のX線画像（高解像度化）

→ CLIP類似度: 0.95（意味的にほぼ同一）
→ ✅ 正常: 内容は変わっていない
```

### ケース2: 幻覚（ハルシネーション）検出

```
元画像: 胸部X線
超解像画像: 全く異なる臓器の画像に変化

→ CLIP類似度: 0.45（意味的に異なる）
→ 🚨 異常: 幻覚の可能性
```

## 評価基準

### CLIP コサイン類似度（-1.0 〜 1.0）

| 範囲 | 評価 | 意味 |
|-----|------|------|
| **> 0.95** | ✅ **ほぼ同一** | 意味的に完全に一致 |
| **0.85 - 0.95** | ✅ **非常に類似** | 同じ内容の画像 |
| **0.70 - 0.85** | ⚠️ **類似** | 似た内容だが一部異なる |
| **0.50 - 0.70** | ⚠️ **やや類似** | 内容が一部異なる可能性 |
| **< 0.50** | 🚨 **全く異なる** | 異なる画像（幻覚の可能性） |

## CLIP + LPIPS 統合幻覚検出

CLIP（意味）× LPIPS（知覚）を組み合わせることで、より正確に幻覚を検出できます。

### 判定ロジック

```python
if CLIP < 0.70 and LPIPS > 0.3:
    → 🚨 幻覚の可能性が極めて高い
    （意味的にも知覚的にも異なる）

elif CLIP > 0.85 and LPIPS < 0.2:
    → ✅ 幻覚なし・高品質
    （意味的にも知覚的にも一致）

elif CLIP < 0.70:
    → ⚠️ 意味的に異なる可能性

elif LPIPS > 0.5:
    → ⚠️ 知覚的に大きく異なる
```

### 検出パターン例

| CLIP | LPIPS | 判定 | 説明 |
|------|-------|------|------|
| 0.95 | 0.10 | ✅ 正常 | 高品質な超解像 |
| 0.88 | 0.25 | ✅ 正常 | やや劣化あるが内容は一致 |
| 0.65 | 0.35 | 🚨 異常 | **幻覚の可能性大** |
| 0.45 | 0.60 | 🚨 異常 | **完全に異なる画像** |

## 使用方法

### 1. GUI（modern_gui.py）から使用

1. **元画像比較モード**を選択
2. 元画像、画像1、画像2を選択
3. 「分析開始」をクリック

結果表示に以下が追加されます：
- **CLIP Similarity（意味的類似度）**: 0.XXXX
- **CLIP + LPIPS統合判定**: ✅ 正常 / 🚨 異常

### 2. コマンドラインから使用

```python
from advanced_image_analyzer import analyze_images

analyze_images(
    img1_path="upscaled_image.png",
    img2_path="upscaled_image2.png",
    original_path="original.png"  # 元画像を指定
)
```

結果JSON:
```json
{
    "clip_similarity": 0.9234,
    "lpips": 0.1523,
    "summary": {
        "warnings": [
            "✅ 優良【統合判定】: CLIP & LPIPS両方で高品質確認 - 幻覚なし"
        ]
    }
}
```

## 技術詳細

### 使用モデル
- **モデル名**: `openai/clip-vit-base-patch32`
- **アーキテクチャ**: Vision Transformer (ViT-B/32)
- **パラメータ数**: 約1.5億
- **学習データ**: 4億画像-テキストペア

### 処理フロー

```
1. 画像をPIL Image形式に変換
   ↓
2. CLIPProcessor で前処理（224×224にリサイズ、正規化）
   ↓
3. CLIP Vision Encoder でEmbedding抽出（512次元ベクトル）
   ↓
4. L2正規化
   ↓
5. コサイン類似度計算（内積）
```

### GPU対応

- **CUDA対応GPU**: 自動的にGPUで高速計算
- **CPU**: GPUがない場合は自動的にCPUで処理
- **初回実行**: モデルダウンロードで1-2分かかる（約350MB）
- **2回目以降**: キャッシュから読み込み（数秒）

## インストール

```bash
# 仮想環境をアクティベート
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# transformers ライブラリをインストール
pip install transformers>=4.30.0

# または requirements.txt から一括インストール
pip install -r requirements.txt
```

## トラブルシューティング

### Q1: "transformers not found" エラー

```bash
pip install transformers
```

### Q2: CLIPモデルのダウンロードが遅い

初回のみHugging Faceから約350MBのモデルをダウンロードします。
キャッシュ先: `~/.cache/huggingface/`

### Q3: GPU使用時にメモリ不足

CLIPモデルは約1.5GB VRAMを使用します。
他のGPUプロセスを終了してください。

### Q4: "CLIP計算をスキップしました" と表示される

`transformers`ライブラリがインストールされていません。
requirements.txtを確認してください。

## 参考文献

1. **CLIP論文**:
   - Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*.
   - URL: https://arxiv.org/abs/2103.00020

2. **Hugging Face CLIP**:
   - https://huggingface.co/openai/clip-vit-base-patch32

3. **本プロジェクトでの応用**:
   - AI超解像画像の幻覚検出
   - 医療画像の内容一貫性評価

## 更新履歴

- **v1.6 (2025-10-25)**: CLIP Embeddings機能追加
  - OpenAI CLIP-ViT-Base-Patch32統合
  - CLIP + LPIPS統合幻覚検出実装
  - GPU対応、自動キャッシング

---

**AI Image Analyzer Pro v1.6**
CLIP機能により、意味的類似度評価が可能になりました。
