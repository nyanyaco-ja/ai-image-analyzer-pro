# AI Image Analyzer Pro

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17282715.svg)](https://doi.org/10.5281/zenodo.17282715)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

> **Universal AI Super-Resolution Quality Evaluation Tool** with 18 metrics (CLIP, LPIPS, SSIM, PSNR...) for cross-domain hallucination detection. Medical, Satellite, Microscopy, Anime, and more. GPU-accelerated with detailed statistical reports.

**日本語:** AI超解像処理された画像を**18項目の指標（CLIP統合）**でドメイン横断的に詳細比較・評価する汎用ツールです。医療画像、衛星画像、顕微鏡、アニメ等あらゆる分野に対応。GPU対応でLPIPS・CLIP（深層学習ベース）を含む高度な画質評価・幻覚検出を実行できます。

**Keywords**: AI Super-Resolution, Quality Assessment, CLIP, LPIPS, Hallucination Detection, Cross-Domain, Medical Imaging, Satellite Imagery, Microscopy, Computer Vision

![アプリロゴ](images/maou.jpg)

---

## ⚠️ Important Disclaimer

**AI Image Analyzer Pro** is an academic research tool for AI safety and quality evaluation methodologies. It is **NOT intended for clinical diagnosis, medical practice, critical infrastructure decision-making, or as final product selection criteria** for companies or organizations.

The calculation logic, results, and application **may contain computational errors or bugs**. The developer makes **no warranties or guarantees** regarding accuracy or validity and assumes **no liability** for any consequences arising from its use. **Always verify results with domain experts before making critical decisions.**

---

## ⚠️ 重要な免責事項（日本語）

本稿で公開する**『AI Image Analyzer Pro』は、AIの安全性と品質評価手法に関する学術研究ツール**を目的としており、臨床診断、医療行為、重要インフラの意思決定、または企業・組織における最終的な製品選定基準として利用することを想定していません。

本ツールの計算ロジック、結果、およびアプリには、**計算誤差やバグが含まれている可能性**があり、その正確性や妥当性について、開発者は一切の保証と責任を負いません。**重要な判断を行う前に、必ず専門家による検証を実施してください。**

---

## 📊 Data Attribution

This project uses the **NIH ChestX-ray14 Dataset (ChestX-ray8)** provided by the NIH Clinical Center. We sincerely thank the NIH Clinical Center for making this dataset publicly available.

**Required Citation:**

Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 3462-3471.

**Dataset Download:** https://nihcc.app.box.com/v/ChestXray-NIHCC

---

## 🔬 300枚分析プロジェクト公開中（v1.5.1対応）

**NIH ChestX-ray14データセット**を使用した**Upscayl 3モデル × 300枚（計900データポイント）**の定量評価を実施しました。

**※ 本ツールはUpscayl以外のAI超解像ツール（waifu2x、Real-ESRGAN、ESRGAN等）にも対応しています。Upscaylは評価例として使用しました。**

📊 **分析結果データ:**
- `data/batch_analysis_100images.csv`: 初回100枚データ
- **300枚統合データ**: 全900データポイント（300枚 × 3モデル）の包括的分析
- `analysis_output/` フォルダ: 23種類の統計分析プロット + サマリCSV
- **Zenodo (DOI: 10.5281/zenodo.17282715)**: 永久保存版データアーカイブ
  - [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17470655.svg)](https://doi.org/10.5281/zenodo.17470655)

**主な発見（Upscayl 300枚データ）:**
- ✅ **model1 (Standard)**: 36.7%安全率 - 最も信頼性が高い
- ⚠️ **model2 (Digital Art)**: 13.0%安全率 - 要注意（最も問題多い）
- 🎯 **model3 (High Fidelity)**: 32.7%安全率、最高PSNR 40.7dB
- 📊 **正常画像**: 168枚/300枚（56.0%）がハルシネーション検出0
- 🔍 **P6（品質ばらつき）**: 最重要検出パターン（225件検出）

---

## 🎯 Cross-Domain Applications (CLIP-Enhanced)

**CLIP統合により、あらゆる画像ドメインで高精度な幻覚検出が可能です。**

### 主要応用分野

#### 🏥 医療画像（Medical Imaging）
- **X線、CT、MRI等**のAI超解像品質評価
- 診断画像の内容一貫性評価、幻覚検出
- **実績**: NIH ChestX-ray14データセット300枚分析完了

#### 🛰️ 衛星画像・リモートセンシング（Satellite & Remote Sensing）
- 地形解析、土地利用変化モニタリング
- 災害前後比較での幻覚（存在しない建物等）検出
- 植生・水域の正確性評価

#### 🏭 製造業・品質管理（Manufacturing QA）
- PCB（プリント基板）自動外観検査
- 溶接部・欠陥検出システムの品質評価
- AI超解像による誤検出防止

#### 🔬 顕微鏡画像（Microscopy）
- 細胞形態の一貫性評価（生物学）
- 結晶構造・材料組織観察（材料科学）
- 組織病理画像の超解像品質評価

#### 📸 古写真・歴史資料の復元（Historical Photo Restoration）
- 文化財デジタルアーカイブの品質管理
- 劣化写真修復時の内容保持確認
- 建築物・人物の特徴保存評価

#### 🎨 アニメ・イラスト（Anime & Art）
- キャラクター一貫性の保持確認
- アートスタイルの維持評価
- デジタルアート作品のアップスケール品質管理

#### 🌌 天体写真（Astrophotography）
- 深宇宙探査画像の高解像度化
- 銀河・星雲構造の保持確認
- 望遠鏡画像処理の品質評価

#### 🏗️ インフラ点検（Infrastructure Inspection）
- ドローン点検画像の品質評価
- 橋梁・トンネルのひび割れ検出精度向上
- 非破壊検査（NDT）画像の正確性確認

### なぜCLIPで精度が向上するのか？

| 従来メトリクス | 評価内容 | 限界 |
|-------------|---------|------|
| SSIM/PSNR | ピクセル・構造の一致 | 内容変化を検出できない |
| LPIPS | 知覚的類似度 | 意味的変化に弱い |
| **CLIP (新)** | **意味的内容の一致** | **物体・構造変化を検出** ✅ |

**統合判定例**:
```
CLIP < 0.70 & LPIPS > 0.3 → 🚨 幻覚の可能性極めて高い
CLIP > 0.85 & LPIPS < 0.2 → ✅ 幻覚なし・高品質
```

**※ 静止画像のみ対応（動画は非対応）**

---

## 特徴

### 📊 18項目の総合評価
- **構造・知覚**: SSIM、MS-SSIM、PSNR、LPIPS、**CLIP Embeddings（意味的類似度）**
- **鮮明度**: シャープネス、コントラスト、エントロピー
- **ノイズ**: ノイズレベル、エッジ密度、アーティファクト検出
- **色・テクスチャ**: ΔE、高周波比率、テクスチャ複雑度、ヒストグラム相関
- **局所品質**: 局所品質平均、LAB明度
- **総合**: 総合スコア（0-100点）
- **🔬 AI幻覚検出**: CLIP + LPIPS統合による高精度異常検出

### 🎯 対応AI処理の種類

本ツールは様々なAI画像処理の品質評価・幻覚検出に対応しています：

#### 異解像度比較（超解像評価）
- **AI超解像**: 低解像度 → 高解像度への変換（例: 1000px → 2000px）
- **アップスケーリング**: 画像サイズ拡大の品質評価
- **高精細化**: 解像度向上処理の精度検証

#### 同解像度比較（画質改善評価）
- **ノイズ除去**: 2000px → 2000px（ノイズ削減処理の評価）
- **色調補正**: 2000px → 2000px（色補正の品質確認）
- **AI復元**: 2000px → 2000px（劣化画像の復元精度評価）
- **画質改善**: 2000px → 2000px（コントラスト・シャープネス改善の検証）

処理前（Before）と処理後（After）の画像を比較し、18項目の指標で詳細分析します。分析開始時に自動的に処理パターンを検出・表示します。

### 🚀 バッチ処理モード
- 大量画像の自動分析（100枚以上推奨）
- CSV出力で全データ記録
- 統計分析スクリプトで閾値自動決定
- ハルシネーション検出ロジック搭載

### 💻 GPU対応
- NVIDIA CUDA対応GPUで高速処理
- Kornia統合で主要計算をGPUアクセラレーション
- CPU版も完全サポート

### 🎨 モダンGUI
- CustomTkinterによる見やすいインターフェース
- リアルタイム進捗表示
- バッチ処理設定をGUIで変更可能

---

## 動作環境

- **OS**: Windows / Linux / macOS
- **Python**: 3.8以上
- **PyTorch**: **2.6.0以上**（CLIP機能使用に必須）
- **GPU（オプション）**: NVIDIA CUDA対応GPU（推奨）
- **推奨GPU**: RTX 4000シリーズ以上（6GB VRAM以上）

**⚠️ CLIP機能を使うには PyTorch 2.6.0+ が必須です**
- セキュリティ脆弱性（CVE-2025-32434）対応のため

---

## インストール

### 1. 仮想環境の作成（推奨）

```bash
# Windowsの場合
python -m venv venv
venv\Scripts\activate

# Linux/macOSの場合
python -m venv venv
source venv/bin/activate
```

### 2. 必要なライブラリのインストール

#### GPU版（推奨）
NVIDIA CUDA対応GPUをお持ちの場合：

```bash
# まずCUDA対応PyTorchをインストール
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 残りの依存関係をインストール
pip install -r requirements.txt
```

#### CPU版
GPUがない場合：

```bash
pip install -r requirements.txt
```

### 3. GPUの動作確認（GPU版の場合）

```bash
python test_gpu.py
```

正しく認識されていれば、以下のような出力が表示されます：
```
PyTorchバージョン: 2.5.1+cu121
CUDA利用可能: True
GPU名: NVIDIA GeForce RTX 4070 Ti SUPER
```

---

## 使い方

### 🖼️ 単一画像比較モード

#### GUIで起動（推奨）

**Windows:**
```bash
venv\Scripts\python.exe modern_gui.py
```

**Linux/Mac:**
```bash
./venv/bin/python modern_gui.py
```

**仮想環境を有効化している場合:**
```bash
python modern_gui.py
```

1. 「📁 画像1を選択」で比較したい画像1を選択
2. 「📁 画像2を選択」で比較したい画像2を選択
3. （オプション）「🎯 元画像」で低解像度の元画像を選択
4. 「🚀 分析開始」をクリック
5. 「📂 結果フォルダを開く」で結果確認

#### コマンドライン実行

```bash
python advanced_image_analyzer.py 画像1.png 画像2.png
```

元画像を指定する場合：
```bash
python advanced_image_analyzer.py 画像1.png 画像2.png --original 元画像.png
```

出力先ディレクトリを指定する場合：
```bash
python advanced_image_analyzer.py 画像1.png 画像2.png --output ./results
```

---

### 📊 バッチ処理モード（大量画像の自動分析）

**医療画像データセットでの統計的評価に最適**

#### Step 1: 設定ファイル作成

```bash
python batch_analyzer.py --create-config
```

`batch_config.json` が生成されます。

#### Step 2: 設定ファイル編集

```json
{
  "original_dir": "dataset/original/",
  "upscaled_dirs": {
    "model1": "dataset/upscayl_standard/",
    "model2": "dataset/upscayl_digital_art/",
    "model3": "dataset/upscayl_high_fidelity/"
  },
  "output_csv": "batch_analysis.csv",
  "output_detail_dir": "",
  "sample_size": 100
}
```

**推奨設定:**
- `output_detail_dir`: 空文字 `""` （詳細プロット無効で高速化）
- `sample_size`: 100以上（統計的信頼性のため）

#### Step 3: バッチ処理実行（GUIまたはCLI）

**GUIで実行（推奨）:**
```bash
python modern_gui.py
```
- 「バッチ処理」タブを選択
- 処理枚数を選択（10/50/100/全て）
- 「バッチ処理を開始」をクリック

**コマンドラインで実行:**
```bash
python batch_analyzer.py batch_config.json
```

**処理時間目安（RTX 4070 Ti SUPER）:**
- 100枚 × 3モデル（詳細プロット無効）: 約6-7分
- 100枚 × 3モデル（詳細プロット有効）: 約30分

#### Step 4: 統計分析で閾値決定

```bash
python analyze_results.py batch_analysis.csv
```

**出力結果（`analysis_output/` フォルダ）:**

| ファイル | 内容 |
|---------|------|
| `model_comparison.csv` | モデル別平均スコア比較 |
| `recommended_thresholds.json` | 推奨閾値（25/75パーセンタイル） |
| `results_with_risk_score.csv` | リスクスコア付きデータ |
| `hallucination_*.png` | ハルシネーション検出プロット（2種類） |
| `strategy_map_*.png` | 戦略マップ（5種類） |
| `pca_*.png` | 主成分分析（2種類） |
| `radar_chart_*.png` | レーダーチャート（3種類） |
| `violin_*.png` | バイオリンプロット（6種類） |
| `medical_*.png` | 医療画像特化プロット（2種類） |
| `tradeoff_*.png` | トレードオフ分析（2種類） |

**合計23種類**の統計分析プロットが自動生成されます。

---

## 出力ファイル

### 単一画像比較モードの出力

分析結果は `analysis_results/` ディレクトリに保存されます：

| ファイル名 | 内容 |
|-----------|------|
| `comparison_report.png` | ★総合レポート（グラフとスコア表示）★ |
| `detailed_analysis.png` | 詳細分析可視化（12枚の分析画像） |
| `difference.png` | 差分画像 |
| `heatmap.png` | 差分ヒートマップ |
| `comparison.png` | 3枚並べて比較 |
| `edges_img1.png` | 画像1のエッジ検出結果 |
| `edges_img2.png` | 画像2のエッジ検出結果 |
| `analysis_results.json` | 分析結果データ（JSON形式） |

### バッチ処理モードの出力

| ファイル名 | 内容 |
|-----------|------|
| `batch_analysis.csv` | 全画像の17項目スコア（1行1データポイント） |
| `analysis_output/` | 23種類の統計分析プロット |

---

## 評価項目の詳細

### 構造類似性・知覚品質（4指標）
1. **SSIM** (0-1、高いほど良い): 構造類似度
2. **MS-SSIM** (0-1、高いほど良い): マルチスケール構造類似度
3. **PSNR** (dB、高いほど良い): ピーク信号対雑音比
4. **LPIPS** (0-1、低いほど良い): 知覚的類似度（深層学習ベース）

### 鮮明度・コントラスト（3指標）
5. **Sharpness** (高いほど良い): エッジ鮮明度（Laplacian分散）
6. **Contrast** (高いほど良い): コントラスト（標準偏差）
7. **Entropy** (高いほど良い): エントロピー（情報量）

### ノイズ・アーティファクト（3指標）
8. **Noise Level** (低いほど良い): ノイズレベル（局所標準偏差）
9. **Edge Density** (%): エッジ密度
10. **Artifacts** (低いほど良い): アーティファクト検出（高周波異常）

### 色・テクスチャ（4指標）
11. **ΔE (Color Difference)** (低いほど良い): LAB色空間での色差
12. **High-Freq Ratio** (%): 高周波成分比率
13. **Texture Complexity**: テクスチャ複雑度
14. **Histogram Correlation** (0-1、高いほど良い): ヒストグラム相関

### 局所品質（2指標）
15. **Local Quality (Mean)** (0-1、高いほど良い): 局所品質平均
16. **LAB Brightness**: LAB色空間での明度

### 総合評価
17. **Total Score** (0-100、高いほど良い): 総合スコア

---

## ハルシネーション検出ロジック

以下の矛盾パターンを検出：

### パターン1: SSIM高 × PSNR低
**2つの検出方法を併用**：
- **固定閾値方式（絶対値基準）**: SSIM > 0.97 & PSNR < 25
  - メリット: 客観的で再現性が高い、論文発表向き
  - デメリット: データによっては検出できない場合がある
- **動的閾値方式（相対値基準）**: SSIM ≥ 75パーセンタイル & PSNR ≤ 25パーセンタイル
  - メリット: 少量データでも相対的な異常を検出可能
  - デメリット: データ分布に依存し、基準が変動する

**両方の結果を統合**して報告されます（重複除外）。

### パターン2: シャープネス高 × ノイズ高
- 閾値: シャープネス > 75パーセンタイル & ノイズ > 75パーセンタイル
- 意味: 鮮明なのにノイズが多い（過剰処理の疑い）

### パターン3: エッジ密度異常 × 低品質
- 閾値: エッジ密度 > 90パーセンタイル & 局所品質 < 25パーセンタイル
- 意味: エッジが過剰なのに品質が低い（不自然なエッジ追加の疑い）

**これらのパターンは、AIが「存在しない構造を追加」した可能性を示唆します。**

### GUI機能: ハルシネーション疑いデータ抽出

「⚠️ ハルシネーション疑いデータ抽出」ボタンで以下を出力：

1. **hallucination_suspects_[名前].csv** - 疑いデータ一覧
2. **hallucination_summary_[名前].csv** - モデル別統計
3. **hallucination_report_[名前].txt** - 詳細レポート（両方の閾値を明記）
4. **hallucination_analysis_[名前].png** - 6パネルのグラフ（パターン別内訳含む）

### GUI機能: クリーンデータセット抽出（v1.5新機能）

「✨ 正常データ抽出（AI学習用）」ボタンで**ハルシネーション検出0の正常データ**を自動抽出：

**出力フォルダ構成（`clean_dataset_YYYYMMDD_HHMMSS/`）:**
```
clean_dataset_20251010_123357/
├── original/         # 元画像（正常と判定された画像のみ）
├── model1_clean/     # model1で正常な超解像画像
├── model2_clean/     # model2で正常な超解像画像
├── model3_clean/     # model3で正常な超解像画像
├── metadata.csv      # 詳細メタデータ（各モデルのSSIM/PSNR/スコア等）
└── README.txt        # 使い方説明書
```

**用途:**
1. **AI学習データ** - 蒸留学習（Knowledge Distillation）、転移学習の教師データ
2. **品質フィルタリング済みデータセット** - ハルシネーションのないクリーンなデータのみ
3. **ベンチマークデータ** - 新モデル開発時の評価基準

**300枚データでの抽出結果例:**
- 正常画像: 168枚/300枚（56.0%）
- model1: 110枚（36.7%安全率）⭐ 最も信頼性が高い
- model2: 39枚（13.0%安全率）
- model3: 98枚（32.7%安全率）

**📦 データ公開:**
- ✅ **300枚統合CSV**: `data/batch_analysis_300images.csv`（本リポジトリに含む）
- 📊 **品質評価メタデータセット（30KB）**: Zenodoで公開中 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17470655.svg)](https://doi.org/10.5281/zenodo.17470655)
  - metadata.csv（247ペアの評価結果: SSIM、PSNR等17項目）
  - README.txt（NIH ChestX-ray14データセットとの組み合わせ方法）
  - AI学習・転移学習・ベンチマーク用途に最適
10.5281/zenodo.17470655
---

## トラブルシューティング

### GPUが使われない
```bash
python test_gpu.py
```
で確認してください。`CUDA利用可能: False`の場合は、CUDA版PyTorchを再インストール：
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### エラー: "No module named 'lpips'"
```bash
pip install lpips
```

### エラー: "No module named 'customtkinter'"
```bash
pip install customtkinter
```

### エラー: "No module named 'kornia'"
```bash
pip install kornia
```

### 日本語ファイル名で文字化け
このツールは日本語パスに対応していますが、一部の環境では問題が発生する場合があります。その場合は英数字のみのパスを使用してください。

---

## 更新履歴

### v1.6（現在）
- ✅ **多言語対応実装**（日本語/英語）
  - GUIで言語切り替え機能追加（設定タブ）
  - 全アコーディオン、ボタン、メッセージの完全翻訳対応
  - 評価モード、参照ボタン、警告メッセージの多言語化
  - locales/ja.json、en.jsonで翻訳管理
- ✅ **並列処理ON/OFF機能追加**
  - バッチ処理タブ・論文用タブで並列処理を選択可能
  - デフォルトOFF（少量データでの性能低下を防止）
  - ワーカー数の手動設定に対応
  - 大規模データセット（1000枚以上）で効果的
- ✅ **Bicubic縮小時のビット深度・カラー形式保持**
  - 元画像のビット深度を維持（8-bit → 8-bit、24-bit → 24-bit）
  - グレースケール・カラー形式の自動判定と保持
  - cv2.IMREAD_UNCHANGED使用で忠実な縮小処理
- ✅ **大規模データセット対応強化**
  - 15000サンプル対応の並列処理実装
  - チェックポイント機能（処理中断時の再開対応）
  - 論文用ベンチマーク評価の詳細ガイド追加
- ✅ **UI/UX改善**
  - スクロール量の最適化（3倍 → 1.75倍）
  - よくある間違いと正しいワークフローの説明追加

### v1.5
- ✅ **クリーンデータセット抽出機能追加**（AI学習用）
  - 正常データ（ハルシネーション検出0）を自動抽出
  - 元画像と超解像画像をペアでコピー、モデル別フォルダ分類
  - metadata.csv生成（蒸留学習・転移学習対応）
  - AI学習データ、品質フィルタリング済みデータセット、ベンチマークデータとして利用可能
- ✅ 17項目すべてを活用した包括的ハルシネーション検出（26パターン: 9組み合わせ + 17単独）
- ✅ 信頼度分類システム（高信頼度5+、中信頼度3-4、低信頼度1-2）
- ✅ 300枚データ対応・分析完了
- ✅ ハルシネーション検出に2つの方法を実装（固定閾値 + 動的閾値）
- ✅ パターン1で両方の結果を統合して報告
- ✅ レポートに実際の閾値を明記（検証可能性の向上）
- ✅ 円グラフで信頼度別分布を可視化

### v1.4
- ✅ バッチ処理に追加モード実装（既存CSVにデータ追加可能）
- ✅ 重複データ自動処理（同じimage_id + modelは新データで上書き）
- ✅ GUIに追加モード選択チェックボックス追加（デフォルト ON）
- ✅ 200枚分析対応
- ✅ ハルシネーション疑いデータ抽出機能追加（GUI）

### v1.3
- ✅ 100枚分析プロジェクト公開（Upscayl 3モデル評価）
- ✅ Kornia統合でGPU最適化強化
- ✅ バッチ処理GUIに進捗表示追加
- ✅ 23種類の統計分析プロット自動生成
- ✅ ハルシネーション検出ロジック実装
- ✅ Zenodo DOI取得・公開

### v1.2
- ✅ バッチ処理モード追加（大量画像の自動分析）
- ✅ CSV出力機能（17項目スコア記録）
- ✅ 統計分析スクリプト（閾値決定）
- ✅ 医療画像データセット対応

### v1.1
- ✅ GPU対応（CUDA）
- ✅ 元画像比較機能追加
- ✅ 17項目評価システム
- ✅ モダンGUI実装

### v1.0
- 初回リリース
- 15項目評価
- 基本的な画像比較機能

---

## 📜 License / ライセンス

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

**CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International)**

### ✅ Free Use (非営利利用は無償)

**Permitted users / 利用可能なユーザー:**
- 🎓 Academic institutions (universities, research labs) / 教育機関（大学、研究機関）
- 👤 Individual researchers (personal research projects) / 個人研究者（個人の研究プロジェクト）
- 🏫 Educational purposes (teaching, learning, coursework) / 教育目的（授業、学習、課題）

**Permitted activities / 許可される用途:**
- Research and academic publications / 学術研究・論文執筆
- Educational materials and teaching / 教育教材・授業での使用
- Personal projects and experimentation / 個人プロジェクト・実験
- Open-source contributions / オープンソースへの貢献

### ❌ Commercial Use Prohibited (企業利用は有償)

**Requires commercial license / 商用ライセンスが必要な場合:**
- 🏢 Corporate/Enterprise use (for-profit companies) / 営利企業での利用（企業内での使用）
- 🏭 Manufacturing QA systems (production line integration) / 製造業QAシステム（生産ラインへの組み込み）
- 💼 Commercial services (SaaS, consulting) / 商業サービス（SaaS、コンサルティング）
- 📊 Internal business operations (quality control, workflow automation) / 社内業務での利用（品質管理、ワークフロー自動化）
- 🤝 **Industry-academia collaborations by for-profit companies / 営利企業による産学連携研究**

### 🤝 Special Exemptions (特例)

The following are **allowed without commercial license** / 以下の場合は**商用ライセンス不要**:
- ✅ Non-profit organizations (NPO, NGO) / 非営利組織（NPO、NGO）
- ✅ Government research institutions / 政府研究機関
- ✅ Open-source project contributions / オープンソースプロジェクトへの貢献

### 🏢 Industry-Academia Collaboration Requirements / 産学連携研究の必須条件

**For industry-academia collaborations by for-profit companies:**

営利企業が本ツールの機能を利用した産学連携研究を希望する場合、無償での利用は認められません。利用の唯一の条件として、以下の事項を必須とします：

1. **Include the developer (mohumohu neco) as a co-researcher** / 本ツールの開発者（mohumohu neco）を必ず共同研究者として迎え入れること
2. **Pay appropriate compensation** (research funds, personnel costs, technical consulting fees, etc.) / 開発者に対し、**貢献に見合った適切な対価（研究費、人件費、技術指導料等）**を支払うこと

This is a legitimate requirement under CC BY-NC 4.0 license and international open-source standards (similar to MongoDB, Qt, Elasticsearch).

CC BY-NC 4.0ライセンスに基づく正当な権利として明記。無償での産学連携利用は認めず、開発者の貢献に対する適切な評価と対価支払いを必須条件とします。

### 📞 Commercial Licensing Contact / お問い合わせ

For commercial use and industry-academia collaboration inquiries:

企業利用・産学連携研究に関するお問い合わせ:
- **Email**: s.shiny.n.works@gmail.com
- **GitHub Issues**: [Create an issue with [COMMERCIAL LICENSE] tag / [COMMERCIAL LICENSE] タグで Issue作成]

詳細: [LICENSE](LICENSE) ファイルを参照

---

## 開発者

**mohumohu neco**

📧 Contact: s.shiny.n.works@gmail.com

---

## 参考情報

### 使用データセット

**NIH ChestX-ray14 Dataset (ChestX-ray8)**

本分析で使用した胸部X線画像データセットは、NIH Clinical Centerから提供されている公的なデータセット（ChestX-ray14/CXR8）に基づいています。データの利用にあたり、NIH Clinical Centerへ深く感謝の意を表します。

**データセットのダウンロード元:**
- https://nihcc.app.box.com/v/ChestXray-NIHCC

**必須引用（Required Citation）:**

Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 3462-3471.

**参考リンク:**
- NIH公式発表: https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

### 対応する超解像ツール（例）

本ツールは**どのAI超解像ツールでも評価可能**です。以下は代表的な例です：

**100枚分析で使用したツール:**
- **Upscayl** - https://upscayl.org/
  - GitHub: https://github.com/upscayl/upscayl
  - ライセンス: AGPL-3.0
  - GUI操作が簡単、複数モデル搭載

**その他の対応ツール:**
- **waifu2x** - イラスト・アニメ特化の超解像
- **Real-ESRGAN** - 実写画像の超解像（NVIDIA公式）
- **ESRGAN** - 汎用超解像モデル
- **その他** - 画像ファイルを出力できるツールなら何でも評価可能

### 評価手法の参考文献
- SSIM: Wang et al. (2004) "Image Quality Assessment: From Error Visibility to Structural Similarity"
- LPIPS: Zhang et al. (2018) "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
- MS-SSIM: Wang et al. (2003) "Multi-scale Structural Similarity for Image Quality Assessment"

---

**🎯 推奨ワークフロー:**

### 個別評価モード
1. AI超解像ツール（Upscayl、waifu2x、Real-ESRGANなど）で複数の設定を試す
2. 元画像と超解像画像2枚をこのツールで比較
3. `comparison_report.png`で総合評価を確認
4. 最も優れた設定を選択

### バッチ評価モード（100枚以上推奨）
1. 複数モデルで大量画像を超解像処理
2. バッチ処理で全データを自動分析
3. `analyze_results.py`で統計分析
4. `analysis_output/`の23種類のプロットで傾向把握
5. ハルシネーション検出結果を確認
6. 推奨閾値を参考にモデル選定
