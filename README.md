# AI Image Analyzer Pro

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17282715.svg)](https://doi.org/10.5281/zenodo.17282715)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

> Professional AI super-resolution quality evaluation tool with 17 metrics. Batch processing, hallucination detection, and medical image analysis support. GPU-accelerated with detailed statistical reports.

**日本語:** AI超解像処理された医療画像を**17項目の指標**で詳細比較・評価するプロフェッショナルツールです。GPU対応でLPIPS（深層学習ベース）を含む高度な画質評価を実行できます。

![アプリロゴ](images/maou.jpg)

---

## ⚠️ Important Disclaimer

**AI Image Analyzer Pro** is an academic prototype proposal for AI safety and evaluation methodologies. It is **NOT intended for clinical diagnosis, medical practice, or as product selection criteria** for companies or organizations.

The calculation logic, results, and application **may contain computational errors or bugs**. The developer makes **no warranties or guarantees** regarding accuracy or validity and assumes **no liability** for any consequences arising from its use.

---

## 📊 Data Attribution

This project uses the **NIH ChestX-ray14 Dataset (ChestX-ray8)** provided by the NIH Clinical Center. We sincerely thank the NIH Clinical Center for making this dataset publicly available.

**Required Citation:**

Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 3462-3471.

**Dataset Download:** https://nihcc.app.box.com/v/ChestXray-NIHCC

---

## ⚠️ 重要な免責事項（日本語）

本稿で公開する**『AI Image Analyzer Pro』は、AIの安全性と評価手法に関する学術的なプロトタイプの提案**を目的としており、臨床診断や医療行為、または企業・組織における製品の選定基準として利用することを想定していません。

本ツールの計算ロジック、結果、およびアプリには、**計算誤差やバグが含まれている可能性**があり、その正確性や妥当性について、開発者は一切の保証と責任を負いません。

---

## 🔬 100枚分析プロジェクト公開中

**NIH ChestX-ray14データセット**を使用した**Upscayl 3モデル × 100枚（計300データポイント）**の定量評価を実施しました。

**※ 本ツールはUpscayl以外のAI超解像ツール（waifu2x、Real-ESRGAN、ESRGAN等）にも対応しています。Upscaylは評価例として使用しました。**

📊 **分析結果データ:**
- `data/batch_analysis_100images.csv`: 全300データポイント（100枚 × 3モデル）の生データ
- `analysis_output/` フォルダ: 23種類の統計分析プロット + サマリCSV
- **Zenodo (DOI: 10.5281/zenodo.17282715)**: 永久保存版データアーカイブ
  - [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17282715.svg)](https://doi.org/10.5281/zenodo.17282715)

**主な発見（Upscaylの場合）:**
- ✅ model3 (High Fidelity): 最高PSNR 41.5dB、最高構造保持
- ⚠️ model1 (Standard): 9%のハルシネーション検出率
- 🔍 「ノイズ」と「微細構造」の区別には専門家評価が必須

---

## 🎯 応用可能な分野

**主要用途:** 医療画像（X線、CT、MRI等）のAI超解像品質評価・ハルシネーション検出

**他分野への応用:**
- 🛰️ **衛星画像・リモートセンシング** - 地形解析、災害監視での高解像度化品質評価等
- 🏭 **製造業・品質管理** - 自動外観検査、欠陥検出システムでのAI超解像処理の品質評価等
- 🏗️ **インフラ・エネルギー** - ドローン点検、非破壊検査画像での微細構造の正確性評価等
- 🔬 **顕微鏡画像** - 生物学・材料科学での微細構造可視化の品質評価等
- 📸 **古写真・歴史資料の復元** - 劣化した写真の超解像品質評価等
- 🎨 **アニメ・イラスト** - デジタルアート作品のアップスケール品質評価等
- 🌌 **天体写真** - 望遠鏡画像の解像度向上品質評価等

**※ 静止画像のみ対応（動画は非対応）**

---

## 特徴

### 📊 17項目の総合評価
- **構造・知覚**: SSIM、MS-SSIM、PSNR、LPIPS
- **鮮明度**: シャープネス、コントラスト、エントロピー
- **ノイズ**: ノイズレベル、エッジ密度、アーティファクト検出
- **色・テクスチャ**: ΔE、高周波比率、テクスチャ複雑度、ヒストグラム相関
- **局所品質**: 局所品質平均、LAB明度
- **総合**: 総合スコア（0-100点）

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
- **GPU（オプション）**: NVIDIA CUDA対応GPU（推奨）
- **推奨GPU**: RTX 4000シリーズ以上（16GB VRAM）

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

### v1.5（現在）
- ✅ ハルシネーション検出に2つの方法を実装（固定閾値 + 動的閾値）
- ✅ パターン1で両方の結果を統合して報告
- ✅ レポートに実際の閾値を明記（検証可能性の向上）
- ✅ 円グラフでサブパターン別検出数を可視化

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

## ライセンス

**CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International)**

- ✅ 非営利目的での使用・改変・再配布は自由
- ✅ クレジット表記必須（"mohumohu neco"の名前を残すこと）
- ⚠️ **営利目的での使用は事前許可が必要**
- 📧 商用利用のお問い合わせ: s.shiny.n.works@gmail.com

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
