# AI高解像度画像比較分析ツール v1.1

AI超解像処理された画像を17項目の指標で詳細比較・評価するツールです。GPU対応でLPIPS（深層学習ベース）を含む高度な画質評価を実行できます。

## 特徴

- **17項目の総合評価**: SSIM、MS-SSIM、PSNR、LPIPS、シャープネス、コントラスト、ノイズレベル、エッジ保持率、アーティファクト、色差、周波数分析、エントロピー、テクスチャ、局所品質、ヒストグラム、LAB明度、総合スコア
- **GPU対応**: NVIDIA CUDA対応GPUで高速処理（RTX 4050など）
- **元画像比較**: 低解像度の元画像を登録して、AI超解像の精度を評価
- **モダンGUI**: CustomTkinterによる見やすいインターフェース
- **詳細レポート**: グラフ・ヒートマップ・エッジ検出などの可視化画像を自動生成

## 動作環境

- **OS**: Windows / Linux / macOS
- **Python**: 3.8以上
- **GPU（オプション）**: NVIDIA CUDA対応GPU（推奨）

## インストール

### 1. 仮想環境の作成（推奨）

プロジェクトディレクトリで仮想環境を作成します：

```bash
# Windowsの場合
python -m venv venv
venv\Scripts\activate

# Linux/macOSの場合
python -m venv venv
source venv/bin/activate
```

仮想環境をアクティベートすると、プロンプトに `(venv)` が表示されます。

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
# requirements.txtから全てインストール（CPU版PyTorchが入ります）
pip install -r requirements.txt
```

**注意**: GPU版を使う場合は、先にCUDA対応PyTorchをインストールしてから、requirements.txtでその他のライブラリをインストールしてください。requirements.txtにはCPU版PyTorchが指定されているため、GPU版を上書きしないように注意が必要です。

### 3. GPUの動作確認（GPU版の場合）

```bash
python test_gpu.py
```

正しく認識されていれば、以下のような出力が表示されます：
```
PyTorchバージョン: 2.5.1+cu121
CUDA利用可能: True
GPU名: NVIDIA GeForce RTX 4050 Laptop GPU
```

## 起動方法

### 仮想環境のアクティベート

毎回使用する前に、仮想環境をアクティベートしてください：

```bash
# Windowsの場合
venv\Scripts\activate

# Linux/macOSの場合
source venv/bin/activate
```

### GUIモードで起動（推奨）

```bash
python modern_gui.py
```

### コマンドラインから直接実行

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

## 使い方

### 1. GUIでの操作手順

1. **modern_gui.py** を起動
2. 「📁 画像1を選択」ボタンで比較したい画像1を選択
3. 「📁 画像2を選択」ボタンで比較したい画像2を選択
4. （オプション）「🎯 元画像（オプション）」で低解像度の元画像を選択
5. 「🚀 分析開始」ボタンをクリック
6. リアルタイムで分析の進行状況が表示されます
7. 完了後、「📂 結果フォルダを開く」で結果を確認できます

### 2. 元画像機能について

元画像（低解像度画像）を登録すると、以下の指標がより正確に評価されます：

- **SSIM**: 各画像が元画像とどれだけ構造的に似ているか
- **PSNR**: 各画像が元画像とどれだけ信号品質が近いか
- **色差（ΔE）**: 各画像が元画像の色をどれだけ正確に再現しているか

元画像がない場合は、画像1 vs 画像2 の類似度評価になります。

## 出力ファイル

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

## 評価項目の詳細

### 勝者判定がある項目（11項目）
1. **SSIM（構造類似性）**: 画像の構造的な類似度
2. **PSNR（信号対雑音比）**: 信号品質の近さ
3. **シャープネス**: エッジの鮮明さ
4. **コントラスト**: 明暗の差
5. **ノイズレベル**: ノイズの少なさ
6. **エッジ保持率**: 細部・輪郭の保持度
7. **アーティファクト**: 圧縮歪み・ブロックノイズの少なさ
8. **色差（ΔE）**: 色の正確さ
9. **高周波成分**: 細かい模様・テクスチャの量
10. **エントロピー**: 画像の情報量・複雑さ
11. **テクスチャ複雑度**: テクスチャの豊富さ

### 参考指標（同等判定）
12. **MS-SSIM**: マルチスケールでの構造類似度
13. **LPIPS**: AI深層学習ベースの知覚的類似度
14. **局所品質**: パッチ単位での品質均一性
15. **ヒストグラム相関**: 輝度分布の類似度
16. **LAB明度**: 知覚的な明るさ

### 総合評価
17. **総合スコア**: 7項目（シャープネス、コントラスト、エントロピー、ノイズ、エッジ、アーティファクト、テクスチャ）の総合評価

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

### 日本語ファイル名で文字化け
このツールは日本語パスに対応していますが、一部の環境では問題が発生する場合があります。その場合は英数字のみのパスを使用してください。

## バッチ処理モード（大量画像の自動分析）

### **医療画像データセットでの実験用**

300枚以上の画像を自動で分析して、統計的に根拠のある閾値を決定できます。

#### **Step 1: 設定ファイル作成**

```bash
python batch_analyzer.py --create-config
```

`batch_config.json` が生成されます。

#### **Step 2: 設定ファイル編集**

```json
{
  "original_dir": "dataset/original/",
  "upscaled_dirs": {
    "upscayl_model1": "dataset/upscayl_model1/",
    "upscayl_model2": "dataset/upscayl_model2/",
    "upscayl_model3": "dataset/upscayl_model3/"
  },
  "output_csv": "results/batch_analysis.csv",
  "output_detail_dir": "results/detailed/"
}
```

#### **Step 3: バッチ処理実行**

```bash
python batch_analyzer.py batch_config.json
```

300枚の画像が自動で分析され、17項目スコアがCSVに記録されます。

#### **Step 4: 統計分析で閾値決定**

```bash
python analyze_results.py results/batch_analysis.csv
```

**出力結果：**
- モデル別ランキング
- 17項目の相関マトリックス
- 推奨閾値（25/75パーセンタイル基準）
- ハルシネーション検出ロジック
- リスクスコア付きCSV

**分析結果から得られる情報：**
- 各指標の統計的に妥当な閾値
- モデル別の性能比較
- ハルシネーション発生パターン
- 医療画像用の品質基準

---

## 更新履歴

### v1.2（現在）
- ✅ バッチ処理モード追加（大量画像の自動分析）
- ✅ CSV出力機能（17項目スコア記録）
- ✅ 統計分析スクリプト（閾値決定）
- ✅ ハルシネーション検出ロジック提案
- ✅ 医療画像データセット対応

### v1.1
- ✅ GPU対応（CUDA）
- ✅ 元画像比較機能追加
- ✅ 17項目評価システム
- ✅ モダンGUI実装
- ✅ スクロール対応UI
- ✅ 勝者カウント修正（全17項目を正確にカウント）

### v1.0
- 初回リリース
- 15項目評価
- 基本的な画像比較機能

## ライセンス

MIT License

## 開発者

mohumohu neco

AI アシスタント: Claude Code

---

**🎯 推奨ワークフロー:**
1. AI超解像ツール（waifu2x、Real-ESRGANなど）で複数の設定を試す
2. 元画像と超解像画像2枚をこのツールで比較
3. comparison_report.pngで総合評価を確認
4. 最も優れた設定を選択
