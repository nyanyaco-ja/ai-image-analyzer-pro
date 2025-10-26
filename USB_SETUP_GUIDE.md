# USB経由での別PC環境セットアップガイド

このガイドでは、本プロジェクトをUSBメモリ経由で別のPCに移行し、バッチ分析を実行する手順を説明します。

---

## 📦 Step 1: USB用ZIPファイル作成（元のPC）

### Windows PowerShell を使用する場合

1. **PowerShellを管理者権限で起動**

2. **ZIP作成スクリプトを実行**
   ```powershell
   cd C:\Projects\image_compare
   .\create_portable_zip.ps1
   ```

3. **作成されたZIPファイルを確認**
   - 保存先: `C:\Projects\image_compare_portable.zip`
   - サイズ: 約3-5MB（ライブラリ・画像・結果フォルダを除く）

### 手動でZIP作成する場合

以下のフォルダ・ファイルを**含める**：
- ✅ すべての`.py`ファイル
- ✅ `docs/`フォルダ（`論文投稿戦略_n1000_deep_learning.md`を除く）
- ✅ `data/`フォルダ
- ✅ `README.md`、`QUICKSTART.md`等のドキュメント
- ✅ `requirements.txt`
- ✅ `icon.ico`、`images/maou.jpg`

以下のフォルダ・ファイルを**除外**：
- ❌ `.git/`（Gitリポジトリ）
- ❌ `__pycache__/`（Pythonキャッシュ）
- ❌ `venv/`、`env/`（仮想環境）
- ❌ `results/`、`analysis_results/`、`output/`（分析結果）
- ❌ `.cache/`、`huggingface/`、`transformers_cache/`（モデルキャッシュ）
- ❌ `.vscode/`、`.idea/`（IDE設定）
- ❌ `*.log`（ログファイル）
- ❌ `*.png`、`*.jpg`（画像ファイル、一部例外除く）
- ❌ `docs/論文投稿戦略_n1000_deep_learning.md`（非公開メモ）
- ❌ `session_notes_*.md`（セッション記録）

---

## 💾 Step 2: USBメモリにコピー

1. **USBメモリを接続**
2. **ZIPファイルをコピー**
   ```
   image_compare_portable.zip → USBメモリ
   ```

---

## 🖥️ Step 3: 別PCでのセットアップ

### 3.1 ZIPファイルを展開

1. USBメモリから任意の場所にZIPを展開
   ```
   例: C:\Projects\image_compare\
   ```

### 3.2 Python環境の確認

**必要なバージョン**:
- Python 3.9 以上（推奨: 3.10 または 3.11）
- CUDA対応GPU（推奨: RTX 4050以上）
- CUDA Toolkit 12.1以上（PyTorch 2.5.0以降対応）

**確認コマンド**:
```bash
python --version
nvidia-smi  # GPU確認
```

### 3.3 仮想環境作成（推奨）

```bash
cd C:\Projects\image_compare

# 仮想環境作成
python -m venv venv

# 仮想環境有効化
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
venv\Scripts\activate.bat
```

### 3.4 必要なライブラリをインストール

```bash
# PyTorchを先にインストール（CUDA対応版）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# その他のライブラリをインストール
pip install -r requirements.txt
```

**インストール時間**: 約5-10分（回線速度による）

**容量**: 約3-4GB（PyTorch、CUDA等含む）

### 3.5 GPU動作確認

```python
python -c "import torch; print('CUDA利用可能:', torch.cuda.is_available()); print('GPU名:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**正常な出力例**:
```
CUDA利用可能: True
GPU名: NVIDIA GeForce RTX 4050 Laptop GPU
```

---

## 🚀 Step 4: バッチ分析実行

### 4.1 GUI起動

```bash
python modern_gui.py
```

### 4.2 バッチ分析の設定

1. **「バッチ処理」タブを開く**

2. **フォルダパスを設定**
   - 元画像フォルダ: 1000px画像（5000枚）
   - AIモデル1結果フォルダ: 2000px画像（5000枚）
   - AIモデル2結果フォルダ: 2000px画像（5000枚）
   - AIモデル3結果フォルダ: 2000px画像（5000枚）

3. **出力CSVパスを指定**
   ```
   例: C:\Results\batch_analysis.csv
   ```

4. **評価モードを選択**
   - 実用評価モード: 実際の運用シナリオ
   - 学術評価モード: 論文用（Bicubic GT使用時）

### 4.3 バッチ処理実行

1. **「バッチ処理実行」ボタンをクリック**

2. **処理時間の目安**
   - RTX 4050: 約6-8時間（15,000枚）
   - RTX 4090: 約3-4時間（15,000枚）

3. **完了後、CSVが生成される**
   ```
   C:\Results\batch_analysis.csv
   ```

### 4.4 統計分析・26パターン検出実行（必須！）

**⚠️ 深層学習のラベル生成に必須のステップ！**

1. **統計分析タブで、生成されたCSVパスを入力**

2. **「統計分析・プロット23種類生成」ボタンをクリック**

3. **処理時間**: 約5-10分

4. **出力ファイル**
   ```
   analysis_output/results_with_26pattern_detection.csv  ← 深層学習用！
   analysis_output/*.png（23種類のプロット）
   ```

5. **重要**: `results_with_26pattern_detection.csv`には以下が含まれる
   - `detection_count`: 検出パターン数（深層学習のラベル生成に使用）
   - `detected_patterns`: 検出されたパターン名
   - `confidence_level`: リスクレベル

---

## 📤 Step 5: 結果を元のPCに戻す

### 5.1 必要なファイルをUSBにコピー

```
analysis_output/results_with_26pattern_detection.csv → USB
analysis_output/*.png（23種類のプロット）→ USB
```

### 5.2 元のPCで結果を確認

```bash
# CSVを確認
# detection_count列があることを確認
head analysis_output/results_with_26pattern_detection.csv
```

---

## 🔬 Step 6: 深層学習の準備（元のPC）

結果CSVが揃ったら、深層学習のラベル生成へ進みます：

```bash
# ラベル生成スクリプト実行
python scripts/generate_labels.py \
  --csv analysis_output/results_with_26pattern_detection.csv \
  --output labels/train_labels.csv \
  --threshold 3  # detection_count ≥ 3 → hallucination
```

---

## ⚠️ トラブルシューティング

### GPU認識されない場合

```bash
# CUDAバージョン確認
nvidia-smi

# PyTorchのCUDA対応確認
python -c "import torch; print(torch.version.cuda)"

# 再インストール（CUDA 12.1版）
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### メモリ不足エラー

バッチサイズを調整（`batch_config.json`の`limit`パラメータ）:
```json
{
  "limit": 1000,  # 1000枚ずつ分割処理
  "append_mode": true  # 追加モードで複数回実行
}
```

### 処理が遅い場合

- GPU使用率確認: `nvidia-smi -l 1`
- GPU使用率が低い → CPU処理にフォールバックしている可能性
- ドライバ更新、CUDA再インストールを検討

---

## 📋 チェックリスト

バッチ分析実行前:
- [ ] Python 3.9以上がインストールされている
- [ ] CUDA対応GPUが認識されている
- [ ] 仮想環境を作成した
- [ ] `requirements.txt`のライブラリをすべてインストールした
- [ ] GPUが正常に動作している（`torch.cuda.is_available()` → True）
- [ ] 15,000枚の画像データが準備されている

バッチ分析実行後:
- [ ] `batch_analysis.csv`が生成された
- [ ] **統計分析・26パターン検出を実行した**（重要！）
- [ ] `results_with_26pattern_detection.csv`が生成された
- [ ] `detection_count`列が含まれている

---

## 📞 サポート

問題が発生した場合:
1. エラーメッセージを記録
2. GPU使用状況を確認（`nvidia-smi`）
3. Pythonバージョン、ライブラリバージョンを確認
4. ログファイル（`*.log`）を確認
