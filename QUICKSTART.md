# クイックスタートガイド

[English version](QUICKSTART_EN.md)

## 🚀 起動方法

### GUI起動（推奨）

#### Windows（バッチファイル）
```bash
launch.bat
```
または
```bash
launch.ps1
```

#### Windows（直接実行）
```bash
venv\Scripts\python.exe modern_gui.py
```

#### Linux/Mac
```bash
./venv/bin/python modern_gui.py
```

#### 仮想環境を有効化している場合
```bash
python modern_gui.py
```

---

## 📝 基本的な使い方

### 単一画像比較

1. 「📁 画像1を選択」で比較したい画像1を選択
2. 「📁 画像2を選択」で比較したい画像2を選択
3. （オプション）「🎯 元画像」で低解像度の元画像を選択
4. 「🚀 分析開始」をクリック
5. 「📂 結果フォルダを開く」で結果確認

### バッチ処理

1. GUIの「バッチ処理」タブを開く
2. 入力フォルダと出力フォルダを選択
3. 処理枚数を選択（10/50/100/全て）
4. 「バッチ処理を開始」をクリック

---

## ⚙️ コマンドライン実行

### 単一画像分析
```bash
venv\Scripts\python.exe advanced_image_analyzer.py 画像1.png 画像2.png
```

### 元画像を指定する場合
```bash
venv\Scripts\python.exe advanced_image_analyzer.py 画像1.png 画像2.png --original 元画像.png
```

### バッチ処理
```bash
venv\Scripts\python.exe batch_analyzer.py batch_config.json
```

---

## 🔧 トラブルシューティング

### エラー: `ModuleNotFoundError`
仮想環境が有効化されていません。上記の起動方法を使用してください。

### エラー: `python: コマンドが見つかりません`
仮想環境のPythonを直接指定してください：
- Windows: `venv\Scripts\python.exe`
- Linux/Mac: `./venv/bin/python`

---

詳細なドキュメントは [README.md](README.md) を参照してください。
