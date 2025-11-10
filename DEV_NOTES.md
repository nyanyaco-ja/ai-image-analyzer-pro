# 開発メモ (Development Notes)

## 開発環境の既知の問題

### Pylance警告: customtkinterのインポート解決エラー

**現象:**
```
インポート "customtkinter" を解決できませんでした
インポート "cv2" を解決できませんでした
```

**原因:**
- Windows環境の仮想環境でGUIアプリケーションを開発
- VSCodeがWSL側のPythonインタープリターを参照している
- WSL環境には`customtkinter`や`cv2`がインストールされていない

**影響:**
- コード実行には影響なし（Windows側で正常に動作）
- Pylanceが警告を表示するのみ

**実際のインストール状況（確認済み）:**
```bash
(venv) PS C:\Projects\image_compare> python -c "import cv2; print(cv2.__version__)"
4.12.0

(venv) PS C:\Projects\image_compare> python -c "import customtkinter; print(customtkinter.__version__)"
# 正常にインポート可能
```

- Windows仮想環境には**cv2 (OpenCV) 4.12.0**が正常にインストール済み
- customtkinterも正常にインストール済み
- バッチ処理でSSIM/PSNR計算が正常動作している証拠あり

**対処:**
- この警告は無視して問題なし
- 実際のアプリケーションはWindows側のPython仮想環境で実行されるため正常動作

**試した解決策:**
- `.vscode/settings.json`でWindows側のPythonパスを指定 → 大量のエラー発生のため撤回

**推奨:**
- VSCode上でPylance警告が出ていても、コード自体に問題はないため無視して開発を継続

---

## プロジェクト構成

### 仮想環境
- **場所**: Windows側 (例: `C:\Projects\image_compare\venv\`)
- **Python**: Windows版Python
- **主要パッケージ**: customtkinter, opencv-python, pillow, numpy, pandas, etc.

### 開発エディタ
- **VSCode**: WSL環境から開いている
- **Git**: WSL側で操作
- **実行環境**: Windows側

---

## 更新履歴

- 2025-11-10: Pylance警告の記録を追加
