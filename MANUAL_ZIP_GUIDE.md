# 手動でUSB用ZIPファイルを作成する方法

PowerShellスクリプトがうまく動作しない場合、手動でZIPファイルを作成できます。

---

## 方法1: Windowsエクスプローラーで手動作成（推奨・最も簡単）

### Step 1: 必要なファイルをコピー

1. **新しいフォルダを作成**
   ```
   C:\Projects\image_compare_for_usb\
   ```

2. **以下のファイル・フォルダをコピー**

   #### ✅ コピーするファイル・フォルダ
   ```
   C:\Projects\image_compare\
   ├── *.py（すべてのPythonファイル）
   │   ├── modern_gui.py
   │   ├── advanced_image_analyzer.py
   │   ├── batch_analyzer.py
   │   ├── analyze_results.py
   │   ├── result_interpreter.py
   │   └── その他.py
   ├── docs\（フォルダごと）※論文投稿戦略.mdは除く
   ├── data\（フォルダごと）
   ├── images\（フォルダごと）
   ├── analysis_output\（既に分析済みの場合のみ）
   ├── requirements.txt
   ├── README.md
   ├── QUICKSTART.md
   ├── BATCH_PROCESSING_GUIDE.md
   ├── USB_SETUP_GUIDE.md
   ├── LICENSE
   └── icon.ico
   ```

   #### ❌ コピーしないフォルダ・ファイル
   ```
   .git\（Gitリポジトリ）
   __pycache__\（Pythonキャッシュ）
   venv\, env\, ENV\（仮想環境）
   results\（分析結果）
   analysis_results\（分析結果）
   output\（出力フォルダ）
   .cache\（キャッシュ）
   huggingface\, transformers_cache\（モデルキャッシュ）
   .vscode\, .idea\（IDE設定）
   *.log（ログファイル）
   *.png, *.jpg（画像ファイル、一部例外除く）
   docs\論文投稿戦略_n1000_deep_learning.md（非公開メモ）
   session_notes_*.md（セッション記録）
   temp_batch_config.json
   ```

### Step 2: ZIPファイルに圧縮

1. **フォルダを右クリック**
   ```
   C:\Projects\image_compare_for_usb\ → 右クリック
   ```

2. **「送る」→「圧縮（zip形式）フォルダー」を選択**

3. **ファイル名を変更**
   ```
   image_compare_for_usb.zip → image_compare_portable.zip
   ```

4. **完成！**
   ```
   サイズ: 約3-5MB
   ```

### Step 3: USBメモリにコピー

```
C:\Projects\image_compare_portable.zip → USBメモリ
```

---

## 方法2: コマンドラインで作成（PowerShellが使える場合）

### シンプル版スクリプトを使用

```powershell
cd C:\Projects\image_compare
.\create_portable_zip_simple.ps1
```

このスクリプトは構文エラーを修正した簡易版です。

---

## 方法3: 7-Zipを使用（最速・推奨）

### 7-Zipをインストール済みの場合

```powershell
cd C:\Projects\image_compare

7z a -tzip C:\Projects\image_compare_portable.zip * `
  -xr!.git `
  -xr!__pycache__ `
  -xr!venv `
  -xr!env `
  -xr!results `
  -xr!analysis_results `
  -xr!output `
  -xr!.cache `
  -xr!huggingface `
  -xr!transformers_cache `
  -xr!.vscode `
  -xr!.idea `
  -x!*.log `
  -x!*.pyc `
  -x!temp_batch_config.json `
  -x!"session_notes_*.md"
```

---

## 方法4: Git で管理されているファイルのみをエクスポート

Gitリポジトリとして管理している場合、追跡されているファイルのみをアーカイブできます：

```bash
cd /mnt/c/Projects/image_compare
git archive --format=zip --output=../image_compare_portable.zip HEAD
```

ただし、この方法では以下が含まれない可能性があります：
- Gitに追跡されていない新規ファイル
- .gitignoreで除外されているファイル

---

## ファイルサイズの目安

| 内容 | サイズ |
|------|--------|
| Pythonコード + ドキュメント | 約1-2MB |
| analysis_output（プロット画像含む） | 約2-3MB |
| **合計** | **約3-5MB** |

---

## 確認方法

ZIPファイルを作成したら、以下を確認してください：

### ✅ チェックリスト

1. **ZIPファイルを開いて確認**
   - `modern_gui.py`が含まれている
   - `requirements.txt`が含まれている
   - `docs\`フォルダが含まれている
   - `USB_SETUP_GUIDE.md`が含まれている

2. **除外されているか確認**
   - `.git\`フォルダが含まれていない
   - `__pycache__\`フォルダが含まれていない
   - `venv\`フォルダが含まれていない
   - `results\`フォルダが含まれていない

3. **ファイルサイズ確認**
   - 3-5MB程度であればOK
   - 100MB以上の場合、不要なフォルダが含まれている可能性

---

## トラブルシューティング

### PowerShellスクリプトが動かない

→ **方法1の手動コピー**を推奨します（最も確実）

### ZIPファイルが大きすぎる

以下のフォルダが含まれていないか確認：
- `venv\`, `env\`（仮想環境、数GB）
- `huggingface\`, `.cache\`（モデルキャッシュ、数GB）
- `results\`, `analysis_results\`（分析結果、数百MB-数GB）

### USBメモリに入らない

- FAT32形式のUSBの場合、4GB以上のファイルは保存できません
- NTFS形式にフォーマットするか、OneDrive/Google Drive等のクラウド経由で転送

---

## 別PCでのセットアップ

ZIPファイルを別PCに移したら、`USB_SETUP_GUIDE.md`を参照してください。
