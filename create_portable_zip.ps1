# USB用ポータブルZIPファイル作成スクリプト
# 不要なファイル（.git、__pycache__、venv、結果フォルダ等）を除外

$sourcePath = "C:\Projects\image_compare"
$destinationZip = "C:\Projects\image_compare_portable.zip"
$tempDir = "C:\Projects\image_compare_temp"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "USB用ポータブルZIP作成中..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 一時ディレクトリ作成
if (Test-Path $tempDir) {
    Remove-Item -Path $tempDir -Recurse -Force
}
New-Item -ItemType Directory -Path $tempDir | Out-Null

# 除外パターン
$excludePatterns = @(
    ".git"
    "__pycache__"
    "venv"
    "env"
    "ENV"
    ".vscode"
    ".idea"
    "results"
    "analysis_results"
    "output"
    ".cache"
    "huggingface"
    "transformers_cache"
    "*.log"
    "*.pyc"
    "*.pyo"
    "*.pyd"
    ".DS_Store"
    "Thumbs.db"
    "desktop.ini"
    "temp_batch_config.json"
    "session_notes_*.md"
    "note_article_draft.md"
    "*.png"
    "*.jpg"
    "*.jpeg"
    "*.bmp"
    "*.tiff"
    "*.webp"
)

# 画像ファイルの例外（残すファイル）
$keepFiles = @(
    "icon.ico"
    "images\maou.jpg"
)

Write-Host "`n📂 ファイルをコピー中..." -ForegroundColor Yellow

# すべてのファイルを再帰的にコピー（除外パターンを適用）
Get-ChildItem -Path $sourcePath -Recurse | ForEach-Object {
    $relativePath = $_.FullName.Substring($sourcePath.Length + 1)

    # 除外パターンにマッチするかチェック
    $shouldExclude = $false
    foreach ($pattern in $excludePatterns) {
        if ($relativePath -like "*$pattern*") {
            $shouldExclude = $true
            break
        }
    }

    # 例外的に残すファイル
    foreach ($keepFile in $keepFiles) {
        if ($relativePath -eq $keepFile) {
            $shouldExclude = $false
            break
        }
    }

    # analysis_output内の画像は残す
    if ($relativePath -like "analysis_output\*" -and ($relativePath -like "*.png" -or $relativePath -like "*.jpg")) {
        $shouldExclude = $false
    }

    # 除外しない場合はコピー
    if (-not $shouldExclude) {
        $destPath = Join-Path -Path $tempDir -ChildPath $relativePath

        if ($_.PSIsContainer) {
            # ディレクトリの場合
            if (-not (Test-Path $destPath)) {
                New-Item -ItemType Directory -Path $destPath -Force | Out-Null
            }
        } else {
            # ファイルの場合
            $destDir = Split-Path -Parent $destPath
            if (-not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
            }
            Copy-Item -Path $_.FullName -Destination $destPath -Force
        }
    }
}

Write-Host "`n📦 ZIP圧縮中..." -ForegroundColor Yellow

# 既存のZIPファイルを削除
if (Test-Path $destinationZip) {
    Remove-Item -Path $destinationZip -Force
}

# ZIP作成
Compress-Archive -Path "$tempDir\*" -DestinationPath $destinationZip -CompressionLevel Optimal

# 一時ディレクトリ削除
Remove-Item -Path $tempDir -Recurse -Force

# ファイルサイズ確認
$zipSize = (Get-Item $destinationZip).Length / 1MB

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "✅ ZIP作成完了！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "📁 保存先: $destinationZip" -ForegroundColor Cyan
Write-Host "💾 サイズ: $([math]::Round($zipSize, 2)) MB" -ForegroundColor Cyan
Write-Host "`n📋 次のステップ:" -ForegroundColor Yellow
Write-Host "  1. USBメモリにこのZIPファイルをコピー" -ForegroundColor White
Write-Host "  2. 別PCでZIPを展開" -ForegroundColor White
Write-Host "  3. 必要なPythonライブラリをインストール: pip install -r requirements.txt" -ForegroundColor White
Write-Host "  4. python modern_gui.py でGUI起動" -ForegroundColor White
Write-Host "========================================`n" -ForegroundColor Green
