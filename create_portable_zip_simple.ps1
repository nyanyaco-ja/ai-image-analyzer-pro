# USB用ポータブルZIPファイル作成スクリプト（簡易版）
# 不要なファイルを除外してZIP作成

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "USB用ポータブルZIP作成中..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$sourcePath = "C:\Projects\image_compare"
$destinationZip = "C:\Projects\image_compare_portable.zip"

# 既存のZIPファイルを削除
if (Test-Path $destinationZip) {
    Remove-Item -Path $destinationZip -Force
    Write-Host "既存のZIPファイルを削除しました" -ForegroundColor Yellow
}

Write-Host "`n📦 ZIP圧縮中..." -ForegroundColor Yellow
Write-Host "除外フォルダ: .git, __pycache__, venv, results, .cache 等" -ForegroundColor Gray

# Compress-Archiveで直接圧縮（除外パターン指定）
Get-ChildItem -Path $sourcePath -Recurse |
    Where-Object {
        $_.FullName -notmatch '\\\.git\\' -and
        $_.FullName -notmatch '\\__pycache__\\' -and
        $_.FullName -notmatch '\\venv\\' -and
        $_.FullName -notmatch '\\env\\' -and
        $_.FullName -notmatch '\\results\\' -and
        $_.FullName -notmatch '\\analysis_results\\' -and
        $_.FullName -notmatch '\\output\\' -and
        $_.FullName -notmatch '\\.cache\\' -and
        $_.FullName -notmatch '\\huggingface\\' -and
        $_.FullName -notmatch '\\transformers_cache\\' -and
        $_.FullName -notmatch '\\.vscode\\' -and
        $_.FullName -notmatch '\\.idea\\' -and
        $_.Extension -ne '.log' -and
        $_.Extension -ne '.pyc' -and
        $_.Name -ne 'Thumbs.db' -and
        $_.Name -ne '.DS_Store' -and
        $_.Name -ne 'temp_batch_config.json' -and
        $_.Name -notlike 'session_notes_*'
    } |
    ForEach-Object {
        $relativePath = $_.FullName.Substring($sourcePath.Length + 1)

        # 画像ファイルの処理（analysis_output内は残す）
        if ($_.Extension -match '\.(png|jpg|jpeg|bmp|tiff|webp)$') {
            if ($relativePath -like 'analysis_output\*' -or
                $relativePath -eq 'icon.ico' -or
                $relativePath -eq 'images\maou.jpg') {
                # これらは残す
                $_
            }
            # それ以外の画像は除外（何も返さない）
        } else {
            # 画像以外のファイルは含める
            $_
        }
    } |
    Compress-Archive -DestinationPath $destinationZip -CompressionLevel Optimal

# ファイルサイズ確認
if (Test-Path $destinationZip) {
    $zipSize = (Get-Item $destinationZip).Length / 1MB

    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "✅ ZIP作成完了！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "📁 保存先: $destinationZip" -ForegroundColor Cyan
    Write-Host "💾 サイズ: $([math]::Round($zipSize, 2)) MB" -ForegroundColor Cyan
    Write-Host "`n📋 次のステップ:" -ForegroundColor Yellow
    Write-Host "  1. USBメモリにこのZIPファイルをコピー" -ForegroundColor White
    Write-Host "  2. 別PCでZIPを展開" -ForegroundColor White
    Write-Host "  3. pip install -r requirements.txt でライブラリインストール" -ForegroundColor White
    Write-Host "  4. python modern_gui.py でGUI起動" -ForegroundColor White
    Write-Host "========================================`n" -ForegroundColor Green
} else {
    Write-Host "`n❌ ZIP作成に失敗しました" -ForegroundColor Red
}
