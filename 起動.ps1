# 画像比較分析ツール起動スクリプト (PowerShell)

Write-Host "=========================================="
Write-Host "   画像比較分析ツール 起動中..."
Write-Host "=========================================="
Write-Host ""

# 仮想環境のチェック
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "仮想環境を有効化します..." -ForegroundColor Green
    & "venv\Scripts\Activate.ps1"
}
else {
    Write-Host "仮想環境が見つかりません" -ForegroundColor Yellow
    Write-Host "  初回起動の場合は setup.ps1 を実行してください" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "続行するには Enter キーを押してください"
    exit
}

Write-Host ""
Write-Host "GUIを起動します..." -ForegroundColor Green
Write-Host ""

# GUIアプリを起動
python image_analyzer_gui.py

# エラーチェック
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "エラーが発生しました" -ForegroundColor Red
    Read-Host "続行するには Enter キーを押してください"
}
