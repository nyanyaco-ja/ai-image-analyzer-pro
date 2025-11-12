# モダンGUI起動スクリプト

Write-Host "=========================================="
Write-Host "   AI Image Analyzer Pro 起動中..."
Write-Host "=========================================="
Write-Host ""

# 仮想環境のチェック
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "仮想環境を有効化します..." -ForegroundColor Green
    & "venv\Scripts\Activate.ps1"
}
else {
    Write-Host "仮想環境が見つかりません" -ForegroundColor Yellow
    Write-Host "  setup.ps1 を実行してください" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "続行するには Enter キーを押してください"
    exit
}

Write-Host ""
Write-Host "モダンGUIを起動します..." -ForegroundColor Cyan
Write-Host ""

# モダンGUIアプリを起動
python modern_gui.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "エラーが発生しました" -ForegroundColor Red
    Read-Host "続行するには Enter キーを押してください"
}
