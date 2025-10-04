# 画像比較分析ツール セットアップスクリプト (PowerShell)

Write-Host "=========================================="
Write-Host "   画像比較分析ツール"
Write-Host "   初期セットアップ"
Write-Host "=========================================="
Write-Host ""

# Pythonのバージョンチェック
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Pythonが見つかりました" -ForegroundColor Green
    Write-Host "  $pythonVersion" -ForegroundColor Cyan
}
catch {
    Write-Host "Pythonがインストールされていません" -ForegroundColor Red
    Write-Host "  https://www.python.org/downloads/ からダウンロードしてください" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "続行するには Enter キーを押してください"
    exit
}

Write-Host ""

# 仮想環境の作成
if (-not (Test-Path "venv")) {
    Write-Host "仮想環境を作成中..." -ForegroundColor Cyan
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "仮想環境を作成しました" -ForegroundColor Green
    }
    else {
        Write-Host "仮想環境の作成に失敗しました" -ForegroundColor Red
        Read-Host "続行するには Enter キーを押してください"
        exit
    }
}
else {
    Write-Host "仮想環境は既に存在します" -ForegroundColor Green
}

Write-Host ""

# 仮想環境をアクティベート
Write-Host "仮想環境を有効化します..." -ForegroundColor Green
& "venv\Scripts\Activate.ps1"

Write-Host ""

# 必要なライブラリをインストール
Write-Host "必要なライブラリをインストール中..." -ForegroundColor Cyan
Write-Host "(少し時間がかかります)" -ForegroundColor Yellow
Write-Host ""

pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=========================================="
    Write-Host "   セットアップ完了！" -ForegroundColor Green
    Write-Host "=========================================="
    Write-Host ""
    Write-Host "次回からは「起動.ps1」を実行して起動できます" -ForegroundColor Cyan
    Write-Host ""
}
else {
    Write-Host ""
    Write-Host "インストール中にエラーが発生しました" -ForegroundColor Red
}

Read-Host "続行するには Enter キーを押してください"
