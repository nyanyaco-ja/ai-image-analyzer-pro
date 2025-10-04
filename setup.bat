@echo off
chcp 65001 > nul
title 画像比較分析ツール - 初期セットアップ

echo ==========================================
echo    画像比較分析ツール
echo    初期セットアップ
echo ==========================================
echo.

REM Pythonのバージョンチェック
python --version > nul 2>&1
if errorlevel 1 (
    echo ✗ Pythonがインストールされていません
    echo   https://www.python.org/downloads/ からダウンロードしてください
    pause
    exit
)

echo ✓ Pythonが見つかりました
python --version
echo.

REM 仮想環境の作成
if not exist venv (
    echo 仮想環境を作成中...
    python -m venv venv
    echo ✓ 仮想環境を作成しました
) else (
    echo ✓ 仮想環境は既に存在します
)
echo.

REM 仮想環境をアクティベート
call venv\Scripts\activate.bat
echo ✓ 仮想環境を有効化しました
echo.

REM 必要なライブラリをインストール
echo 必要なライブラリをインストール中...
echo （少し時間がかかります）
echo.

pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ✗ インストール中にエラーが発生しました
    pause
    exit
)

echo.
echo ==========================================
echo    セットアップ完了！
echo ==========================================
echo.
echo 次回からは「起動.bat」をダブルクリックして起動できます
echo.
pause
