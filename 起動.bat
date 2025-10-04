@echo off
chcp 65001 > nul
title 画像比較分析ツール

echo ==========================================
echo    画像比較分析ツール 起動中...
echo ==========================================
echo.

REM 仮想環境をアクティベート
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo ✓ 仮想環境を有効化しました
) else (
    echo ⚠ 仮想環境が見つかりません
    echo   初回起動の場合は setup.bat を実行してください
    pause
    exit
)

echo.
echo ✓ GUIを起動します...
echo.

REM GUIアプリを起動
python image_analyzer_gui.py

REM エラーチェック
if errorlevel 1 (
    echo.
    echo ✗ エラーが発生しました
    pause
)
