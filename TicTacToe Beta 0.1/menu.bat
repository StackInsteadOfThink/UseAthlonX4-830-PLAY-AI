@echo off
title 三子棋 AI 启动器
cd /d "D:\TicTacToe Beta 0.1"
set "PY=C:\Users\PCDN\AppData\Local\Programs\Python\Python312\python.exe"

:menu
cls
echo ==============================================
echo   三子棋 AI   (速龙 X4 830 训练)
echo ==============================================
echo   1. 训练 AI
echo   2. 和 AI 下棋
echo   0. 退出
echo ==============================================
set /p choice= 请输入选项后回车: 

if "%choice%"=="1" goto train
if "%choice%"=="2" goto play
if "%choice%"=="0" exit /b
goto menu

:train
cls
set /p ep= 训练局数 (默认 50000): 
if "%ep%"=="" set ep=50000
echo 正在训练，请耐心等待...
"%PY%" tictactoe_train.py %ep%
echo.
pause
goto menu

:play
cls
echo 你执 X(先手), 输入 1-9 落子, 0 退出
"%PY%" tictactoe_play.py
echo.
pause
goto menu