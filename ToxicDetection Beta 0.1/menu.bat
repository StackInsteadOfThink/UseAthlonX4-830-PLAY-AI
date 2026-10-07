@echo off
title 毒舌/歧视检测启动器
cd /d "D:\ToxicDetection Beta 0.1"
set "PY=C:\Users\PCDN\AppData\Local\Programs\Python\Python312\python.exe"

:menu
cls
echo ==============================================
echo   毒舌/种族歧视检测 (速龙 X4 830 训练)
echo   判定: 正常 / 脏话 / 种族歧视
echo ==============================================
echo   1. 训练模型
echo   2. 检测一句话
echo   0. 退出
echo ==============================================
set /p choice= 请输入选项后回车: 

if "%choice%"=="1" goto train
if "%choice%"=="2" goto predict
if "%choice%"=="0" exit /b
goto menu

:train
cls
echo 正在训练中英文检测模型，请稍候...
"%PY%" train.py
echo.
pause
goto menu

:predict
cls
echo 输入一句话(中英文均可), 输入 0 退出
echo.
"%PY%" predict.py
echo.
pause
goto menu