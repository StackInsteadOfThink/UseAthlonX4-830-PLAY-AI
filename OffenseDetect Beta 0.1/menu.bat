@echo off
title 脏话与歧视检测启动器
cd /d "D:\OffenseDetect Beta 0.1"
set "PY=C:\Users\PCDN\AppData\Local\Programs\Python\Python312\python.exe"

:menu
cls
echo ==============================================
echo   脏话 + 种族歧视检测  (速龙 X4 830 训练)
echo ==============================================
echo   1. 训练模型
echo   2. 检测 (输入文本)
echo   0. 退出
echo ==============================================
set /p choice= 请输入选项后回车: 

if "%choice%"=="1" goto train
if "%choice%"=="2" goto detect
if "%choice%"=="0" exit /b
goto menu

:train
cls
echo 正在训练 4 个检测模型，请稍候...
"%PY%" train.py
echo.
pause
goto menu

:detect
cls
echo 输入文本(中英文均可), 自动判定脏话 + 歧视对象, 输入 0 退出
echo.
"%PY%" predict.py
echo.
pause
goto menu