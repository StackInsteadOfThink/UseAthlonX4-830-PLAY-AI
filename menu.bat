@echo off
title CIFAR 识图启动器
cd /d "%~dp0"
:menu
cls
echo.
echo   ============================
echo     CIFAR 识图 启动器
echo   ============================
echo     1. 识别一张图
echo     2. 训练模型 ^(约105分钟^)
echo     3. 下载数据 ^(只需一次^)
echo     0. 退出
echo   ============================
set /p c=  请选择 ^(0-3^):
if "%c%"=="1" goto infer
if "%c%"=="2" goto train
if "%c%"=="3" goto data
if "%c%"=="0" exit /b
goto menu
:infer
echo.
echo  请把图片文件拖到窗口后回车：
set /p img=  图片:
set "img=%img:"=%"
if "%img%"=="" goto menu
python cifar_predict.py "%img%"
echo.
pause
goto menu
:train
echo.
echo  开始训练，约105分钟，请耐心等待...
python cifar_cnn_v2.py 20
echo.
pause
goto menu
:data
echo.
echo  开始下载数据，约170MB...
python download_data.py
echo.
pause
goto menu