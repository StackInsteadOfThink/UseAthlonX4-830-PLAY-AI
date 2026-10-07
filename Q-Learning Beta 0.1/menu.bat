@echo off
title Q-Learning 走迷宫启动器
cd /d "D:\Q-Learning Beta 0.1"
set "PY=C:\Users\PCDN\AppData\Local\Programs\Python\Python312\python.exe"

:menu
cls
echo ==============================================
echo   Q-Learning 走迷宫   (速龙 X4 830 训练)
echo ==============================================
echo   1. 生成迷宫 Q 文件
echo   2. 训练 AI 权重
echo   3. 推理走迷宫
echo   0. 退出
echo ==============================================
set /p choice= 请输入选项后回车: 

if "%choice%"=="1" goto gen
if "%choice%"=="2" goto train
if "%choice%"=="3" goto play
if "%choice%"=="0" exit /b
goto menu

:gen
cls
set /p n= 迷宫尺寸 (如 5 / 12 / 20): 
echo 正在生成迷宫...
"%PY%" maze_gen.py %n%
echo.
pause
goto menu

:train
cls
set /p n= 迷宫尺寸 (如 5 / 12 / 20): 
set /p ep= 训练轮数 (如 2000 / 30000 / 1000000): 
echo 正在训练，请耐心等待...
"%PY%" qlearning_train.py %n% %ep%
echo.
pause
goto menu

:play
cls
echo 当前文件夹里的迷宫 Q 文件:
dir /b *.q 2>nul
echo.
set /p qf= 输入迷宫文件名 (如 Q_20x20.q): 
set /p wf= 输入权重文件名 (如 Q_20x20.npz): 
echo 正在推理走迷宫...
"%PY%" qlearning_play.py "%qf%" "%wf%"
echo.
pause
goto menu