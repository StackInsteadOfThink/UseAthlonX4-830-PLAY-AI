@echo off
title MiniChat Beta 0.1 - 对话/训练
cd /d "%~dp0"
set "PY=C:\Users\PCDN\AppData\Local\Programs\Python\Python312\python.exe"
:menu
echo.
echo =============================================
echo    MiniChat Beta 0.1  (300 万参数 · 英文)
echo    [1] 聊天推理   (加载 minichat.npz)
echo    [2] 重新训练   (从零训练, 需数天)
echo    [3] 退出
echo =============================================
set /p sel=  请选择 1/2/3: 
if "%sel%"=="1" goto infer
if "%sel%"=="2" goto train
if "%sel%"=="3" goto end
echo   输入无效, 请重试
timeout /t 1 >nul
goto menu
:infer
"%PY%" chat.py
pause
goto end
:train
"%PY%" train.py
pause
goto end
:end