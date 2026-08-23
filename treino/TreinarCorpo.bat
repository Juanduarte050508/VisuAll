@echo off
setlocal EnableExtensions
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
title VisuAll - Treinar gestos corporais

REM ============================================================
REM  Treina o modelo de GESTOS CORPORAIS (AJUDAR, COMPUTADOR,
REM  CONVERSAR, NEUTRO, PESSOA, SURDO) a partir dos clipes
REM  gravados no modo "corpo" do Gravar.bat.
REM
REM    data\raw_body_videos\<GESTO>\*.mp4
REM              |
REM      [treinar_corpo.py]  pose + 2 maos -> 225 numeros por quadro
REM              v
REM    mobile\app\src\main\assets\gestos\geral\model.tflite + labels.txt
REM
REM  ATENCAO: diferente das letras, os gestos NAO tem modelo
REM  individual -- existe so um modelo geral. Por isso e preciso
REM  ter clipes de TODOS os gestos (inclusive NEUTRO), senao o
REM  app perde os que ficaram de fora. O script barra isso
REM  sozinho e avisa quais faltam.
REM
REM  Este e o unico treino que precisa do TensorFlow (~600 MB).
REM  Na 1a vez ele instala, e demora.
REM ============================================================

cd /d "%~dp0.."
set "RAIZ=%CD%"

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERRO: Python nao encontrado no PATH.
    pause
    exit /b 1
)

python -c "import cv2, mediapipe, numpy, sklearn" >nul 2>&1
if errorlevel 1 (
    echo Instalando dependencias basicas...
    python -m pip install -r requirements.txt
    if errorlevel 1 ( pause & exit /b 1 )
)

python -c "import tensorflow" >nul 2>&1
if errorlevel 1 (
    echo.
    echo O treino de gestos corporais usa TensorFlow, que ainda nao esta
    echo instalado. Sao uns 600 MB -- pode demorar bastante na primeira vez.
    echo.
    python -m pip install tensorflow
    if errorlevel 1 (
        echo.
        echo ERRO ao instalar o TensorFlow.
        pause
        exit /b 1
    )
)

if not exist "%RAIZ%\data\raw_body_videos" (
    echo.
    echo ============================================================
    echo  Nenhum clipe de gesto corporal encontrado.
    echo.
    echo  Abra Gravar.bat, aperte TAB ate o modo dizer "corpo",
    echo  e grave os gestos. Lembre do NEUTRO tambem.
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo Lendo os clipes. Isto demora: cada quadro passa pelo detector de
echo corpo E pelo de maos. Alguns minutos e normal.
echo.

python "%RAIZ%\treino\treinar_corpo.py" %*
if errorlevel 1 (
    echo.
    echo Terminou com problema. Veja a mensagem acima.
    pause
    exit /b 1
)

echo.
pause
exit /b 0
