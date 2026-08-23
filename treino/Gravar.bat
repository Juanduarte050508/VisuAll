@echo off
setlocal EnableExtensions
chcp 65001 >nul
title VisuAll - Gravar amostras de treino

REM ============================================================
REM  Abre a janela de gravacao: aparece a letra, ESPACO grava,
REM  3s de contagem + 3s gravando, salva sozinho em data\ e ja
REM  pula pra proxima letra.
REM
REM  Requisito: Python 3.10+ no PATH e as libs do requirements.txt
REM  (o proprio .bat instala se faltar).
REM ============================================================

cd /d "%~dp0.."

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERRO: Python nao encontrado no PATH.
    echo Instale em https://python.org/downloads/ marcando "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

python -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo Instalando dependencias pela primeira vez, aguarde...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERRO ao instalar as dependencias.
        pause
        exit /b 1
    )
)

python treino\gravar.py
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)

exit /b 0
