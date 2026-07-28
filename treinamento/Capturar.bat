@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title VisuAll - Capturar amostras de treino

REM ============================================================
REM  VisuAll - abre a ferramenta de captura de video/fotos pra
REM  treinar letras com movimento e gestos corporais (e letras
REM  paradas). Duplo-clique, escolha o rotulo, clique GRAVAR.
REM
REM  Na 1a vez instala sozinho um Python isolado (venv) com as
REM  dependencias -- nao precisa mexer em terminal.
REM
REM  Unico requisito: Python 3.10+ instalado e no PATH.
REM ============================================================

cd /d "%~dp0"

call "%~dp0_ambiente.bat"
if errorlevel 1 goto erro

"%~dp0.venv\Scripts\python.exe" "%~dp0capturar.py"
exit /b 0

:erro
echo.
pause
exit /b 1
