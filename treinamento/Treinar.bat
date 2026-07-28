@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title VisuAll - Treinar modelos

REM ============================================================
REM  VisuAll - extrai landmarks do que foi gravado com
REM  Capturar.bat e treina os modelos, deixando os arquivos
REM  novos prontos em mobile/app/src/main/assets/.
REM
REM  Na 1a vez instala sozinho um Python isolado (venv) com as
REM  dependencias -- nao precisa mexer em terminal.
REM
REM  Unico requisito: Python 3.10+ instalado e no PATH.
REM ============================================================

cd /d "%~dp0"

call "%~dp0_ambiente.bat"
if errorlevel 1 goto erro

"%~dp0.venv\Scripts\python.exe" "%~dp0treinar.py"
exit /b 0

:erro
echo.
pause
exit /b 1
