@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title VisuAll - Interface de treinamento

REM ============================================================
REM  VisuAll - abre a interface grafica do treinamento (importar
REM  pasta externa, extrair landmarks, treinar por categoria, ver
REM  status). Pra um atalho rapido "treinar tudo que ja gravei",
REM  use Treinar.bat.
REM
REM  Na 1a vez instala sozinho um Python isolado (venv) com as
REM  dependencias -- nao precisa mexer em terminal.
REM
REM  Unico requisito: Python 3.10+ instalado e no PATH.
REM ============================================================

cd /d "%~dp0"

call "%~dp0_ambiente.bat"
if errorlevel 1 goto erro

"%~dp0.venv\Scripts\python.exe" "%~dp0interface_treinamento.py"
if errorlevel 1 (
    echo.
    echo Nao foi possivel abrir a interface. Veja o erro acima.
    goto erro
)
exit /b 0

:erro
echo.
pause
exit /b 1
