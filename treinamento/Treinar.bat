@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title VisuAll - Treinar modelos

REM ============================================================
REM  VisuAll - atalho rapido: extrai landmarks de tudo que foi
REM  gravado com Capturar.bat (ou importado manualmente) e treina
REM  os modelos, deixando os arquivos novos prontos em
REM  mobile/app/src/main/assets/.
REM
REM  Pra mais controle (treinar so uma categoria, importar uma
REM  pasta externa, ver o status), use abrir_treinamento.bat, que
REM  abre a interface grafica do mesmo motor (treinar_visuall.py).
REM
REM  Na 1a vez instala sozinho um Python isolado (venv) com as
REM  dependencias -- nao precisa mexer em terminal.
REM
REM  Unico requisito: Python 3.10+ instalado e no PATH.
REM ============================================================

cd /d "%~dp0"

call "%~dp0_ambiente.bat"
if errorlevel 1 goto erro

"%~dp0.venv\Scripts\python.exe" "%~dp0treinar_visuall.py" extrair --tipos todos
if errorlevel 1 goto erro
"%~dp0.venv\Scripts\python.exe" "%~dp0treinar_visuall.py" treinar --tipos todos
if errorlevel 1 goto erro

echo.
echo Terminado. Recompile o app Android (assembleDebug) pra usar os modelos novos.
pause
exit /b 0

:erro
echo.
pause
exit /b 1
