@echo off
setlocal EnableExtensions
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
title VisuAll - Voltar um modelo anterior

REM ============================================================
REM  Desfaz um treino: devolve o modelo que existia antes.
REM
REM  Sempre que um treino SUBSTITUI um modelo geral (TreinarCorpo,
REM  ou Treinar sem --reforcar), o anterior e guardado em
REM  treino\modelos_anteriores\. Aqui voce escolhe qual devolver.
REM
REM  (Reforcar.bat nao aparece aqui porque ele nao substitui nada
REM  -- acrescenta modelos por letra sem apagar o que existia.)
REM ============================================================

cd /d "%~dp0.."

python "%~dp0restaurar.py"

echo.
pause
exit /b 0
