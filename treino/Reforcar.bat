@echo off
setlocal EnableExtensions
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
title VisuAll - Aprimorar letras

REM ============================================================
REM  Aprimora letras ESPECIFICAS sem mexer no resto do alfabeto.
REM
REM  Use este quando o app JA funciona e voce so quer melhorar
REM  algumas letras. Ele gera os "modelos individuais" dessas
REM  letras -- que o app testa antes do modelo geral -- e nao
REM  encosta no modelo geral nem no labels.txt.
REM
REM  Se voce quer REFAZER o alfabeto inteiro do zero (gravou
REM  todas as 21 letras de novo), use Treinar.bat.
REM ============================================================

cd /d "%~dp0"

echo.
echo  ============================================================
echo   APRIMORAR LETRAS
echo  ============================================================
echo.
echo   Quais letras voce quer melhorar?
echo.
echo   Paradas .......... A B C D E F G I L M N O P Q R S T U V W Y
echo   Com movimento .... H J K X Z
echo.
echo   Escreva separando por virgula. Exemplo:  E,F,G
echo   (deixe vazio e tecle ENTER pra cancelar)
echo.

set "LETRAS="
set /p LETRAS="   Letras: "

if not defined LETRAS (
    echo.
    echo   Cancelado -- nada foi alterado.
    echo.
    pause
    exit /b 0
)

echo.
echo   Aprimorando: %LETRAS%
echo   O modelo geral NAO sera alterado.
echo.

call "%~dp0Treinar.bat" --reforcar "%LETRAS%"
exit /b %errorlevel%
