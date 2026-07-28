@echo off
REM Chamado via "call _ambiente.bat" pelos launchers Capturar.bat,
REM Treinar.bat e abrir_treinamento.bat -- nao roda nada sozinho. Cria (se
REM preciso) um venv Python isolado dentro de treinamento\.venv e instala as
REM dependencias do requirements.txt da raiz do repo, so na primeira vez.

set "VENV=%~dp0.venv"
set "MARCA=%VENV%\.deps_ok"
REM Console do Windows (cmd.exe) geralmente NAO usa UTF-8 por padrao, e os
REM scripts imprimem emoji/setas (as, ->). Sem isso o Python quebra com
REM UnicodeEncodeError ao dar print nessas linhas.
set "PYTHONUTF8=1"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERRO] Nao encontrei Python no PATH.
    echo Instale o Python 3.10+ em https://python.org/downloads/ e marque
    echo "Add python.exe to PATH" durante a instalacao. Depois rode este
    echo arquivo de novo.
    exit /b 1
)

if not exist "%VENV%\Scripts\python.exe" (
    echo Preparando o ambiente pela primeira vez, so vai demorar agora...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo [ERRO] Falha ao criar o ambiente virtual Python.
        exit /b 1
    )
)

if not exist "%MARCA%" (
    echo Instalando dependencias ^(so na 1a vez, pode demorar alguns minutos^)...
    "%VENV%\Scripts\python.exe" -m pip install --upgrade pip >nul
    "%VENV%\Scripts\python.exe" -m pip install -r "%~dp0..\requirements.txt"
    if errorlevel 1 (
        echo [ERRO] Falha ao instalar as dependencias. Veja o erro acima.
        exit /b 1
    )
    type nul > "%MARCA%"
)

exit /b 0
