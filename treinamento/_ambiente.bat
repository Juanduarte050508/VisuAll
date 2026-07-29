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

REM Confere a VERSAO, nao so a existencia. O mediapipe e o tensorflow que este
REM projeto usa instalam em Python 3.9 (existe wheel) mas quebram no import
REM com "TypeError: unhashable type: 'list'" -- um erro que nao diz nada sobre
REM a causa. Sem esta checagem, quem tiver 3.9 monta o ambiente inteiro (varios
REM minutos de download) e so descobre o problema no fim, sem pista do motivo.
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    REM Sem variavel: !VAR! dependeria de o chamador ter ligado a expansao
    REM atrasada, e o %VAR% seria expandido antes do for rodar.
    python -c "import sys; print('[ERRO] Seu Python e a versao ' + sys.version.split()[0] + ', e este projeto precisa de 3.10 ou mais novo.')"
    echo.
    echo   O motivo: as bibliotecas de visao ^(mediapipe^) e de treino
    echo   ^(tensorflow^) usadas aqui nao funcionam mais em 3.9 -- elas
    echo   instalam, mas quebram na hora de usar.
    echo.
    echo   O que fazer: instale o Python 3.11 em https://python.org/downloads/
    echo   e marque "Add python.exe to PATH" durante a instalacao. Nao precisa
    echo   desinstalar o 3.9. Depois apague a pasta treinamento\.venv, se ela
    echo   existir, e rode este arquivo de novo.
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
