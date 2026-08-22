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

REM ---------------------------------------------------------------------------
REM Escolha do interpretador.
REM
REM NAO da pra chamar "python" direto. E comum ter varias versoes instaladas e o
REM "python" do PATH ser a MAIS ANTIGA -- foi exatamente o caso aqui: com o 3.11
REM instalado, "python --version" respondia 3.9.0, e o proprio lancador tinha o
REM 3.9 como padrao ("py -0" marcava "-3.9-64 *"). Depender do PATH significa
REM pedir pra cada pessoa da equipe reordenar variavel de ambiente na mao.
REM
REM O lancador "py" do Windows conhece todas as versoes registradas e aceita
REM escolher uma explicitamente. Testamos da mais nova pra mais velha e ficamos
REM na primeira que satisfaz o minimo; "py -3" e "python" ficam por ultimo, como
REM recurso pra quem instalou de um jeito que nao registra no lancador.
REM
REM O minimo e 3.10 porque o mediapipe e o tensorflow deste projeto INSTALAM em
REM 3.9 (existe wheel cp39) mas quebram no import com
REM "TypeError: unhashable type: 'list'" -- um erro que nao diz nada sobre a
REM causa. Sem esta checagem, quem estiver em 3.9 monta o ambiente inteiro
REM (varios minutos de download) e so descobre o problema no fim, sem pista.
set "PY="
for %%C in ("py -3.13" "py -3.12" "py -3.11" "py -3.10" "py -3" "python") do (
    if not defined PY (
        %%~C -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
        if not errorlevel 1 set "PY=%%~C"
    )
)

if not defined PY (
    echo [ERRO] Nao encontrei nenhum Python 3.10 ou mais novo neste PC.
    echo.
    echo   O motivo de exigir 3.10: as bibliotecas de visao ^(mediapipe^) e de
    echo   treino ^(tensorflow^) usadas aqui nao funcionam em 3.9 -- elas
    echo   instalam, mas quebram na hora de usar.
    echo.
    echo   O que fazer, num terminal:
    echo     winget install --id Python.Python.3.11 --exact --source winget
    echo.
    echo   Nao precisa desinstalar a versao antiga nem mexer no PATH: este
    echo   script acha a nova sozinho pelo lancador "py". Depois de instalar,
    echo   feche o terminal, abra outro e rode este arquivo de novo.
    exit /b 1
)

for /f "delims=" %%V in ('%PY% -c "import sys; print(sys.version.split()[0])"') do set "PYVER=%%V"
echo Usando Python %PYVER% ^(via %PY%^).

REM Um venv guarda pra sempre a versao de quem o criou -- ele nao se atualiza
REM sozinho quando um Python mais novo aparece no PC. Se sobrou um .venv feito
REM com versao velha, instalar dependencias dentro dele produziria o mesmo erro
REM sem causa aparente, entao paramos e pedimos pra apagar.
if exist "%VENV%\Scripts\python.exe" (
    "%VENV%\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo [ERRO] A pasta treinamento\.venv foi criada com um Python antigo.
        echo   Apague a pasta treinamento\.venv e rode este arquivo de novo. Ela
        echo   sera recriada com o Python %PYVER%, e nada mais e perdido: os
        echo   videos gravados ficam em treinamento\dados, nao dentro do .venv.
        exit /b 1
    )
)

if not exist "%VENV%\Scripts\python.exe" (
    echo Preparando o ambiente pela primeira vez, so vai demorar agora...
    %PY% -m venv "%VENV%"
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
