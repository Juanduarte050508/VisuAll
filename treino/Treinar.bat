@echo off
setlocal EnableExtensions
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
title VisuAll - Treinar modelos do app

REM ============================================================
REM  Pipeline completo: videos/fotos -> modelo .onnx do app.
REM
REM    data\raw_images\<L>\  --[extract_from_images.py]-->  dataset_static.npz
REM    data\raw_videos\<L>\  --[extract_from_videos.py]-->  dataset_dynamic.npz
REM                                       |
REM                          [exportar_onnx.py] treina e exporta
REM                                       v
REM    mobile\app\src\main\assets\letras_estaticas\geral\model.onnx + labels.txt
REM    mobile\app\src\main\assets\letras_dinamicas\geral\model.onnx + labels.txt
REM
REM  IMPORTANTE: as duas categorias sao independentes. Se voce
REM  gravou so fotos, o treino das fotos acontece do mesmo jeito
REM  -- uma categoria sem dados nunca cancela a outra.
REM ============================================================

cd /d "%~dp0.."
set "RAIZ=%CD%"

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERRO: Python nao encontrado no PATH.
    echo Instale em https://python.org/downloads/ marcando "Add python.exe to PATH".
    pause
    exit /b 1
)

python -c "import cv2, mediapipe, numpy, sklearn, skl2onnx, onnxruntime" >nul 2>&1
if errorlevel 1 (
    echo Instalando dependencias, aguarde alguns minutos...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERRO ao instalar as dependencias.
        pause
        exit /b 1
    )
)

REM Os .npz sao sempre refeitos a partir das fotos/videos, que sao a fonte
REM da verdade -- assim um dataset velho nunca e treinado por engano.
if exist "%RAIZ%\data\dataset_static.npz"  del /q "%RAIZ%\data\dataset_static.npz"
if exist "%RAIZ%\data\dataset_dynamic.npz" del /q "%RAIZ%\data\dataset_dynamic.npz"

REM ---------- 1. fotos (letras paradas) ----------
echo.
echo ============================================================
echo  [1/3] LETRAS PARADAS - lendo as fotos
echo ============================================================
if exist "%RAIZ%\data\raw_images" (
    python "%RAIZ%\linear\backend\data_extraction\extract_from_images.py"
) else (
    echo Pulando: voce ainda nao gravou nenhuma letra parada.
)

REM ---------- 2. videos (letras com movimento) ----------
echo.
echo ============================================================
echo  [2/3] LETRAS COM MOVIMENTO - lendo os videos
echo ============================================================
if exist "%RAIZ%\data\raw_videos" (
    python "%RAIZ%\linear\backend\data_extraction\extract_from_videos.py"
) else (
    echo Pulando: voce ainda nao gravou nenhuma letra com movimento.
)

REM ---------- 2b. exemplos de "Nada" (opcional, mas ajuda muito) ----------
if exist "%RAIZ%\dataaw_negativos" (
    echo.
    echo ============================================================
    echo  [2b] EXEMPLOS DE "NADA" - mao a mostra sem fazer letra
    echo ============================================================
    python "%RAIZ%	reino\extrair_negativos.py"
)

REM ---------- 3. treina e exporta o que existir ----------
set "TEM_ALGO="
if exist "%RAIZ%\data\dataset_static.npz"  set "TEM_ALGO=1"
if exist "%RAIZ%\data\dataset_dynamic.npz" set "TEM_ALGO=1"

if not defined TEM_ALGO (
    echo.
    echo ============================================================
    echo  Nao ha nada pra treinar ainda.
    echo.
    echo  Grave amostras primeiro com Gravar.bat, depois volte aqui.
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  [3/3] TREINANDO e gerando os modelos do app
echo ============================================================
python "%RAIZ%\treino\exportar_onnx.py" %*
if errorlevel 1 (
    echo.
    echo A exportacao terminou com problema. Veja a mensagem acima.
    pause
    exit /b 1
)

echo.
pause
exit /b 0
