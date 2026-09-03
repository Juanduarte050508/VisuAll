# VisuAll Computer

Python workspace for the desktop recognizer, model training and shared tests.

## Layout

| Path | Purpose |
|---|---|
| `linear/` | Single-file desktop backend plus the web frontend it serves. |
| `modular/` | Same desktop backend split into smaller modules. |
| `treino/` | Recording, importing, extraction and training tools. |
| `tests/` | Shared contract fixtures between Python training and Android Kotlin math. |
| `models/` | Python `.pkl` models used by the desktop backend. |
| `data/` | Local recordings and generated datasets. Ignored by Git. |

## Setup

Use Python 3.11 when possible; the full training stack is kept tested on
Python 3.10-3.12.

```bat
cd computer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run The Desktop Backend

```bat
python linear\backend\app.py
python modular\app_backend_unificado.py
```

Open `linear\frontend\index.html` in the browser while the backend is running.

## Train Models For The Android App

Double-click the scripts in `computer\treino\`, or run them from this folder:

```text
treino\Gravar.bat
treino\Reforcar.bat
treino\Treinar.bat
treino\TreinarCorpo.bat
treino\RestaurarModelo.bat
```

Training outputs are written into `..\mobile\app\src\main\assets\`. Rebuild the Android app after training.

## Checks

```bat
python -m compileall -q treino linear modular tests
python -m unittest discover -s treino/tests -v
python tests\gerar_fixtures_contrato.py
```

For the Android-side contract check, run the Gradle tests from `..\mobile`.
