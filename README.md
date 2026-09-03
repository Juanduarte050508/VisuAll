# VisuAll

Real-time Libras recognition split into two work areas:

| Path | What lives there |
|---|---|
| `mobile/` | Native Android app. It runs on-device with MediaPipe Tasks, ONNX Runtime and TFLite. |
| `computer/` | Desktop Python backend, web frontend, model training tools, datasets, model files and shared contract fixtures. |

Files intentionally kept at the repository root:

- `.github/`: CI workflows for Android and Python training.
- `.gitignore`: shared ignore rules.
- `CHANGELOG.md`: project history.
- `LICENSE`: license.
- `README.md`: this map.

## Mobile

Open `mobile/` in Android Studio, or build from the terminal:

```bat
cd mobile
.\gradlew.bat assembleDebug
```

The APK is generated at:

```text
mobile\app\build\outputs\apk\debug\app-debug.apk
```

More details: [mobile/README.md](mobile/README.md).

## Computer And Training

All Python commands now start from `computer/`:

Use Python 3.11 when possible; the full training stack is kept tested on
Python 3.10-3.12.

```bat
cd computer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Desktop backend options:

```bat
python linear\backend\app.py
python modular\app_backend_unificado.py
```

Training shortcuts live in `computer\treino\`:

```text
computer\treino\Gravar.bat
computer\treino\Reforcar.bat
computer\treino\Treinar.bat
computer\treino\TreinarCorpo.bat
computer\treino\RestaurarModelo.bat
```

Training outputs are written directly into `mobile\app\src\main\assets\`, so rebuild the Android app after training.

More details: [computer/README.md](computer/README.md) and [computer/treino/README.md](computer/treino/README.md).
