<div align="center">

# 🤟 VisuAll

### Real-Time Brazilian Sign Language (Libras) Recognition

**An AI system that translates Libras — alphabet and body signs — into text, live from a webcam.**

[![Python](https://img.shields.io/badge/Python-3.10+-2F81F7?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-0097A7?style=flat-square&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras_LSTM-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-MLP-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/FIAP_Challenge-2026-2F81F7?style=flat-square)]()

<!-- 📸 COLOQUE AQUI UM GIF DE DEMO (10-15s mostrando o reconhecimento ao vivo)
<img src="docs/demo.gif" width="700" alt="VisuAll live demo"/>
-->

</div>

---

## 💡 The Problem

Over **2 million people in Brazil** are deaf or hard of hearing, and Libras is
their primary language — yet very few hearing Brazilians understand it. VisuAll
aims to lower that communication barrier with accessible, real-time sign
recognition that runs on a regular computer with a webcam. No special hardware,
no gloves, no sensors.

> Developed for the **FIAP Challenge 2026** in partnership with **J0VI**.

---

## 🧠 How It Works

VisuAll combines **two recognition engines** in one unified system:

| Engine | What it recognizes | Model | Input |
|---|---|---|---|
| ✋ **Alphabet** | Static letters (A, B, C…) | MLP (scikit-learn) | 21 hand landmarks |
| 👋 **Alphabet (dynamic)** | Letters with motion (H, J, X, Z…) | MLP over frame sequences | Landmark sequences |
| 🧍 **Body Signs** | Full words/signs ("olá", "obrigado"…) | LSTM (Keras) | MediaPipe Holistic (pose + hands + face) |

### Architecture

```
                    ┌──────────────────────────────┐
   Webcam ───────►  │   MediaPipe (Hands/Holistic) │
                    │   landmark extraction         │
                    └──────────────┬───────────────┘
                                   │ normalized landmarks
                    ┌──────────────▼───────────────┐
                    │        Routing layer          │
                    │  (alphabet mode / sign mode)  │
                    └──────┬───────────────┬───────┘
                           │               │
              ┌────────────▼───┐   ┌───────▼────────────┐
              │  MLP models    │   │  Keras LSTM        │
              │  static+dynamic│   │  sequence model    │
              └────────────┬───┘   └───────┬────────────┘
                           │               │
                    ┌──────▼───────────────▼───────┐
                    │      Token-list builder       │
                    │  letters/signs → phrases      │
                    └──────────────┬───────────────┘
                                   │ WebSocket
                    ┌──────────────▼───────────────┐
                    │     Web frontend (live UI)    │
                    └──────────────────────────────┘
```

### Key Technical Features

- **Adaptive facial calibration** — landmark normalization adapts to each
  user's position and distance from the camera, improving accuracy across
  different setups.
- **Token-list phrase architecture** — recognized letters and signs are
  emitted as tokens and assembled into phrases, instead of raw per-frame
  predictions.
- **Static + dynamic letter handling** — letters that require motion (H, J,
  X, Z) are handled by a separate sequence-aware model, a common gap in
  alphabet-only recognizers.
- **Unified backend** — alphabet and body-sign engines, originally three
  separate codebases, were integrated into a single backend serving one
  frontend over WebSocket.

---

## 🚧 Project Status

| Module | Status |
|---|---|
| Alphabet recognition (static MLP) | ✅ Published in this repo |
| Alphabet recognition (dynamic MLP) | ✅ Published in this repo |
| Body-sign recognition (Holistic + LSTM) | 🔜 Integration sprint complete — publishing soon |
| Unified backend + frontend | 🔜 Publishing soon |
| FIAP Challenge 2026 presentation | 🗓️ In preparation |

---

## 🗂️ Repository Structure

VisuAll ships **two Python backends implementing the same recognition
pipeline**, plus **one native mobile port**. If you're new here, this is
what lives where and why there's more than one backend folder:

| Path | What it is |
|---|---|
| `linear/backend/app.py` | **Reference implementation — read it, don't expect to run it.** A single Python file with the full pipeline (camera capture → MediaPipe Holistic → MLP/LSTM classification → WebSocket → frontend). This is the spec the Kotlin port is checked against: `SentenceTranslator.kt`, `FaceMarkerEngine.kt`, `LibrasFragment.kt` and `LibrasAnalyzer.kt` all name the function here that they mirror. It **cannot recognise anything as checked out** — see the note under Getting Started. |
| `linear/frontend/` | The web UI (`index.html`) that `linear/backend/app.py` serves over WebSocket. Same caveat as the backend. |
| `treinamento/` | **Everything about collecting data and training models.** Record clips from a webcam (`Capturar.bat`), then extract landmarks and train (`Treinar.bat`, or `abrir_treinamento.bat` for the full GUI); models are written straight into `mobile/app/src/main/assets/`. See `treinamento/README.md`. |
| `tests/` | Shared fixtures (`fixtures/landmark_contract.json`) pinning the landmark math that the Python training side and the Kotlin app side must both reproduce, plus the script that regenerates them. |
| `modular/` | **The same backend as `linear/backend/app.py`, split into 10 small modules** (`m01_visuall_config.py` … `m10_visuall_servidor.py`) instead of one file. Entry point is `app_backend_unificado.py`. Behaves identically and reuses `linear/frontend/` — easier to navigate if you're extending the backend. |
| `models/` | Pre-trained model files ready for inference. See `models/README.md` for what each file is. |
| `mobile/` | **Android port** (Kotlin) of the same recognition pipeline, running fully on-device (MediaPipe Tasks API + ONNX Runtime + TFLite — no backend/WebSocket needed). See `mobile/README.md` to build and run it, and [CHANGELOG.md](CHANGELOG.md) for why its calibration constants have the values they do. |
| `mobile/tools/` | Windows scripts to set up an Android emulator matching the mobile app's requirements (webcam-backed front camera, compatible CPU architecture). See `mobile/tools/README.md`. |
| `.github/workflows/` | CI: compiles and unit-tests `mobile/` on every push/PR, and separately installs the Python stack to run the training-pipeline tests and confirm MediaPipe still starts. |
| `VALIDACAO.md` | The fixed ~10 minute on-device test to run before and after any recognition change, so "it feels better" can be replaced by a number. |

**In short:** `linear/backend/app.py` and `modular/` are two shapes of
the same desktop/Python system — pick whichever is easier to read for
what you're doing. `mobile/` is a separate, independent port of that
same logic to Android.

---

## ⚙️ Getting Started

> Requires **Python 3.10+**, a webcam, and **Git LFS**.

> ### Install Git LFS before cloning
>
> Every model file in this repo (`*.task`, `*.onnx`, `*.tflite`, `*.pkl`,
> `*.npz` — see `mobile/.gitattributes`) is stored with
> [Git LFS](https://git-lfs.com). Clone without it and you get **132-byte text
> pointers with the right filenames** instead of models. Nothing errors: the
> Android app compiles, installs, opens the camera, and recognises nothing.
>
> ```bash
> git lfs install          # once per machine, before cloning
> git lfs pull             # if you already cloned without it
> ```
>
> To check a clone is healthy, look at a size — it should be megabytes, not
> bytes:
>
> ```bash
> git lfs status
> ls -l mobile/app/src/main/assets/hand_landmarker.task   # ~7.6 MB, not 132 B
> ```

```bash
# 1. Clone the repository
git clone https://github.com/Juanduarte050508/VisuAll.git
cd VisuAll

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the backend — either version works the same way:
python linear/backend/app.py            # single-file reference version
python modular/app_backend_unificado.py # same thing, split into modules
```

Then open `linear/frontend/index.html` in your browser and start signing. ✋

> ### ⚠️ The Python backends do not recognise anything as checked out
>
> They start, the camera opens, and nothing is ever recognised — every model
> load sits in its own `try/except` that prints a warning and continues with
> `None`. So the failure is quiet, and easy to mistake for "the recognition is
> bad". What is missing:
>
> | What `app.py` looks for, beside itself | Status |
> |---|---|
> | `modelo_mlp_estatico.pkl`, `letras_mlp_estatico.pkl` | Same model exists as `models/static_model.pkl` + `static_classes.pkl` (verified: `MLPClassifier`, 42 inputs, 21 classes) — different filename, so it isn't found. |
> | `modelo_mlp.pkl`, `letras_mlp.pkl` | Same as `models/dynamic_model.pkl` + `dynamic_classes.pkl` (verified: 420 inputs, 5 classes). |
> | `modelo_sinais.h5`, `classes_sinais.npy` | **Not in this repository at all.** The file header says they came "from the Articulacao folder", which was never committed. Body gestures cannot work here. |
>
> Copying and renaming the two pairs from `models/` should restore the
> alphabet — untested, and the pickles were saved with scikit-learn 1.6.1 while
> `requirements.txt` now resolves to 1.9.0, which warns about exactly this.
>
> **None of this affects the Android app or the training tools**, which are what
> is actively developed. `mobile/` loads its own models from
> `mobile/app/src/main/assets/`, and `treinamento/` writes them there.
> Use `treinamento/` to train and `mobile/` to run; read `linear/` and
> `modular/` to understand the algorithm.

Looking for the Android app instead? It doesn't need this backend at all —
everything runs on-device. See [`mobile/README.md`](mobile/README.md).

---

## 🗺️ Roadmap

- [x] Static alphabet recognition (MLP)
- [x] Dynamic letters (H, J, X, Z) via sequence model
- [x] Body-sign recognition with MediaPipe Holistic + LSTM
- [x] Unify three codebases into a single backend/frontend
- [x] Adaptive facial marker calibration
- [ ] Publish the fully integrated version in this repo
- [ ] Expand the body-sign vocabulary
- [ ] React frontend + Node.js/Express API refactor
- [ ] Mobile exploration (ONNX → TFLite)

---

## 🧰 Tech Stack

`Python` · `OpenCV` · `MediaPipe (Hands & Holistic)` · `scikit-learn (MLP)` ·
`TensorFlow / Keras (LSTM)` · `WebSocket` · `HTML/CSS/JS`

---

## 👥 Team

| Member | Role |
|---|---|
| **Juan Duarte** | Technical lead — models, integration, backend/frontend |
| **Victor** | Presentation & design |

*FIAP Challenge 2026 · Partner brand: J0VI*

---

## 🔬 Technical Notes

### Why MLP instead of CNN?

The classifier input isn't an image — it's already-normalized **landmarks**
(coordinates relative to the wrist). MediaPipe Hands solves the hard part
(detecting the hand and locating 21 points) with its own well-optimized CNN.

After that, the problem becomes tabular: given a coordinate vector, which
letter is it? In that scenario, MLP is:

- **Lighter**: 1.7MB (dynamic) + 566KB (static), versus the tens of MB
  typical of CNNs.
- **Faster to infer**: sub-5ms latency on CPU.
- **Accurate enough**: decision boundaries between Libras letters in
  landmark space are well separated.

A CNN would only make sense if we wanted to skip MediaPipe and train
end-to-end from raw frames. That would be more robust to occlusion, but
would cost 100× more in compute and data.

### Static vs. dynamic routing

Which model to use is decided by a simple heuristic:

```python
mov = std(x_wrist) + std(y_wrist) + std(x_index) + std(y_index)
if mov > 0.30:
    use dynamic model
else:
    use static model
```

Alternatives considered:
- **Separate binary classifier** (static vs. dynamic): adds latency and
  another model to maintain, for marginal gain.
- **Unified model with a fixed 10-frame window**: forces static letters to
  "wait" 10 frames before classifying — poor UX.
- **Always run both models in parallel, keep the most confident**: doubles
  inference cost.

The motion-based heuristic is essentially free (4 standard deviations over
5 values) and works well because the human gesture of a dynamic letter is
clearly distinguishable in magnitude from the stability of a static one.

Known limits: hand tremor can falsely trigger dynamic mode (mitigated by
the confidence threshold); slow dynamic gestures may not trigger it
(mitigated by keeping the motion threshold low, 0.30 in normalized
coordinates).

### Landmark normalization

```python
def normalize_landmarks(points):
    base_x, base_y = points[0]   # wrist
    norm = []
    for x, y in points:
        norm.append(x - base_x)
        norm.append(y - base_y)
    max_v = max(abs(v) for v in norm) or 1.0
    return [v / max_v for v in norm]
```

Two operations:
1. **Translation**: subtracts the wrist position → invariant to the hand's
   position on screen.
2. **Uniform scaling**: divides by the largest absolute value → invariant
   to hand size (distance from camera).

Normalization is **not done per-component** (x and y aren't scaled
separately), since that would distort the proportion between dimensions —
hand shape needs to be preserved for the classifier.

### Threading model

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  CaptureThread  │───>│  raw_frame (Lock) │<───│ ProcessThread   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ↓
                                              ┌──────────────────┐
                                              │ camera_data(Lock)│
                                              └──────────────────┘
                                                        ↑
                                                        │ read
                                              ┌──────────────────┐
                                              │  asyncio (main)  │ ──> WebSocket
                                              └──────────────────┘
```

- **CaptureThread**: reads the webcam as fast as possible, overwrites
  `raw_frame`. If processing is slower, old frames are dropped — desired
  behavior (latency over completeness).
- **ProcessThread**: watches `raw_frame`'s timestamp, only processes when
  it's new, writes to `camera_data`.
- **asyncio loop (main)**: reads `camera_data` every 50ms and sends it over
  WebSocket, independent of the processing FPS.

Locks are held briefly (reference copy or small dict copy), never across
I/O or heavy CPU work, which avoids contention.

### Temporal stability before adding to phrase

Instant prediction ≠ confirmed letter. The flow is:

```
instant prediction → count consecutive frames with the same prediction
                   → once threshold is hit (12 static / 2 dynamic) → becomes candidate
                   → respects cooldown since last addition (1s / 0.3s)
                   → adds to phrase
                   → blocks immediate repetition of the same letter for 1s
```

The numbers (12, 2, 1s, 0.3s) came from empirical testing. Dynamic letters
need fewer frames because the gesture itself lasts only ~300ms — waiting
12 frames isn't feasible.

### Model training

**Dataset**
- Static: ~500 samples per letter, captured via photos. ~10k samples total.
- Dynamic: ~400 samples per letter (each sample = a 10-frame window). ~2k
  samples total.

**Balancing**: both training scripts cap the number of samples per class
(`MAX_POR_CLASSE`) to avoid bias. Letters with more data are randomly
sampled down to the cap.

**Hyperparameters**

```python
MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
)
```

- `(256, 128)` empirically outperformed `(128,)` and `(512, 256, 128)`.
- `early_stopping` avoids overfitting on the small dynamic-mode datasets.
- 80/20 train/test split with `stratify` to preserve class proportions.

**Observed metrics**: ~98% accuracy (static test set), ~94% accuracy
(dynamic test set — lower due to less available data).

### Known limitations

1. **Single hand only** (`max_num_hands=1`). Letters like "H" in some
   regional variants use two hands.
2. **No support for dynamic words** (full signs), only the manual alphabet.
3. **Sensitive to poor lighting**: MediaPipe loses hand tracking in very
   dark environments.
4. **Low dataset diversity**: recorded by a small number of people. Risk
   of skin-tone and hand-size bias.
5. **~50ms WebSocket latency**: acceptable for interactive use, but not
   for strict real-time requirements.

---

## 📜 Origin Story

VisuAll grew out of my Mechatronics capstone project: a
[robotic hand controlled by computer vision](https://github.com/Juanduarte050508/Engineering-Portfolio).
The same MediaPipe landmark approach that moved servo motors now powers
real-time sign language recognition.

---

<div align="center">

**If this project interests you, leave a ⭐ — it helps a lot!**

[Report a bug](https://github.com/Juanduarte050508/VisuAll/issues) · [Juan Duarte](https://github.com/Juanduarte050508)

</div>
