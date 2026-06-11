<!--
  README DO REPOSITÓRIO VisuAll
  Cole no README.md do repo github.com/Juanduarte050508/VisuAll

  ANTES DE COMMITAR, revise:
  1. A seção "Project Status" reflete que a versão integrada ainda não subiu.
     Quando subir o código unificado, apague essa seção ou marque tudo como ✅.
  2. Ajuste os comandos de instalação/execução para os nomes reais dos seus
     arquivos (ex: main.py, server.py — confira como se chamam no seu projeto).
  3. Adicione um GIF/print de demo na seção indicada (faz MUITA diferença).
-->

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

## ⚙️ Getting Started

> Requires **Python 3.10+** and a webcam.

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

# 4. Run
python main.py                # ← ajuste para o nome real do seu entrypoint
```

Then open the frontend in your browser and start signing. ✋

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
