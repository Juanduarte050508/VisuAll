"""
Extrai os clipes de "Nada" (data/raw_negativos/NADA/) em dois datasets de
exemplos NEGATIVOS -- mao a mostra que nao esta fazendo letra nenhuma.

Pra que servem: os modelos individuais sao perguntas de sim/nao ("isto e um
E?"). Se os unicos exemplos de "nao" que eles virem forem OUTRAS LETRAS, eles
nunca aprendem como e uma mao que nao esta sinalizando nada -- e no app
respondem "sim!" pra qualquer gesto solto. Estes clipes sao justamente esses
exemplos que faltavam.

Entrada:  data/raw_negativos/NADA/*.mp4
Saida:    data/dataset_negativos_static.npz    X [N, 42]
          data/dataset_negativos_dynamic.npz   X [M, 420]

Usa exatamente a mesma normalizacao dos outros extractors (e do app).
"""
import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "data" / "raw_negativos"
SAIDA_ESTATICO = RAIZ / "data" / "dataset_negativos_static.npz"
SAIDA_DINAMICO = RAIZ / "data" / "dataset_negativos_dynamic.npz"

JANELA = 10      # igual ao extract_from_videos.py
PULO = 3
STRIDE_FRAME = 3  # 1 quadro a cada 3, pro dataset estatico nao ficar repetitivo


def normalize_landmarks(pontos):
    """Cópia fiel de extract_from_videos.py / LibrasMath.normalizeLandmarks."""
    base_x, base_y = pontos[0]
    norm = []
    for x, y in pontos:
        norm.append(x - base_x)
        norm.append(y - base_y)
    max_v = max(abs(v) for v in norm) or 1.0
    return [v / max_v for v in norm]


def main():
    if not PASTA.exists():
        print("⏭  data/raw_negativos nao existe -- nenhum clipe de 'Nada' gravado.")
        return 0

    videos = []
    for sub in sorted(PASTA.iterdir()):
        if sub.is_dir():
            videos += sorted(list(sub.glob("*.mp4")) + list(sub.glob("*.mov")))

    if not videos:
        print("⏭  Nenhum clipe em data/raw_negativos.")
        print("   Grave alguns no modo 'nada' do Gravar.bat (mao a toa, cocando a")
        print("   cabeca, gesticulando) -- eles deixam os modelos individuais")
        print("   MUITO menos propensos a reconhecer letra onde nao tem.")
        return 0

    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, model_complexity=0,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    estaticos, dinamicos = [], []
    print("Lendo %d clipe(s) de 'Nada'..." % len(videos))

    for caminho in videos:
        cap = cv2.VideoCapture(str(caminho))
        frames_lm = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (320, 240))
            resultado = hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            if resultado.multi_hand_landmarks:
                pontos = [[lm.x, lm.y]
                          for lm in resultado.multi_hand_landmarks[0].landmark]
                dados = normalize_landmarks(pontos)
                frames_lm.append(dados)
                if i % STRIDE_FRAME == 0:
                    estaticos.append(dados)
            else:
                frames_lm = []   # mesma regra do extract_from_videos
            i += 1
        cap.release()

        j = 0
        while j + JANELA <= len(frames_lm):
            dinamicos.append(np.array(frames_lm[j:j + JANELA]).flatten())
            j += PULO

    hands.close()

    if estaticos:
        np.savez(SAIDA_ESTATICO, X=np.array(estaticos, dtype=np.float32))
        print("✅ %d amostras negativas estaticas  -> %s"
              % (len(estaticos), SAIDA_ESTATICO.name))
    else:
        print("⚠  Nenhum quadro com mao visivel nos clipes de 'Nada'.")
        print("   Lembre: a mao precisa APARECER (so que sem fazer letra).")

    if dinamicos:
        np.savez(SAIDA_DINAMICO, X=np.array(dinamicos, dtype=np.float32))
        print("✅ %d amostras negativas dinamicas  -> %s"
              % (len(dinamicos), SAIDA_DINAMICO.name))

    return 0


if __name__ == "__main__":
    sys.exit(main())
