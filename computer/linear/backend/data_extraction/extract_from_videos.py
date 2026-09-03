"""
Extrai landmarks dos vídeos e salva como dataset para treinar a MLP.

Estrutura esperada:
  videos/
    H/  video1.mp4  video2.mp4 ...
    J/  ...
    K/  ...
    X/  ...
    Z/  ...

Gera: computer/data/dataset_dynamic.npz  (X com shape [N, 420], y com shape [N])
"""
import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

# Windows + console cp1252 quebra nos prints com "→"/"✅" abaixo
# (UnicodeEncodeError). Força UTF-8 na saída, sem alterar as mensagens.
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============ CONFIGURAÇÃO ============
# Caminhos ancorados no próprio arquivo, não no diretório de onde se chama o
# script: data_extraction -> backend -> linear -> computer.
RAIZ          = Path(__file__).resolve().parents[3]
PASTA_VIDEOS  = str(RAIZ / "data" / "raw_videos")   # pasta raiz com subpastas por letra
JANELA        = 10                        # quantos frames por amostra
PULO          = 3                         # pula N frames entre janelas (evita amostras idênticas)
SAIDA         = str(RAIZ / "data" / "dataset_dynamic.npz")
# ======================================

def normalize_landmarks(pontos):
    base_x, base_y = pontos[0]
    norm = []
    for x, y in pontos:
        norm.append(x - base_x)
        norm.append(y - base_y)
    max_v = max(abs(v) for v in norm) or 1.0
    return [v / max_v for v in norm]

hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

X_all, y_all = [], []
total_videos = 0
letras = sorted([d for d in os.listdir(PASTA_VIDEOS)
                 if os.path.isdir(os.path.join(PASTA_VIDEOS, d))])
print(f"Letras encontradas: {letras}\n")

for letra in letras:
    pasta = Path(PASTA_VIDEOS) / letra
    videos = list(pasta.glob("*.mp4")) + list(pasta.glob("*.mov"))
    total_videos += len(videos)
    amostras_letra = 0

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        frames_lm = []  # landmarks de cada frame do vídeo

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small   = cv2.resize(frame, (320, 240))
            rgb     = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                pontos = [[lm.x, lm.y] for lm in results.multi_hand_landmarks[0].landmark]
                dados  = normalize_landmarks(pontos)
                frames_lm.append(dados)
            else:
                # frame sem mão — reseta sequência
                frames_lm = []

        cap.release()

        # Monta janelas deslizantes
        i = 0
        while i + JANELA <= len(frames_lm):
            janela = frames_lm[i:i + JANELA]
            X_all.append(np.array(janela).flatten())  # 10 * 42 = 420 valores
            y_all.append(letra)
            amostras_letra += 1
            i += PULO

    print(f"  {letra}: {len(videos)} vídeos → {amostras_letra} amostras")

hands.close()

# Dois casos bem diferentes que antes davam a mesma mensagem (enganosa):
# não existir vídeo NENHUM (normal — é só não ter gravado ainda) e existir
# vídeo mas nenhum servir (aí sim é problema nos clipes).
if total_videos == 0:
    print("\n⏭  Nenhum vídeo encontrado em 'raw_videos'.")
    print("   Isso é normal se você ainda não gravou letras com movimento")
    print("   (H, J, K, X, Z). Nada a fazer aqui — seguindo em frente.")
    raise SystemExit(0)

if len(X_all) == 0:
    print(f"\n❌ Achei {total_videos} vídeo(s), mas nenhuma amostra saiu deles:")
    print(f"   nenhum teve {JANELA} quadros seguidos com a mão visível.")
    print("   Regrave mantendo a mão dentro do quadro do início ao fim do clipe.")
    raise SystemExit(1)

X = np.array(X_all, dtype=np.float32)
y = np.array(y_all)

np.savez(SAIDA, X=X, y=y)
print(f"\n✅ Dataset salvo em '{SAIDA}'")
print(f"   Total de amostras: {len(X)}")
print(f"   Shape X: {X.shape}")
print(f"   Distribuição: { {l: int((y==l).sum()) for l in letras} }")
