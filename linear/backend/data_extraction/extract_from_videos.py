"""
Extrai landmarks dos vídeos e salva como dataset para treinar a MLP.

Estrutura esperada:
  videos/
    H/  video1.mp4  video2.mp4 ...
    J/  ...
    K/  ...
    X/  ...
    Z/  ...

Gera: dataset_mlp.npz  (X com shape [N, 420], y com shape [N])
"""
import csv
import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

# ============ CONFIGURAÇÃO ============
ROOT          = Path(__file__).resolve().parents[3]
PASTA_VIDEOS  = ROOT / "data" / "raw_videos"   # pasta raiz com subpastas por letra
JANELA        = 10                        # quantos frames por amostra
PULO          = 3                         # pula N frames entre janelas (evita amostras idênticas)
SAIDA         = ROOT / "data" / "dataset_dynamic.npz"
SAIDA_CSV     = ROOT / "data" / "dynamic_external_dataset.csv"
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

X_all, y_all, rows_csv = [], [], []
letras = sorted([d for d in os.listdir(PASTA_VIDEOS)
                 if os.path.isdir(os.path.join(PASTA_VIDEOS, d))])
print(f"Letras encontradas: {letras}\n")

for letra in letras:
    pasta = Path(PASTA_VIDEOS) / letra
    videos = list(pasta.glob("*.mp4")) + list(pasta.glob("*.mov"))
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
            features = np.array(janela).flatten()  # 10 * 42 = 420 valores
            X_all.append(features)
            y_all.append(letra)
            rows_csv.append((video_path.stat().st_mtime_ns + i, letra, "external_video", features))
            amostras_letra += 1
            i += PULO

    print(f"  {letra}: {len(videos)} vídeos → {amostras_letra} amostras")

hands.close()

X = np.array(X_all, dtype=np.float32)
y = np.array(y_all)

SAIDA.parent.mkdir(parents=True, exist_ok=True)
np.savez(SAIDA, X=X, y=y)
with SAIDA_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "label", "source"] + [f"f{i}" for i in range(420)])
    for timestamp, label, source, dados in rows_csv:
        writer.writerow([timestamp, label, source] + list(dados))

print(f"\n✅ Dataset salvo em '{SAIDA}'")
print(f"✅ CSV mobile salvo em '{SAIDA_CSV}'")
print(f"   Total de amostras: {len(X)}")
print(f"   Shape X: {X.shape}")
print(f"   Distribuição: { {l: int((y==l).sum()) for l in letras} }")
