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
from collections import Counter
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from pathlib import Path

# ============ CONFIGURAÇÃO ============
ROOT          = Path(__file__).resolve().parents[3]
PASTA_VIDEOS  = ROOT / "data" / "raw_videos"   # pasta raiz com subpastas por letra
JANELA        = 10                        # quantos frames por amostra
PULO          = 3                         # pula N frames entre janelas (evita amostras idênticas)
SAIDA         = ROOT / "data" / "dataset_dynamic.npz"
SAIDA_CSV     = ROOT / "data" / "dynamic_external_dataset.csv"
# Mesmo arquivo .task usado no celular — ver comentário em extract_from_images.py.
MODELO_MAOS   = ROOT / "mobile" / "app" / "src" / "main" / "assets" / "hand_landmarker.task"
# ======================================

def normalize_landmarks(pontos):
    base_x, base_y = pontos[0]
    norm = []
    for x, y in pontos:
        norm.append(x - base_x)
        norm.append(y - base_y)
    max_v = max(abs(v) for v in norm) or 1.0
    return [v / max_v for v in norm]

def novo_detector():
    # RunningMode.VIDEO ativa o rastreamento temporal entre frames (igual ao
    # HandLandmarker do celular) em vez de redetectar do zero a cada frame —
    # reduz jitter nos landmarks e casa a extração com a inferência real.
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELO_MAOS)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
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
        # Um detector novo por vídeo: RunningMode.VIDEO exige timestamps
        # estritamente crescentes dentro de UM stream contínuo, e cada vídeo
        # aqui é uma sequência independente (não um único stream ao vivo).
        hands = novo_detector()
        frame_idx = 0
        frames_lm = []  # landmarks de cada frame do vídeo

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small      = cv2.resize(frame, (320, 240))
            rgb        = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = frame_idx * 33  # ~30fps; só precisa ser crescente
            results    = hands.detect_for_video(mp_image, timestamp_ms)
            frame_idx += 1

            if results.hand_landmarks:
                pontos = [[lm.x, lm.y] for lm in results.hand_landmarks[0]]
                dados  = normalize_landmarks(pontos)
                frames_lm.append(dados)
            else:
                # frame sem mão — reseta sequência
                frames_lm = []

        hands.close()
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
# Counter em vez de (y==l).sum(): com y vazio, a comparação numpy retorna um
# escalar (não um array), e .sum() quebra com AttributeError.
contagem = Counter(y.tolist())
print(f"   Distribuição: { {l: contagem.get(l, 0) for l in letras} }")
