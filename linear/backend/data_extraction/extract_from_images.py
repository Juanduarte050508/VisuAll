"""
Extrai landmarks de IMAGENS estáticas e salva dataset para treinar MLP estático.

Estrutura esperada:
  imagens/
    A/  foto1.jpg  foto2.png ...
    B/  ...
    C/  ...
    ...

Gera: dataset_mlp_estatico.npz  (X com shape [N, 42], y com shape [N])
"""
import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

# ============ CONFIGURAÇÃO ============
PASTA_IMAGENS = "../../data/raw_images"   # pasta raiz com subpastas por letra
SAIDA         = "../../data/dataset_static.npz"
EXTENSOES     = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
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
    static_image_mode=True,       # True para imagens estáticas
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
)

X_all, y_all = [], []
letras = sorted([d for d in os.listdir(PASTA_IMAGENS)
                 if os.path.isdir(os.path.join(PASTA_IMAGENS, d))])
print(f"Letras encontradas: {letras}\n")

for letra in letras:
    pasta = Path(PASTA_IMAGENS) / letra
    imagens = [p for p in pasta.iterdir() if p.suffix.lower() in EXTENSOES]
    amostras_letra = 0
    falhas = 0

    for img_path in imagens:
        frame = cv2.imread(str(img_path))
        if frame is None:
            falhas += 1
            continue

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            pontos = [[lm.x, lm.y] for lm in results.multi_hand_landmarks[0].landmark]
            dados  = normalize_landmarks(pontos)   # 42 valores (21 pontos * x,y)
            X_all.append(dados)
            y_all.append(letra)
            amostras_letra += 1
        else:
            falhas += 1

    print(f"  {letra}: {len(imagens)} imagens → {amostras_letra} amostras  "
          f"({falhas} sem mão detectada)")

hands.close()

if len(X_all) == 0:
    print("\n❌ Nenhuma amostra extraída! Verifique as imagens e a pasta.")
else:
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all)

    np.savez(SAIDA, X=X, y=y)
    print(f"\n✅ Dataset salvo em '{SAIDA}'")
    print(f"   Total de amostras: {len(X)}")
    print(f"   Shape X: {X.shape}")
    print(f"   Distribuição: { {l: int((y==l).sum()) for l in letras} }")
