"""
Extrai landmarks (pose + mãos) dos vídeos de gestos corporais e monta o
dataset pra treinar o modelo do modo CORPO.

Estrutura esperada:
  data/raw_videos_corpo/
    AJUDAR/      video1.mp4  video2.mp4 ...
    COMPUTADOR/  ...
    CONVERSAR/   ...
    NEUTRO/      ...   (pessoa parada, mão à mostra, sem sinalizar nada)
    PESSOA/      ...
    SURDO/       ...

Gera: data/dataset_corpo.npz  (X com shape [N, 30, 225], y com shape [N])

Este extrator não existia no repositório (o body_model.tflite publicado veio
de um pipeline externo, "vindo da pasta Articulação" — ver docstring de
linear/backend/app.py). O contrato de features abaixo foi reconstruído a
partir do consumidor real (mobile/app/.../BodyGestureEngine.kt, que é um
port 1:1 do Python de referência) para garantir que o dataset gerado aqui
bate exatamente com o que o app espera na hora de rodar o modelo:

  - 75 pontos * 3 coords (x,y,z) = 225 valores por frame:
      [0:33]  pose (33 pontos, na ordem do PoseLandmarker)
      [33:54] mão ESQUERDA (21 pontos, handedness "Left")
      [54:75] mão DIREITA  (21 pontos, handedness "Right")
    Ponto/mão ausente no frame fica com [0,0,0] nessas posições.
  - o x de TODOS os pontos (pose e mãos) é multiplicado por
    aspect_x = 0.75 * (largura/altura do frame) antes de normalizar — mesma
    correção de proporção retrato->4:3 usada no celular (ver aspectX em
    LibrasAnalyzer.kt). Gravar em 4:3 (o padrão deste extrator/capturar.py)
    deixa aspect_x ≈ 1.0, ou seja, na prática é quase um no-op; a fórmula só
    existe pra manter o mesmo comportamento se algum vídeo vier em outra
    proporção.
  - normalização (normalizar_frame): centraliza x,y em
    (ombro_esq + ombro_dir)/2 (pontos 11 e 12 do pose) e escala pela
    distância 3D entre os ombros (x,y,z, não só x,y). Só x,y são
    normalizados; z fica cru (saída direta do MediaPipe).
  - só entra no buffer o frame que tem pose E pelo menos uma mão detectada —
    igual ao LibrasAnalyzer só chamar processarFrame() nessas condições.
  - janela final: sempre 30 frames, reamostrados pelo MESMO algoritmo de
    seleção de índice do Kotlin (resample: pega frames existentes num índice
    calculado, não interpola valores) — não pelo np.linspace tradicional.

Sem espelhamento aqui, igual extract_from_images.py/extract_from_videos.py
(letras) — o vídeo é processado exatamente como veio gravado. O app já
espelha a própria câmera frontal ao vivo antes de detectar (espelharImagem
em LibrasAnalyzer.kt), então a correção de espelho é responsabilidade de
quem grava (treinamento/capturar.py salva o quadro cru, sem espelhar) e do
app na hora de usar — não do extrator.
"""
import cv2
import mediapipe as mp
import numpy as np
import os
from collections import Counter
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from pathlib import Path

# ============ CONFIGURAÇÃO ============
ROOT               = Path(__file__).resolve().parents[3]
PASTA_VIDEOS_CORPO = ROOT / "data" / "raw_videos_corpo"
JANELA_MODELO      = 30     # frames por amostra (BODY_WINDOW no Kotlin)
MIN_FRAMES_VALIDOS = 10     # abaixo disso o clipe é descartado (BODY_MIN_FRAMES)
N_POSE             = 33
N_HAND             = 21
N_FEATURES         = (N_POSE + N_HAND * 2) * 3   # 225
IDX_OMBRO_ESQ      = 11
IDX_OMBRO_DIR      = 12
SAIDA              = ROOT / "data" / "dataset_corpo.npz"
# Mesmos arquivos .task usados no celular.
MODELO_MAOS = ROOT / "mobile" / "app" / "src" / "main" / "assets" / "hand_landmarker.task"
MODELO_POSE = ROOT / "mobile" / "app" / "src" / "main" / "assets" / "pose_landmarker_lite.task"
# ======================================


def novo_detector_maos():
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELO_MAOS)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            # Mesmos limiares do HandLandmarker do celular (LibrasAnalyzer.kt).
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
    )


def novo_detector_pose():
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELO_POSE)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            # Mesmos limiares do PoseLandmarker do celular (BodyGestureEngine.kt).
            min_pose_detection_confidence=0.35,
            min_pose_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
    )


def escrever_ponto(frame_vec, indice_ponto, x, y, z):
    base = indice_ponto * 3
    frame_vec[base] = x
    frame_vec[base + 1] = y
    frame_vec[base + 2] = z


def extrair_frame(pose_result, hand_result, aspect_x):
    """Monta o vetor cru de 225 valores (sem normalizar) de um frame —
    equivalente a BodyGestureEngine.extractFrame()."""
    frame_vec = np.zeros(N_FEATURES, dtype=np.float32)

    has_pose = bool(pose_result.pose_landmarks)
    if has_pose:
        for i, lm in enumerate(pose_result.pose_landmarks[0][:N_POSE]):
            escrever_ponto(frame_vec, i, lm.x * aspect_x, lm.y, lm.z)

    has_hand = False
    for hand_idx, landmarks in enumerate(hand_result.hand_landmarks):
        rotulo = ""
        if hand_idx < len(hand_result.handedness) and hand_result.handedness[hand_idx]:
            rotulo = hand_result.handedness[hand_idx][0].category_name
        if rotulo.lower() == "left":
            offset = N_POSE
        elif rotulo.lower() == "right":
            offset = N_POSE + N_HAND
        else:
            avg_x = sum(lm.x for lm in landmarks) / len(landmarks)
            offset = N_POSE if avg_x < 0.5 else N_POSE + N_HAND
        for i, lm in enumerate(landmarks[:N_HAND]):
            escrever_ponto(frame_vec, offset + i, lm.x * aspect_x, lm.y, lm.z)
        has_hand = True

    return frame_vec, has_pose, has_hand


def normalizar_frame(frame_vec):
    """Centraliza em (ombro_esq+ombro_dir)/2 e escala pela distância 3D
    entre os ombros — só x,y; z fica cru. Equivalente a
    BodyGestureEngine.normalize()."""
    normalizado = frame_vec.copy()
    base_esq = IDX_OMBRO_ESQ * 3
    base_dir = IDX_OMBRO_DIR * 3
    center_x = (frame_vec[base_esq] + frame_vec[base_dir]) / 2.0
    center_y = (frame_vec[base_esq + 1] + frame_vec[base_dir + 1]) / 2.0
    dx = frame_vec[base_esq] - frame_vec[base_dir]
    dy = frame_vec[base_esq + 1] - frame_vec[base_dir + 1]
    dz = frame_vec[base_esq + 2] - frame_vec[base_dir + 2]
    escala = (dx * dx + dy * dy + dz * dz) ** 0.5
    if escala < 0.0001:
        escala = 1.0
    n_pontos = N_POSE + N_HAND * 2
    for ponto in range(n_pontos):
        base = ponto * 3
        normalizado[base] = (normalizado[base] - center_x) / escala
        normalizado[base + 1] = (normalizado[base + 1] - center_y) / escala
    return normalizado


def reamostrar(frames, count=JANELA_MODELO):
    """Mesmo algoritmo de BodyGestureEngine.resample(): escolhe frames já
    existentes por índice calculado, não interpola valores."""
    if len(frames) == count:
        return frames
    resultado = []
    for i in range(count):
        indice_origem = int((len(frames) - 1) * i / (count - 1))
        resultado.append(frames[indice_origem])
    return resultado


def processar_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    largura = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640
    altura = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480
    aspect_x = 0.75 * (largura / altura)

    # Um detector novo por vídeo: RunningMode.VIDEO exige timestamps
    # estritamente crescentes dentro de UM stream contínuo.
    maos = novo_detector_maos()
    pose = novo_detector_pose()
    frames_validos = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = frame_idx * 33  # ~30fps; só precisa ser crescente
        frame_idx += 1

        pose_result = pose.detect_for_video(mp_image, timestamp_ms)
        hand_result = maos.detect_for_video(mp_image, timestamp_ms)

        frame_vec, has_pose, has_hand = extrair_frame(pose_result, hand_result, aspect_x)
        if has_pose and has_hand:
            frames_validos.append(normalizar_frame(frame_vec))

    maos.close()
    pose.close()
    cap.release()
    return frames_validos


def main():
    if not PASTA_VIDEOS_CORPO.exists():
        print(f"❌ Pasta não encontrada: {PASTA_VIDEOS_CORPO}")
        print("   Grave clipes com o Capturar (treinamento/) antes de rodar isto.")
        return

    X_all, y_all = [], []
    gestos = sorted([d for d in os.listdir(PASTA_VIDEOS_CORPO)
                     if os.path.isdir(PASTA_VIDEOS_CORPO / d)])
    print(f"Gestos encontrados: {gestos}\n")

    for gesto in gestos:
        pasta = PASTA_VIDEOS_CORPO / gesto
        videos = list(pasta.glob("*.mp4")) + list(pasta.glob("*.mov")) + list(pasta.glob("*.avi"))
        amostras_gesto = 0
        descartados = 0

        for video_path in videos:
            frames_validos = processar_video(video_path)
            if len(frames_validos) < MIN_FRAMES_VALIDOS:
                descartados += 1
                continue
            janela = reamostrar(frames_validos, JANELA_MODELO)
            X_all.append(np.array(janela, dtype=np.float32))
            y_all.append(gesto)
            amostras_gesto += 1

        aviso = f"  ({descartados} descartados: poucos frames com pose+mão)" if descartados else ""
        print(f"  {gesto}: {len(videos)} vídeos → {amostras_gesto} amostras{aviso}")

    if not X_all:
        print("\n❌ Nenhuma amostra extraída! Verifique os vídeos e a pasta.")
        return

    X = np.array(X_all, dtype=np.float32)  # [N, 30, 225]
    y = np.array(y_all)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SAIDA, X=X, y=y)

    print(f"\n✅ Dataset salvo em '{SAIDA}'")
    print(f"   Total de amostras: {len(X)}")
    print(f"   Shape X: {X.shape}")
    contagem = Counter(y.tolist())
    print(f"   Distribuição: { {g: contagem.get(g, 0) for g in gestos} }")


if __name__ == "__main__":
    main()
