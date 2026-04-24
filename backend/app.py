"""
Backend VisuAll — RF (estáticas) + MLP (movimento)
Sem LSTM, sem TensorFlow. Leve e rápido.
"""
import cv2
import mediapipe as mp
import json
import asyncio
import websockets
import base64
import time
import pickle
import os
import math
from collections import deque
from threading import Thread, Lock
import numpy as np

# ============================================================================
# MODELOS
# ============================================================================
print("🔄 Iniciando carregamento dos modelos...")
BASE_DIR              = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR            = os.path.join(BASE_DIR, "..", "models")
MODEL_MLP_PATH        = os.path.join(MODELS_DIR, "dynamic_model.pkl")
CLASSES_MLP_PATH      = os.path.join(MODELS_DIR, "dynamic_classes.pkl")
MODEL_ESTATICO_PATH   = os.path.join(MODELS_DIR, "static_model.pkl")
CLASSES_ESTATICO_PATH = os.path.join(MODELS_DIR, "static_classes.pkl")

# MLP Dinâmico (J, K, X, Z, H...)
try:
    with open(MODEL_MLP_PATH, "rb") as f:
        modelo_mlp = pickle.load(f)
    with open(CLASSES_MLP_PATH, "rb") as f:
        classes_mlp = pickle.load(f)
    print(f"✅ MLP dinâmico carregado ({len(classes_mlp)} letras: {list(classes_mlp)})")
except FileNotFoundError:
    modelo_mlp  = None
    classes_mlp = []
    print("⚠ MLP dinâmico não encontrado")
except Exception as e:
    modelo_mlp  = None
    classes_mlp = []
    print(f"⚠ Erro ao carregar MLP dinâmico: {e}")

# MLP Estático (A, B, C, D...)
try:
    with open(MODEL_ESTATICO_PATH, "rb") as f:
        modelo_estatico = pickle.load(f)
    with open(CLASSES_ESTATICO_PATH, "rb") as f:
        classes_estatico = pickle.load(f)
    print(f"✅ MLP estático carregado ({len(classes_estatico)} letras: {list(classes_estatico)})")
except FileNotFoundError:
    modelo_estatico  = None
    classes_estatico = []
    print("⚠ MLP estático não encontrado — letras estáticas indisponíveis")
except Exception as e:
    modelo_estatico  = None
    classes_estatico = []
    print(f"⚠ Erro ao carregar MLP estático: {e}")

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
JANELA_MLP        = 10
CONFIANCA_MINIMA  = 0.90
LIMIAR_MOVIMENTO  = 0.30   # limiar baixo para acionar MLP dinâmico
TEMPO_PRA_LIMPAR  = 3.0    # segundos segurando dedos esticados para limpar a frase

# ============================================================================
# MEDIAPIPE — FUNÇÕES AUXILIARES
# ============================================================================
def normalize_landmarks(pontos):
    """Normaliza landmarks em relação ao ponto base (pulso)."""
    base_x, base_y = pontos[0]
    norm = []
    for x, y in pontos:
        norm.append(x - base_x)
        norm.append(y - base_y)
    max_v = max(abs(v) for v in norm) or 1.0
    return [v / max_v for v in norm]


def detectar_dedos_esticados(lms):
    """
    Detecta dedos TOTALMENTE esticados — pontas bem acima das bases.
    Mais restrito que mão aberta comum, exige extensão máxima.
    Retorna True se todos os 4 dedos + polegar estiverem esticados.
    """
    MARGEM = 0.06
    indicador = (lms[8][1]  < lms[5][1]  - MARGEM)
    medio     = (lms[12][1] < lms[9][1]  - MARGEM)
    anelar    = (lms[16][1] < lms[13][1] - MARGEM)
    mindinho  = (lms[20][1] < lms[17][1] - MARGEM)
    polegar   = (abs(lms[4][0] - lms[0][0]) > 0.12)
    return indicador and medio and anelar and mindinho and polegar


def calcular_movimento(buffer):
    """Calcula a variação de posição da mão nos últimos frames."""
    if len(buffer) < 5:
        return 0.0
    rec = list(buffer)[-5:]
    # índices no vetor normalizado: [2]=x polegar base, [3]=y polegar base,
    # [16]=x ponta indicador, [17]=y ponta indicador
    try:
        xs  = [f[2]  for f in rec]
        ys  = [f[3]  for f in rec]
        xs2 = [f[16] for f in rec]
        ys2 = [f[17] for f in rec]
        return np.std(xs) + np.std(ys) + np.std(xs2) + np.std(ys2)
    except IndexError:
        return 0.0


# ============================================================================
# ESTADO COMPARTILHADO ENTRE THREADS
# ============================================================================
raw_frame_lock = Lock()
raw_frame      = {"img": None, "ts": 0}

data_lock   = Lock()
camera_data = {
    "status":                "Inicializando...",
    "hands_detected":        0,
    "frame_width":           0,
    "frame_height":          0,
    "frame":                 "",
    "timestamp":             0,
    "letra_atual":           "-",
    "frase":                 "",
    "confianca":             0.0,
    "historico":             [],
    "gesto_limpar_progress": 0.0,
    "modo_deteccao":         "estatico",
}


# ============================================================================
# THREAD DE CAPTURA DE CÂMERA
# ============================================================================
def capture_thread():
    """Captura frames da câmera e armazena no buffer compartilhado."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with data_lock:
            camera_data["status"] = "❌ Câmera não encontrada"
        print("❌ Câmera não encontrada")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    with data_lock:
        camera_data["status"] = "✅ Câmera aberta"
    print("✅ Câmera aberta")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)
        with raw_frame_lock:
            raw_frame["img"] = frame
            raw_frame["ts"]  = time.monotonic()

    cap.release()


# ============================================================================
# THREAD DE PROCESSAMENTO (MediaPipe + modelos)
# ============================================================================
def process_thread():
    """Processa frames: detecta mão, extrai landmarks, classifica gesto."""
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    draw = mp.solutions.drawing_utils

    ultima_predicao         = ""
    contador_estabilidade   = 0
    ultima_letra_adicionada = ""
    ultimo_tempo_adicao     = 0.0
    tempo_inicio_esticado   = None
    tempo_ultimo_limpar     = 0.0
    buffer_lm               = deque(maxlen=max(JANELA_MLP, 10))
    last_ts                 = 0

    while True:
        with raw_frame_lock:
            frame = raw_frame["img"]
            ts    = raw_frame["ts"]

        if frame is None or ts == last_ts:
            time.sleep(0.005)
            continue

        last_ts = ts
        frame   = frame.copy()
        h, w    = frame.shape[:2]

        # Reduz para 320x240 para MediaPipe rodar mais rápido
        small   = cv2.resize(frame, (320, 240))
        rgb     = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        letra_atual = "-"
        confianca   = 0.0
        modo        = "estatico"
        hands_det   = 0
        gesto_prog  = 0.0

        if results.multi_hand_landmarks:
            hands_det = 1
            hl  = results.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, hl, mp.solutions.hands.HAND_CONNECTIONS)
            lms = [[lm.x, lm.y] for lm in hl.landmark]

            dedos_esticados = detectar_dedos_esticados(lms)

            # ── Gesto de limpar frase (mão aberta por TEMPO_PRA_LIMPAR s) ──
            if dedos_esticados:
                now = time.time()
                if tempo_inicio_esticado is None:
                    tempo_inicio_esticado = now
                tempo_segurando = now - tempo_inicio_esticado
                gesto_prog = min(1.0, tempo_segurando / TEMPO_PRA_LIMPAR)

                if tempo_segurando >= TEMPO_PRA_LIMPAR and (now - tempo_ultimo_limpar) > 2.0:
                    with data_lock:
                        if camera_data["frase"].strip():
                            camera_data["historico"].insert(0, camera_data["frase"])
                            camera_data["historico"] = camera_data["historico"][:15]
                        camera_data["frase"] = ""
                    ultima_letra_adicionada = ""
                    tempo_inicio_esticado   = None
                    tempo_ultimo_limpar     = now
                    gesto_prog              = 0.0
                    print("✋ Frase limpa!")
            else:
                tempo_inicio_esticado = None

            # ── Classificação (só quando dedos NÃO estão esticados) ──
            if not dedos_esticados:
                dados = normalize_landmarks(lms)
                if dados:
                    buffer_lm.append(dados)
                    mov = calcular_movimento(buffer_lm) if len(buffer_lm) >= 5 else 0.0

                    usando_mlp = (mov > LIMIAR_MOVIMENTO
                                  and modelo_mlp is not None
                                  and len(buffer_lm) >= JANELA_MLP)

                    if usando_mlp:
                        # MLP Dinâmico — movimento detectado
                        modo    = "dinamico"
                        janela  = list(buffer_lm)[-JANELA_MLP:]
                        entrada = np.array(janela).flatten().reshape(1, -1)
                        probs   = modelo_mlp.predict_proba(entrada)[0]
                        idx     = int(np.argmax(probs))
                        confianca   = float(probs[idx])
                        letra_atual = classes_mlp[idx] if confianca >= CONFIANCA_MINIMA else "-"

                    elif modelo_estatico is not None:
                        # MLP Estático — mão parada
                        modo    = "estatico"
                        entrada = np.array(dados).reshape(1, -1)
                        probs   = modelo_estatico.predict_proba(entrada)[0]
                        idx     = int(np.argmax(probs))
                        confianca   = float(probs[idx])
                        letra_atual = classes_estatico[idx] if confianca >= CONFIANCA_MINIMA else "-"

                    # ── Estabilidade antes de adicionar à frase ──
                    if letra_atual != "-" and letra_atual == ultima_predicao:
                        contador_estabilidade += 1
                    else:
                        contador_estabilidade = 0
                    ultima_predicao = letra_atual

                    now       = time.time()
                    # Letras dinâmicas (MLP) precisam de menos frames consecutivos
                    # pois o gesto passa rápido — J, Z, etc.
                    estab_min = 2 if modo == "dinamico" else 12
                    cooldown  = 0.3 if modo == "dinamico" else 1.0
                    if (contador_estabilidade >= estab_min
                            and letra_atual != "-"
                            and letra_atual != ultima_letra_adicionada
                            and (now - ultimo_tempo_adicao) > cooldown):
                        with data_lock:
                            camera_data["frase"] += letra_atual
                        ultima_letra_adicionada = letra_atual
                        ultimo_tempo_adicao     = now
                        contador_estabilidade   = 0

                    # Reseta ultima_letra_adicionada após 1s pra permitir repetição
                    if (now - ultimo_tempo_adicao) > 1.0:
                        ultima_letra_adicionada = ""

        else:
            # ── CORREÇÃO: removidas referências a rf_res_lock e rf_result
            #    que não existem nesta versão do código ──
            ultima_predicao       = ""
            contador_estabilidade = 0
            tempo_inicio_esticado = None
            buffer_lm.clear()

        # ── Codifica frame em JPEG base64 para enviar via WebSocket ──
        _, buf  = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        jpg_b64 = base64.b64encode(buf).decode("utf-8")

        with data_lock:
            camera_data.update({
                "hands_detected":        hands_det,
                "frame_width":           w,
                "frame_height":          h,
                "letra_atual":           letra_atual,
                "confianca":             round(confianca, 2),
                "modo_deteccao":         modo,
                "gesto_limpar_progress": gesto_prog,
                "frame":                 jpg_b64,
                "timestamp":             time.time(),
            })


# ============================================================================
# SERVIDOR WEBSOCKET
# ============================================================================
async def send_data(websocket):
    """Gerencia uma conexão WebSocket: recebe comandos e envia dados da câmera."""
    print("🔌 Cliente conectado!")

    async def receive():
        """Recebe comandos do frontend (limpar, espaço, apagar, etc.)."""
        async for message in websocket:
            try:
                cmd = json.loads(message)
                act = cmd.get("action")
                with data_lock:
                    if act == "limpar":
                        if camera_data["frase"].strip():
                            camera_data["historico"].insert(0, camera_data["frase"])
                            camera_data["historico"] = camera_data["historico"][:15]
                        camera_data["frase"] = ""
                    elif act == "espaco":
                        camera_data["frase"] += " "
                    elif act == "apagar":
                        camera_data["frase"] = camera_data["frase"][:-1]
                    elif act == "limpar_historico":
                        camera_data["historico"] = []
                    elif act == "remover_item":
                        idx = cmd.get("index", -1)
                        if 0 <= idx < len(camera_data["historico"]):
                            camera_data["historico"].pop(idx)
            except (json.JSONDecodeError, KeyError):
                pass  # ignora mensagens malformadas

    async def send():
        """Envia estado atual da câmera a cada 50ms (~20fps)."""
        while True:
            with data_lock:
                payload = dict(camera_data)
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(0.05)

    try:
        await asyncio.gather(receive(), send())
    except websockets.exceptions.ConnectionClosed:
        print("🔌 Cliente desconectado")


async def main():
    print("🚀 Iniciando WebSocket na porta 8000...")
    async with websockets.serve(send_data, "localhost", 8000):
        print("✅ Servidor pronto! MLP dinâmico + MLP estático ativos")
        print("   Aguardando conexão do frontend em ws://localhost:8000")
        await asyncio.Future()  # roda para sempre


# ============================================================================
# PONTO DE ENTRADA
# ============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  VisuAll Backend — iniciando threads...")
    print("=" * 50)

    t_cap  = Thread(target=capture_thread,  daemon=True, name="CaptureThread")
    t_proc = Thread(target=process_thread,  daemon=True, name="ProcessThread")
    t_cap.start()
    t_proc.start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Encerrado pelo usuário")
