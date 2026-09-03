"""
VisuAll — Backend UNIFICADO (Opção A: botão de modo)
=====================================================
Um app só, com 2 modos que NUNCA rodam ao mesmo tempo:
  - "alfabeto"  -> MLP estático (A,B,C..) + MLP dinâmico (H,J,K,X,Z)  [só mão]
  - "corpo"     -> modelo Keras de sinais (AJUDAR, PESSOA, SURDO..)   [pose+mãos]
Em AMBOS os modos o ROSTO entra como marcador não-manual:
  sobrancelha levantada = INTERROGATIVO (frase vira pergunta).

O frontend escolhe o modo mandando: {"action":"modo","modo":"alfabeto"|"corpo"}.
Só o modelo do modo ativo interpreta — por isso não sai resultado duplicado.

Usa SÓ o MediaPipe Holistic (mão + corpo + rosto num detector só).

>>> Modelos Python do alfabeto:
    computer/models/static_model.pkl, static_classes.pkl
    computer/models/dynamic_model.pkl, dynamic_classes.pkl

>>> Modelos corporais legados, se existirem, ficam junto deste arquivo:
    modelo_sinais.h5, classes_sinais.npy   (vindos da pasta Articulacao)
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
from pathlib import Path
from collections import deque
from threading import Thread, Lock
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def P(nome): return os.path.join(BASE_DIR, nome)
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

print("🔄 Carregando modelos...")

# ── Alfabeto: MLP estático + dinâmico ───────────────────────────────────────
def carrega_pickle(caminho):
    with open(caminho, "rb") as f:
        return pickle.load(f)

try:
    modelo_estatico  = carrega_pickle(MODELS_DIR / "static_model.pkl")
    classes_estatico = carrega_pickle(MODELS_DIR / "static_classes.pkl")
    print(f"✅ MLP estático ({len(classes_estatico)} letras)")
except Exception as e:
    modelo_estatico, classes_estatico = None, []
    print(f"⚠ MLP estático indisponível: {e}")

try:
    modelo_mlp  = carrega_pickle(MODELS_DIR / "dynamic_model.pkl")
    classes_mlp = carrega_pickle(MODELS_DIR / "dynamic_classes.pkl")
    print(f"✅ MLP dinâmico ({len(classes_mlp)} letras)")
except Exception as e:
    modelo_mlp, classes_mlp = None, []
    print(f"⚠ MLP dinâmico indisponível: {e}")

# ── Corpo: modelo Keras ──────────────────────────────────────────────────────
try:
    modelo_corpo  = load_model(P("modelo_sinais.h5"))
    classes_corpo = np.load(P("classes_sinais.npy"), allow_pickle=True)
    print(f"✅ Modelo de corpo ({len(classes_corpo)} sinais: {list(classes_corpo)})")
except Exception as e:
    modelo_corpo, classes_corpo = None, []
    print(f"⚠ Modelo de corpo indisponível: {e}")

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
# Alfabeto
JANELA_MLP       = 10
CONFIANCA_MINIMA = 0.90
LIMIAR_MOVIMENTO = 0.30
TEMPO_PRA_LIMPAR = 3.0
# Corpo
N_POSE, N_HAND   = 33, 21
N_FEATURES       = (N_POSE + 2 * N_HAND) * 3          # 225
JANELA_MODELO    = 30
LIMIAR_INICIO    = 0.050
LIMIAR_FIM       = 0.030
FRAMES_INICIO    = 3
FRAMES_FIM       = 5
MIN_FRAMES_GESTO = 10
MAX_FRAMES_GESTO = 60
CONFIANCA_CORPO  = 0.85
COOLDOWN_CORPO   = 2.0
CLASSE_NEUTRO    = "NEUTRO"
JANELA_MOV       = 5
TEMPO_LIMPAR_CORPO = 5.0   # segundos de mão estendida pra zerar a frase no modo corpo
# Rosto (marcador)
LIMIAR_SOBRANCELHA = 0.38
JANELA_SOBR        = 5
IDX_BROW_L, IDX_BROW_R       = 105, 334
IDX_EYE_TOP_L, IDX_EYE_TOP_R = 159, 386
IDX_EYE_OUT_L, IDX_EYE_OUT_R = 33, 263

VOCABULARIO = {
    "PESSOA":     {"tipo": "subst", "genero": "f", "palavra": "pessoa",     "artigo": "a"},
    "SURDO":      {"tipo": "adj",   "masc": "surdo",   "fem": "surda"},
    "CONVERSAR":  {"tipo": "verbo", "conj": "conversa", "inf": "conversar"},
    "COMPUTADOR": {"tipo": "subst", "genero": "m", "palavra": "computador", "artigo": "o"},
    "AJUDAR":     {"tipo": "verbo", "conj": "ajuda",    "inf": "ajudar"},
}

def traduzir_frase(palavras, interrogativo=False):
    partes, ult_gen, ult_tipo = [], None, None
    for i, p in enumerate(palavras):
        p = p.upper()
        if p == "NEUTRO":
            continue
        if p not in VOCABULARIO:
            # palavra desconhecida (ex.: digitada no alfabeto, tipo JOVI)
            # -> tratada como substantivo proprio (sem artigo)
            partes.append(p.capitalize()); ult_gen = "m"; ult_tipo = "subst"
            continue
        v = VOCABULARIO[p]; t = v["tipo"]
        if t == "subst":
            art = v["artigo"].capitalize() if i == 0 else v["artigo"]
            partes.append(f"{art} {v['palavra']}"); ult_gen = v["genero"]; ult_tipo = "subst"
        elif t == "adj":
            partes.append(v["fem"] if ult_gen == "f" else v["masc"]); ult_tipo = "adj"
        elif t == "verbo":
            partes.append(f"a {v['inf']}" if ult_tipo == "verbo" else v["conj"]); ult_tipo = "verbo"
    if not partes:
        return ""
    frase = " ".join(partes)
    frase = frase[0].upper() + frase[1:]
    return frase + ("?" if interrogativo else "")


def montar_exibicao(tokens, palavra_atual, sobr_ativo):
    """Monta o texto mostrado: tokens ja fechados (traduzidos) + a palavra
    que esta sendo digitada agora (letras cruas). '?' se sobrancelha levantada."""
    base = traduzir_frase(list(tokens))
    if palavra_atual:
        base = (base + " " + palavra_atual).strip()
    if sobr_ativo and base.strip() and not base.rstrip().endswith("?"):
        base = base.rstrip() + "?"
    return base

# ============================================================================
# FUNÇÕES — ALFABETO (mão)
# ============================================================================
def normalize_landmarks(pontos):
    bx, by = pontos[0]
    norm = []
    for x, y in pontos:
        norm += [x - bx, y - by]
    mv = max(abs(v) for v in norm) or 1.0
    return [v / mv for v in norm]

def detectar_dedos_esticados(lms):
    M = 0.06
    ind = lms[8][1]  < lms[5][1]  - M
    med = lms[12][1] < lms[9][1]  - M
    ane = lms[16][1] < lms[13][1] - M
    mind = lms[20][1] < lms[17][1] - M
    pol = abs(lms[4][0] - lms[0][0]) > 0.12
    return ind and med and ane and mind and pol

def calcular_movimento(buffer):
    if len(buffer) < 5:
        return 0.0
    rec = list(buffer)[-5:]
    try:
        return (np.std([f[2] for f in rec]) + np.std([f[3] for f in rec]) +
                np.std([f[16] for f in rec]) + np.std([f[17] for f in rec]))
    except IndexError:
        return 0.0

# ============================================================================
# FUNÇÕES — CORPO (holistic)
# ============================================================================
def extrair_pontos(res):
    pose = np.zeros((N_POSE, 3)); me = np.zeros((N_HAND, 3)); md = np.zeros((N_HAND, 3))
    tem = False
    if res.pose_landmarks:
        pose = np.array([[l.x, l.y, l.z] for l in res.pose_landmarks.landmark])
    if res.left_hand_landmarks:
        me = np.array([[l.x, l.y, l.z] for l in res.left_hand_landmarks.landmark]); tem = True
    if res.right_hand_landmarks:
        md = np.array([[l.x, l.y, l.z] for l in res.right_hand_landmarks.landmark]); tem = True
    return np.concatenate([pose, me, md], axis=0), tem

def normalizar_frame(frame):
    c = (frame[11] + frame[12]) / 2.0
    e = np.linalg.norm(frame[11] - frame[12]) or 1.0
    frame = frame.copy()
    frame[:, :2] = (frame[:, :2] - c[:2]) / e
    return frame

def mov_inst(janela):
    if len(janela) < 3:
        return 0.0
    return float(np.array(janela)[:, N_POSE:, :2].std(axis=0).mean())

def reamostrar(frames, n):
    if len(frames) == n:
        return np.array(frames)
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return np.array([frames[i] for i in idx])

def classificar_corpo(gesto):
    if len(gesto) < MIN_FRAMES_GESTO or modelo_corpo is None:
        return None, 0.0
    seq = reamostrar(gesto, JANELA_MODELO).reshape(JANELA_MODELO, N_FEATURES)
    probs = modelo_corpo.predict(np.expand_dims(seq, 0), verbose=0)[0]
    idx = int(np.argmax(probs))
    return classes_corpo[idx], float(probs[idx])

# ============================================================================
# FUNÇÃO — ROSTO (marcador)
# ============================================================================
_buf_sobr = deque(maxlen=JANELA_SOBR)
def ler_marcador(res):
    if not res.face_landmarks:
        return None, False
    lm = res.face_landmarks.landmark
    dx = lm[IDX_EYE_OUT_L].x - lm[IDX_EYE_OUT_R].x
    dy = lm[IDX_EYE_OUT_L].y - lm[IDX_EYE_OUT_R].y
    esc = (dx * dx + dy * dy) ** 0.5
    if esc < 1e-6:
        return None, False
    gl = lm[IDX_EYE_TOP_L].y - lm[IDX_BROW_L].y
    gr = lm[IDX_EYE_TOP_R].y - lm[IDX_BROW_R].y
    _buf_sobr.append(((gl + gr) / 2.0) / esc)
    suav = float(np.mean(_buf_sobr))
    return suav, (suav >= LIMIAR_SOBRANCELHA)

# ============================================================================
# ESTADO COMPARTILHADO
# ============================================================================
raw_frame_lock = Lock()
raw_frame = {"img": None, "ts": 0}
data_lock = Lock()
camera_data = {
    "status": "Inicializando...", "hands_detected": 0,
    "frame": "", "timestamp": 0,
    "letra_atual": "-", "frase": "", "confianca": 0.0,
    "tokens": [], "palavra_atual": "",
    "historico": [], "gesto_limpar_progress": 0.0,
    "modo_app": "alfabeto", "marcador": False, "sobr_val": 0.0,
    "desenho_ativo": True,
}

# ============================================================================
# CAPTURA
# ============================================================================
def capture_thread():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with data_lock:
            camera_data["status"] = "❌ Câmera não encontrada"
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("✅ Câmera aberta")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01); continue
        frame = cv2.flip(frame, 1)
        with raw_frame_lock:
            raw_frame["img"] = frame
            raw_frame["ts"] = time.monotonic()

# ============================================================================
# PROCESSAMENTO
# ============================================================================
def process_thread():
    holistic = mp.solutions.holistic.Holistic(
        model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    mp_h    = mp.solutions.holistic
    mp_hands = mp.solutions.hands

    # estado alfabeto
    ult_pred = ""; estab = 0; ult_add = ""; t_add = 0.0
    t_inicio_estic = None; t_ult_limpar = 0.0
    buf_lm = deque(maxlen=max(JANELA_MLP, 10))
    # estado corpo
    buf_mov = deque(maxlen=JANELA_MOV); gesto = []; marc_gesto = []
    estado_corpo = "OCIOSO"; c_ini = 0; c_fim = 0
    ult_pal_corpo = "-"; ult_conf_corpo = 0.0; t_corpo = 0.0
    t_estic_corpo = None; t_ult_limpar_corpo = 0.0
    last_ts = 0

    while True:
        with raw_frame_lock:
            frame = raw_frame["img"]; ts = raw_frame["ts"]
        if frame is None or ts == last_ts:
            time.sleep(0.005); continue
        last_ts = ts
        frame = frame.copy()
        with data_lock:
            modo = camera_data["modo_app"]
            desenhar_linhas = camera_data["desenho_ativo"]

        rgb = cv2.cvtColor(cv2.resize(frame, (480, 360)), cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = holistic.process(rgb)
        rgb.flags.writeable = True

        sobr_val, sobr_ativo = ler_marcador(res)
        letra_atual = "-"; confianca = 0.0; hands_det = 0; gesto_prog = 0.0

        # desenho leve do rosto (mostra que o rosto é lido nos 2 modos)
        if desenhar_linhas and res.face_landmarks:
            mp_draw.draw_landmarks(frame, res.face_landmarks, mp_h.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())

        # ───────────────────────── MODO ALFABETO ─────────────────────────
        if modo == "alfabeto":
            hl = res.right_hand_landmarks or res.left_hand_landmarks
            if hl:
                hands_det = 1
                if desenhar_linhas:
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                lms = [[lm.x, lm.y] for lm in hl.landmark]
                esticados = detectar_dedos_esticados(lms)

                if esticados:
                    now = time.time()
                    if t_inicio_estic is None:
                        t_inicio_estic = now
                    seg = now - t_inicio_estic
                    gesto_prog = min(1.0, seg / TEMPO_PRA_LIMPAR)
                    if seg >= TEMPO_PRA_LIMPAR and (now - t_ult_limpar) > 2.0:
                        with data_lock:
                            render = montar_exibicao(camera_data["tokens"], camera_data["palavra_atual"], False)
                            if render.strip():
                                camera_data["historico"].insert(0, render)
                                camera_data["historico"] = camera_data["historico"][:15]
                            camera_data["tokens"] = []; camera_data["palavra_atual"] = ""
                        ult_add = ""; t_inicio_estic = None; t_ult_limpar = now; gesto_prog = 0.0
                else:
                    t_inicio_estic = None

                if not esticados:
                    dados = normalize_landmarks(lms)
                    buf_lm.append(dados)
                    mov = calcular_movimento(buf_lm)
                    usa_mlp = (mov > LIMIAR_MOVIMENTO and modelo_mlp is not None and len(buf_lm) >= JANELA_MLP)
                    if usa_mlp:
                        janela = list(buf_lm)[-JANELA_MLP:]
                        probs = modelo_mlp.predict_proba(np.array(janela).flatten().reshape(1, -1))[0]
                        idx = int(np.argmax(probs)); confianca = float(probs[idx])
                        letra_atual = classes_mlp[idx] if confianca >= CONFIANCA_MINIMA else "-"
                        estab_min, cooldown = 2, 0.3
                    elif modelo_estatico is not None:
                        probs = modelo_estatico.predict_proba(np.array(dados).reshape(1, -1))[0]
                        idx = int(np.argmax(probs)); confianca = float(probs[idx])
                        letra_atual = classes_estatico[idx] if confianca >= CONFIANCA_MINIMA else "-"
                        estab_min, cooldown = 12, 1.0
                    else:
                        estab_min, cooldown = 12, 1.0

                    if letra_atual != "-" and letra_atual == ult_pred:
                        estab += 1
                    else:
                        estab = 0
                    ult_pred = letra_atual
                    now = time.time()
                    if (estab >= estab_min and letra_atual != "-"
                            and letra_atual != ult_add and (now - t_add) > cooldown):
                        with data_lock:
                            camera_data["palavra_atual"] += letra_atual
                        ult_add = letra_atual; t_add = now; estab = 0
                    if (now - t_add) > 1.0:
                        ult_add = ""
            else:
                ult_pred = ""; estab = 0; t_inicio_estic = None; buf_lm.clear()

        # ───────────────────────── MODO CORPO ─────────────────────────
        else:
            pts, tem_mao = extrair_pontos(res)
            buf_mov.append(normalizar_frame(pts))
            mov = mov_inst(buf_mov)
            hands_det = 1 if tem_mao else 0
            # desenha pose + mãos
            if desenhar_linhas and res.pose_landmarks:
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_h.POSE_CONNECTIONS)
            if desenhar_linhas:
                for hlm in (res.left_hand_landmarks, res.right_hand_landmarks):
                    if hlm:
                        mp_draw.draw_landmarks(frame, hlm, mp_h.HAND_CONNECTIONS)

            # ── Gesto de limpar: mão estendida por TEMPO_LIMPAR_CORPO segundos ──
            hl_c = res.right_hand_landmarks or res.left_hand_landmarks
            if hl_c and detectar_dedos_esticados([[lm.x, lm.y] for lm in hl_c.landmark]):
                now = time.time()
                if t_estic_corpo is None:
                    t_estic_corpo = now
                seg = now - t_estic_corpo
                gesto_prog = min(1.0, seg / TEMPO_LIMPAR_CORPO)
                if seg >= TEMPO_LIMPAR_CORPO and (now - t_ult_limpar_corpo) > 2.0:
                    with data_lock:
                        render = montar_exibicao(camera_data["tokens"], camera_data["palavra_atual"], False)
                        if render.strip():
                            camera_data["historico"].insert(0, render)
                            camera_data["historico"] = camera_data["historico"][:15]
                        camera_data["tokens"] = []; camera_data["palavra_atual"] = ""
                    ult_pal_corpo = "-"; t_estic_corpo = None
                    t_ult_limpar_corpo = now; gesto_prog = 0.0
            else:
                t_estic_corpo = None

            if estado_corpo == "OCIOSO":
                if tem_mao and mov > LIMIAR_INICIO:
                    c_ini += 1
                    if c_ini >= FRAMES_INICIO:
                        estado_corpo = "CAPTURANDO"; gesto = list(buf_mov); marc_gesto = [sobr_ativo]; c_fim = 0
                else:
                    c_ini = 0
            elif estado_corpo == "CAPTURANDO":
                gesto.append(normalizar_frame(pts)); marc_gesto.append(sobr_ativo)
                c_fim = c_fim + 1 if mov < LIMIAR_FIM else 0
                if c_fim >= FRAMES_FIM or len(gesto) >= MAX_FRAMES_GESTO:
                    palavra, conf = classificar_corpo(gesto)
                    interrog = (sum(marc_gesto) > len(marc_gesto) / 2) if marc_gesto else False
                    estado_corpo = "OCIOSO"; c_ini = 0; gesto = []; marc_gesto = []
                    if palavra and conf >= CONFIANCA_CORPO and palavra != CLASSE_NEUTRO:
                        now = time.time()
                        if now - t_corpo > COOLDOWN_CORPO:
                            ult_pal_corpo = str(palavra); ult_conf_corpo = conf; t_corpo = now
                            with data_lock:
                                camera_data["tokens"].append(str(palavra))
            letra_atual = ult_pal_corpo
            confianca = ult_conf_corpo

        # ── encode + publica ──
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with data_lock:
            # texto = tokens fechados (traduzidos) + palavra sendo digitada + "?" se sobrancelha
            frase_exibida = montar_exibicao(camera_data["tokens"], camera_data["palavra_atual"], sobr_ativo)
            camera_data["frase"] = frase_exibida
            camera_data.update({
                "hands_detected": hands_det, "letra_atual": letra_atual,
                "confianca": round(confianca, 2), "gesto_limpar_progress": gesto_prog,
                "marcador": bool(sobr_ativo), "sobr_val": round(sobr_val or 0.0, 2),
                "frase_exibida": frase_exibida,
                "frame": base64.b64encode(buf).decode("utf-8"), "timestamp": time.time(),
            })

# ============================================================================
# WEBSOCKET
# ============================================================================
async def send_data(websocket):
    print("🔌 Cliente conectado!")
    async def receive():
        async for message in websocket:
            try:
                cmd = json.loads(message); act = cmd.get("action")
                with data_lock:
                    if act == "modo":
                        novo = cmd.get("modo", "alfabeto")
                        if novo in ("alfabeto", "corpo"):
                            # fecha a palavra que estava sendo digitada (vira token) e MANTEM o resto
                            if camera_data["palavra_atual"].strip():
                                camera_data["tokens"].append(camera_data["palavra_atual"].strip())
                            camera_data["palavra_atual"] = ""
                            camera_data["modo_app"] = novo
                            camera_data["letra_atual"] = "-"
                            print(f"➡ Modo: {novo}")
                    elif act == "desenho":
                        camera_data["desenho_ativo"] = bool(cmd.get("ativo", True))
                        print(f"➡ Linhas de detecção: {'ON' if camera_data['desenho_ativo'] else 'OFF'}")
                    elif act == "limpar":
                        render = montar_exibicao(camera_data["tokens"], camera_data["palavra_atual"], False)
                        if render.strip():
                            camera_data["historico"].insert(0, render)
                            camera_data["historico"] = camera_data["historico"][:15]
                        camera_data["tokens"] = []; camera_data["palavra_atual"] = ""
                    elif act == "espaco":
                        # fecha a palavra atual como token
                        if camera_data["palavra_atual"].strip():
                            camera_data["tokens"].append(camera_data["palavra_atual"].strip())
                            camera_data["palavra_atual"] = ""
                    elif act == "apagar":
                        if camera_data["palavra_atual"]:
                            camera_data["palavra_atual"] = camera_data["palavra_atual"][:-1]
                        elif camera_data["tokens"]:
                            camera_data["tokens"].pop()
                    elif act == "limpar_historico":
                        camera_data["historico"] = []
                    elif act == "remover_item":
                        i = cmd.get("index", -1)
                        if 0 <= i < len(camera_data["historico"]):
                            camera_data["historico"].pop(i)
            except (json.JSONDecodeError, KeyError):
                pass
    async def send():
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
    print("🚀 WebSocket na porta 8000...")
    async with websockets.serve(send_data, "localhost", 8000):
        print("✅ Servidor pronto! (modo inicial: alfabeto)")
        await asyncio.Future()

if __name__ == "__main__":
    print("=" * 50)
    print("  VisuAll UNIFICADO — alfabeto + corpo + rosto")
    print("=" * 50)
    Thread(target=capture_thread, daemon=True).start()
    Thread(target=process_thread, daemon=True).start()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Encerrado")
