import base64
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from m04_visuall_alfabeto import calcular_movimento, detectar_dedos_esticados, normalize_landmarks
from m01_visuall_config import (
    CLASSE_NEUTRO,
    CONFIANCA_CORPO,
    CONFIANCA_MINIMA,
    COOLDOWN_CORPO,
    FRAMES_FIM,
    FRAMES_INICIO,
    JANELA_MLP,
    JANELA_MOV,
    LIMIAR_FIM,
    LIMIAR_INICIO,
    LIMIAR_MOVIMENTO,
    MAX_FRAMES_GESTO,
    TEMPO_LIMPAR_CORPO,
    TEMPO_PRA_LIMPAR,
)
from m05_visuall_corpo import classificar_corpo, extrair_pontos, mov_inst, normalizar_frame
from m07_visuall_estado import camera_data, data_lock, raw_frame, raw_frame_lock
from m02_visuall_modelos import classes_estatico, classes_mlp, modelo_estatico, modelo_mlp
from m06_visuall_rosto import ler_marcador
from m03_visuall_traducao import montar_exibicao


def process_thread():
    holistic = mp.solutions.holistic.Holistic(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils
    mp_h = mp.solutions.holistic
    mp_hands = mp.solutions.hands

    ult_pred = ""
    estab = 0
    ult_add = ""
    t_add = 0.0
    t_inicio_estic = None
    t_ult_limpar = 0.0
    buf_lm = deque(maxlen=max(JANELA_MLP, 10))

    buf_mov = deque(maxlen=JANELA_MOV)
    gesto = []
    marc_gesto = []
    estado_corpo = "OCIOSO"
    c_ini = 0
    c_fim = 0
    ult_pal_corpo = "-"
    ult_conf_corpo = 0.0
    t_corpo = 0.0
    t_estic_corpo = None
    t_ult_limpar_corpo = 0.0
    last_ts = 0

    while True:
        with raw_frame_lock:
            frame = raw_frame["img"]
            ts = raw_frame["ts"]
        if frame is None or ts == last_ts:
            time.sleep(0.005)
            continue
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
        letra_atual = "-"
        confianca = 0.0
        hands_det = 0
        gesto_prog = 0.0

        if desenhar_linhas and res.face_landmarks:
            mp_draw.draw_landmarks(
                frame,
                res.face_landmarks,
                mp_h.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
            )

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
                            render = montar_exibicao(
                                camera_data["tokens"], camera_data["palavra_atual"], False
                            )
                            if render.strip():
                                camera_data["historico"].insert(0, render)
                                camera_data["historico"] = camera_data["historico"][:15]
                            camera_data["tokens"] = []
                            camera_data["palavra_atual"] = ""
                        ult_add = ""
                        t_inicio_estic = None
                        t_ult_limpar = now
                        gesto_prog = 0.0
                else:
                    t_inicio_estic = None

                if not esticados:
                    dados = normalize_landmarks(lms)
                    buf_lm.append(dados)
                    mov = calcular_movimento(buf_lm)
                    usa_mlp = (
                        mov > LIMIAR_MOVIMENTO
                        and modelo_mlp is not None
                        and len(buf_lm) >= JANELA_MLP
                    )
                    if usa_mlp:
                        janela = list(buf_lm)[-JANELA_MLP:]
                        probs = modelo_mlp.predict_proba(
                            np.array(janela).flatten().reshape(1, -1)
                        )[0]
                        idx = int(np.argmax(probs))
                        confianca = float(probs[idx])
                        letra_atual = classes_mlp[idx] if confianca >= CONFIANCA_MINIMA else "-"
                        estab_min, cooldown = 2, 0.3
                    elif modelo_estatico is not None:
                        probs = modelo_estatico.predict_proba(np.array(dados).reshape(1, -1))[0]
                        idx = int(np.argmax(probs))
                        confianca = float(probs[idx])
                        letra_atual = (
                            classes_estatico[idx] if confianca >= CONFIANCA_MINIMA else "-"
                        )
                        estab_min, cooldown = 12, 1.0
                    else:
                        estab_min, cooldown = 12, 1.0

                    if letra_atual != "-" and letra_atual == ult_pred:
                        estab += 1
                    else:
                        estab = 0
                    ult_pred = letra_atual
                    now = time.time()
                    if (
                        estab >= estab_min
                        and letra_atual != "-"
                        and letra_atual != ult_add
                        and (now - t_add) > cooldown
                    ):
                        with data_lock:
                            camera_data["palavra_atual"] += letra_atual
                        ult_add = letra_atual
                        t_add = now
                        estab = 0
                    if (now - t_add) > 1.0:
                        ult_add = ""
            else:
                ult_pred = ""
                estab = 0
                t_inicio_estic = None
                buf_lm.clear()

        else:
            pts, tem_mao = extrair_pontos(res)
            buf_mov.append(normalizar_frame(pts))
            mov = mov_inst(buf_mov)
            hands_det = 1 if tem_mao else 0

            if desenhar_linhas and res.pose_landmarks:
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_h.POSE_CONNECTIONS)
            if desenhar_linhas:
                for hlm in (res.left_hand_landmarks, res.right_hand_landmarks):
                    if hlm:
                        mp_draw.draw_landmarks(frame, hlm, mp_h.HAND_CONNECTIONS)

            hl_c = res.right_hand_landmarks or res.left_hand_landmarks
            if hl_c and detectar_dedos_esticados([[lm.x, lm.y] for lm in hl_c.landmark]):
                now = time.time()
                if t_estic_corpo is None:
                    t_estic_corpo = now
                seg = now - t_estic_corpo
                gesto_prog = min(1.0, seg / TEMPO_LIMPAR_CORPO)
                if seg >= TEMPO_LIMPAR_CORPO and (now - t_ult_limpar_corpo) > 2.0:
                    with data_lock:
                        render = montar_exibicao(
                            camera_data["tokens"], camera_data["palavra_atual"], False
                        )
                        if render.strip():
                            camera_data["historico"].insert(0, render)
                            camera_data["historico"] = camera_data["historico"][:15]
                        camera_data["tokens"] = []
                        camera_data["palavra_atual"] = ""
                    ult_pal_corpo = "-"
                    t_estic_corpo = None
                    t_ult_limpar_corpo = now
                    gesto_prog = 0.0
            else:
                t_estic_corpo = None

            if estado_corpo == "OCIOSO":
                if tem_mao and mov > LIMIAR_INICIO:
                    c_ini += 1
                    if c_ini >= FRAMES_INICIO:
                        estado_corpo = "CAPTURANDO"
                        gesto = list(buf_mov)
                        marc_gesto = [sobr_ativo]
                        c_fim = 0
                else:
                    c_ini = 0
            elif estado_corpo == "CAPTURANDO":
                gesto.append(normalizar_frame(pts))
                marc_gesto.append(sobr_ativo)
                c_fim = c_fim + 1 if mov < LIMIAR_FIM else 0
                if c_fim >= FRAMES_FIM or len(gesto) >= MAX_FRAMES_GESTO:
                    palavra, conf = classificar_corpo(gesto)
                    estado_corpo = "OCIOSO"
                    c_ini = 0
                    gesto = []
                    marc_gesto = []
                    if palavra and conf >= CONFIANCA_CORPO and palavra != CLASSE_NEUTRO:
                        now = time.time()
                        if now - t_corpo > COOLDOWN_CORPO:
                            ult_pal_corpo = str(palavra)
                            ult_conf_corpo = conf
                            t_corpo = now
                            with data_lock:
                                camera_data["tokens"].append(str(palavra))
            letra_atual = ult_pal_corpo
            confianca = ult_conf_corpo

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with data_lock:
            frase_exibida = montar_exibicao(
                camera_data["tokens"], camera_data["palavra_atual"], sobr_ativo
            )
            camera_data["frase"] = frase_exibida
            camera_data.update(
                {
                    "hands_detected": hands_det,
                    "letra_atual": letra_atual,
                    "confianca": round(confianca, 2),
                    "gesto_limpar_progress": gesto_prog,
                    "marcador": bool(sobr_ativo),
                    "sobr_val": round(sobr_val or 0.0, 2),
                    "frase_exibida": frase_exibida,
                    "frame": base64.b64encode(buf).decode("utf-8"),
                    "timestamp": time.time(),
                }
            )
