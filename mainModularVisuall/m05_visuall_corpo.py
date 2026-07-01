import numpy as np

from m01_visuall_config import JANELA_MODELO, MIN_FRAMES_GESTO, N_FEATURES, N_HAND, N_POSE
from m02_visuall_modelos import classes_corpo, modelo_corpo


def extrair_pontos(res):
    pose = np.zeros((N_POSE, 3))
    me = np.zeros((N_HAND, 3))
    md = np.zeros((N_HAND, 3))
    tem = False
    if res.pose_landmarks:
        pose = np.array([[l.x, l.y, l.z] for l in res.pose_landmarks.landmark])
    if res.left_hand_landmarks:
        me = np.array([[l.x, l.y, l.z] for l in res.left_hand_landmarks.landmark])
        tem = True
    if res.right_hand_landmarks:
        md = np.array([[l.x, l.y, l.z] for l in res.right_hand_landmarks.landmark])
        tem = True
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
