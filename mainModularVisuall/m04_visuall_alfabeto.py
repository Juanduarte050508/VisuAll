import numpy as np


def normalize_landmarks(pontos):
    bx, by = pontos[0]
    norm = []
    for x, y in pontos:
        norm += [x - bx, y - by]
    mv = max(abs(v) for v in norm) or 1.0
    return [v / mv for v in norm]


def detectar_dedos_esticados(lms):
    M = 0.06
    ind = lms[8][1] < lms[5][1] - M
    med = lms[12][1] < lms[9][1] - M
    ane = lms[16][1] < lms[13][1] - M
    mind = lms[20][1] < lms[17][1] - M
    pol = abs(lms[4][0] - lms[0][0]) > 0.12
    return ind and med and ane and mind and pol


def calcular_movimento(buffer):
    if len(buffer) < 5:
        return 0.0
    rec = list(buffer)[-5:]
    try:
        return (
            np.std([f[2] for f in rec])
            + np.std([f[3] for f in rec])
            + np.std([f[16] for f in rec])
            + np.std([f[17] for f in rec])
        )
    except IndexError:
        return 0.0
