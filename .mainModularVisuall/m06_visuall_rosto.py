from collections import deque

import numpy as np

from m01_visuall_config import (
    IDX_BROW_L,
    IDX_BROW_R,
    IDX_EYE_OUT_L,
    IDX_EYE_OUT_R,
    IDX_EYE_TOP_L,
    IDX_EYE_TOP_R,
    JANELA_SOBR,
    LIMIAR_SOBRANCELHA,
)

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
