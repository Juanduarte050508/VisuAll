"""Por que o app so classifica quando o proximo sinal comeca.

A captura termina de dois jeitos: END_FRAMES quadros seguidos abaixo de
END_MOTION (o certo -- corta quando o sinal acaba) ou MAX_FRAMES (o teto de
seguranca -- corta 60 quadros depois, no meio do que vier).

Se o limiar de parada for baixo demais, quase tudo cai no teto: o modelo recebe
o fim de um sinal colado no comeco do proximo. Foi exatamente o relato no
aparelho: "a palavra so e capturada depois que eu inicio um novo ciclo dela".

ATENCAO: medir em clipe SUBESTIMA o problema. Um video acaba, e isso encerra a
captura de graca; ao vivo nada acaba. Por isso 0.015 passou na medicao anterior
e falhou no celular.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mede_movimento import movimento_por_quadro, CACHE  # noqa: E402

START, START_FRAMES, END_FRAMES, MAX_FRAMES = 0.050, 3, 5, 60


def como_parou(mv, end_motion):
    """'parada', 'teto' ou 'clipe' (o video acabou antes de qualquer corte)."""
    seguidos = 0
    inicio = None
    for i, v in enumerate(mv):
        seguidos = seguidos + 1 if v > START else 0
        if seguidos >= START_FRAMES:
            inicio = i
            break
    if inicio is None:
        return "nunca"
    baixos = 0
    for j in range(inicio, len(mv)):
        baixos = baixos + 1 if mv[j] < end_motion else 0
        if baixos >= END_FRAMES:
            return "parada"
        if (j - inicio) >= MAX_FRAMES:
            return "teto"
    return "clipe"


def main():
    d = np.load(CACHE, allow_pickle=True)
    seqs, rotulos = d["seqs"], d["rotulos"]
    mvs = [movimento_por_quadro(list(s)) for s in seqs]

    for end_motion in (0.030, 0.020, 0.015, 0.010):
        modos = [como_parou(mv, end_motion) for mv in mvs]
        n = len(modos)
        print("\n=== END_MOTION %.3f ===" % end_motion)
        print("  parada limpa: %3d/%d (%.0f%%)   teto/clipe: %d" % (
            modos.count("parada"), n, 100.0 * modos.count("parada") / n,
            modos.count("teto") + modos.count("clipe")))
        for gesto in sorted(set(rotulos)):
            idx = [i for i, r in enumerate(rotulos) if r == gesto]
            m = [modos[i] for i in idx]
            print("    %-11s parada %3d/%-3d  teto %2d  clipe %2d  nunca %2d" % (
                gesto, m.count("parada"), len(idx),
                m.count("teto"), m.count("clipe"), m.count("nunca")))


if __name__ == "__main__":
    sys.exit(main())
