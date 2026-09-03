"""Quanto do gesto o app realmente captura, e qual limiar de parada preserva ele?

A maquina de estado corta a captura apos BODY_END_FRAMES quadros seguidos com
movimento abaixo de BODY_END_MOTION. Se o sinal tem trechos lentos no meio -- e
tem -- a captura termina antes do fim e o modelo recebe um fragmento.

Mede, pra varios limiares, que fracao do clipe seria capturada.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mede_movimento import movimento_por_quadro, CACHE  # noqa: E402

START = 0.050
START_FRAMES = 3
END_FRAMES = 5


def fracao_capturada(mv, end_motion, end_frames):
    """Reproduz a maquina de estado: devolve a fracao do clipe capturada."""
    seguidos = 0
    inicio = None
    for i, v in enumerate(mv):
        seguidos = seguidos + 1 if v > START else 0
        if seguidos >= START_FRAMES:
            inicio = i
            break
    if inicio is None:
        return 0.0
    baixos = 0
    for j in range(inicio, len(mv)):
        baixos = baixos + 1 if mv[j] < end_motion else 0
        if baixos >= end_frames:
            return (j - inicio) / max(len(mv) - inicio, 1)
    return 1.0


def main():
    d = np.load(CACHE, allow_pickle=True)
    seqs, rotulos = d["seqs"], d["rotulos"]
    cache_mv = {}
    for i in range(len(seqs)):
        cache_mv[i] = movimento_por_quadro(list(seqs[i]))

    combos = [(0.030, 5), (0.030, 10), (0.015, 5), (0.010, 5), (0.010, 10), (0.005, 8)]
    print("Fracao do gesto capturada (100%% = ate o fim do clipe)\n")
    print("%-12s %s" % ("gesto", "  ".join("%.3f/%dq" % c for c in combos)))
    print("-" * 78)
    for gesto in sorted(set(rotulos)):
        idx = [i for i, r in enumerate(rotulos) if r == gesto]
        linha = []
        for end_motion, end_frames in combos:
            fr = [fracao_capturada(cache_mv[i], end_motion, end_frames) for i in idx]
            linha.append("%6.0f%%  " % (100.0 * np.mean(fr)))
        print("%-12s %s" % (gesto, " ".join(linha)))
    print("\n(atual do app = 0.030/5q)")


if __name__ == "__main__":
    sys.exit(main())
