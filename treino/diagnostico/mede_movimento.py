"""O movimento de cada gesto passa dos limiares que o app usa?

O app so COMECA a capturar quando bodyMotion() > BODY_START_MOTION (0.050) por
BODY_START_FRAMES quadros, e considera o gesto encerrado quando cai abaixo de
BODY_END_MOTION (0.030). Se um sinal inteiro fica abaixo desses valores, ele
nunca chega ao classificador -- e o sintoma no aparelho e "esse gesto nao
funciona", indistinguivel de erro do modelo.

Reproduz bodyMotion() exatamente: desvio padrao, sobre uma janela de 5 quadros,
das coordenadas x,y dos 42 pontos das MAOS (pose fica de fora), ja normalizados.
"""
import sys
from pathlib import Path

import numpy as np

CACHE = Path(__file__).parent / "cache_corpo.npz"
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "treino"))

N_POSE = 33
N_TOTAL = 75
JANELA_MOV = 5
START = 0.050
END = 0.030
START_FRAMES = 3


def movimento_por_quadro(seq):
    """bodyMotion() do app, avaliado em cada posicao da sequencia."""
    vals = []
    for i in range(len(seq)):
        janela = seq[max(0, i - JANELA_MOV + 1):i + 1]
        if len(janela) < 3:
            vals.append(0.0)
            continue
        total, n = 0.0, 0
        for ponto in range(N_POSE, N_TOTAL):
            for coord in (0, 1):
                total += float(np.std([q[ponto * 3 + coord] for q in janela]))
                n += 1
        vals.append(total / n if n else 0.0)
    return np.array(vals)


def main():
    d = np.load(CACHE, allow_pickle=True)
    seqs, rotulos = d["seqs"], d["rotulos"]

    print("%-12s %8s %8s %8s | %s | %s" %
          ("gesto", "pico", "mediana", "p90", "passa de 0.050?", "quadros acima de 0.030"))
    print("-" * 96)

    for gesto in sorted(set(rotulos)):
        idx = [i for i, r in enumerate(rotulos) if r == gesto]
        picos, medianas, p90s, dispara, acima = [], [], [], 0, []
        for i in idx:
            mv = movimento_por_quadro(list(seqs[i]))
            if len(mv) == 0:
                continue
            picos.append(mv.max())
            medianas.append(np.median(mv))
            p90s.append(np.percentile(mv, 90))
            # o app exige START_FRAMES quadros seguidos acima do limiar
            seguidos, maior = 0, 0
            for v in mv:
                seguidos = seguidos + 1 if v > START else 0
                maior = max(maior, seguidos)
            if maior >= START_FRAMES:
                dispara += 1
            acima.append(100.0 * float((mv > END).mean()))
        print("%-12s %8.4f %8.4f %8.4f | %2d de %2d clipes  | %.0f%% dos quadros"
              % (gesto, np.mean(picos), np.mean(medianas), np.mean(p90s),
                 dispara, len(idx), np.mean(acima)))

    print("\nLimiares do app: inicia captura acima de %.3f (por %d quadros seguidos),"
          % (START, START_FRAMES))
    print("                 considera parado abaixo de %.3f" % END)


if __name__ == "__main__":
    sys.exit(main())
