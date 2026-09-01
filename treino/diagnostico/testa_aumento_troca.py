"""O aumento por troca de maos ajuda ou atrapalha? (avaliacao honesta)

Contexto: o rotulo de mao do MediaPipe nao e confiavel (concorda com o lado da
tela em 73% dos quadros do treino, muda dentro do mesmo clipe, e difere entre a
API do treino e a do app). A ideia era ensinar o modelo a nao se importar com
qual slot recebe qual mao, duplicando cada amostra com os slots trocados.

No aparelho isso PIOROU: PESSOA passou a sair COMPUTADOR e SURDO quase nao
saia. O teste que fiz antes nao podia ter pego isso -- era sobre os proprios
clipes de treino, saturado em 100%.

Aqui a separacao e por CLIPE (o clipe de teste nunca foi visto), e o teste roda
nas duas ordens de mao:
    app          -> slots como o treino monta
    app trocado  -> slots invertidos, que e o risco real no aparelho
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from experimento import (CACHE, CONFIANCA, monta_treino, monta_teste,  # noqa: E402
                         treina)

N_POSE, N_MAO = 33, 21
A, B, C = N_POSE * 3, (N_POSE + N_MAO) * 3, (N_POSE + 2 * N_MAO) * 3


def troca(X):
    fora = X.copy()
    fora[..., A:B] = X[..., B:C]
    fora[..., B:C] = X[..., A:B]
    return fora


def avalia(cfg, seqs, rotulos, classes, aumenta, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    rs = np.random.RandomState(0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    modos = ("app", "app trocado")
    conf = {m: np.zeros((len(classes), len(classes)), int) for m in modos}
    aceito = {m: np.zeros(len(classes), int) for m in modos}

    for tr, te in skf.split(np.zeros(len(seqs)), rotulos):
        X_tr, y_tr = monta_treino(seqs, rotulos, tr, cfg, rs)
        if aumenta:
            X_tr = np.concatenate([X_tr, troca(X_tr)])
            y_tr = np.concatenate([y_tr, y_tr])
        y_idx = np.array([classes.index(g) for g in y_tr])
        m = treina(X_tr, y_idx, len(classes), cfg, 42)
        X_te, y_te = monta_teste(seqs, rotulos, te, "app")
        if not len(X_te):
            continue
        for modo in modos:
            entrada = X_te if modo == "app" else troca(X_te)
            probs = m.predict(entrada, verbose=0)
            for verdadeiro, p in zip(y_te, probs):
                iv, previsto = classes.index(verdadeiro), int(p.argmax())
                conf[modo][iv, previsto] += 1
                if previsto == iv and p[previsto] >= CONFIANCA:
                    aceito[modo][iv] += 1
    return conf, aceito


def mostra(conf, aceito, classes):
    for modo in ("app", "app trocado"):
        c, a = conf[modo], aceito[modo]
        total = c.sum()
        print("  [%-11s] acerto %5.1f%%  aceito %5.1f%%" % (
            modo, np.trace(c) / total * 100, a.sum() / total * 100))
        for i, cl in enumerate(classes):
            n = c[i].sum()
            if not n:
                continue
            pior = int(np.argmax([c[i, j] if j != i else -1 for j in range(len(classes))]))
            nota = ("  (mais confundido com %s: %d)" % (classes[pior], c[i, pior])) \
                if c[i, pior] else ""
            print("      %-11s acerto %5.1f%%  aceito %5.1f%%%s" % (
                cl, c[i, i] / n * 100, a[i] / n * 100, nota))


def main():
    d = np.load(CACHE, allow_pickle=True)
    seqs, rotulos = d["seqs"], d["rotulos"]
    classes = sorted(set(rotulos))
    cfg = {"recortes": [(0.0, 1.0), (0.0, 0.9), (0.1, 1.0), (0.05, 0.95),
                        (0.0, 0.8), (0.2, 1.0), (0.1, 0.9)],
           "usa_segmento_app": True, "pesos_classe": True,
           "ruido": 0.02, "ruido_copias": 2,
           "bidirecional": True, "unidades": 96, "epocas": 80}

    for nome, aumenta in (("E (o que esta no app)", False),
                          ("E + troca de maos", True)):
        print("\n=== %s ===" % nome, flush=True)
        conf, aceito = avalia(cfg, seqs, rotulos, classes, aumenta)
        mostra(conf, aceito, classes)


if __name__ == "__main__":
    sys.exit(main())
