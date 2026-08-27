"""Rodada 2: configuracoes centradas no trecho que o app captura, 2 sementes.

Mede o que importa no celular: acerto E confianca >= 0.85 avaliando sobre o
segmento que a maquina de estado do app extrai, com o clipe fora do treino.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from experimento import (  # noqa: E402
    CACHE, CONFIANCA, janela, segmento_do_app, treina, tc)


def monta_treino(seqs, rotulos, indices, cfg, rs):
    X, y = [], []
    for i in indices:
        quadros = seqs[i]
        for inicio, fim in cfg.get("recortes", []):
            v = janela(quadros, inicio, fim)
            if v is not None:
                X.append(v); y.append(rotulos[i])

        seg = segmento_do_app(quadros)
        if seg is not None:
            for inicio, fim in cfg.get("recortes_segmento", []):
                v = janela(seg, inicio, fim)
                if v is not None:
                    X.append(v); y.append(rotulos[i])

        if cfg.get("ruido_copias"):
            fontes = [q for q in (quadros, seg) if q is not None]
            for _ in range(cfg["ruido_copias"]):
                origem = fontes[rs.randint(len(fontes))]
                base = janela(origem, 0.0, 1.0)
                if base is not None:
                    X.append(base + rs.normal(0, cfg["ruido"], base.shape).astype(np.float32))
                    y.append(rotulos[i])
    return np.array(X, dtype=np.float32), np.array(y)


def monta_teste(seqs, rotulos, indices):
    X, y = [], []
    for i in indices:
        seg = segmento_do_app(seqs[i])
        if seg is not None:
            X.append(np.array(tc.reamostra(seg), dtype=np.float32))
            y.append(rotulos[i])
    return np.array(X, dtype=np.float32), np.array(y)


def avalia(cfg, seqs, rotulos, classes, sementes):
    from sklearn.model_selection import StratifiedKFold
    conf = np.zeros((len(classes), len(classes)), int)
    aceito = np.zeros(len(classes), int)
    for semente in sementes:
        rs = np.random.RandomState(semente)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in skf.split(np.zeros(len(seqs)), rotulos):
            X_tr, y_tr = monta_treino(seqs, rotulos, tr, cfg, rs)
            y_idx = np.array([classes.index(g) for g in y_tr])
            m = treina(X_tr, y_idx, len(classes), cfg, semente)
            X_te, y_te = monta_teste(seqs, rotulos, te)
            if not len(X_te):
                continue
            probs = m.predict(X_te, verbose=0)
            for verdadeiro, p in zip(y_te, probs):
                iv, previsto = classes.index(verdadeiro), int(p.argmax())
                conf[iv, previsto] += 1
                if previsto == iv and p[previsto] >= CONFIANCA:
                    aceito[iv] += 1
    return conf, aceito


def main():
    d = np.load(CACHE, allow_pickle=True)
    seqs, rotulos = d["seqs"], d["rotulos"]
    classes = sorted(set(rotulos))
    sementes = [42, 7]

    todos = [(0.0, 1.0), (0.0, 0.9), (0.1, 1.0), (0.05, 0.95), (0.0, 0.8),
             (0.2, 1.0), (0.1, 0.9)]
    seg3 = [(0.0, 1.0), (0.0, 0.9), (0.1, 1.0)]

    configs = {
        "C recortes+pesos": {"recortes": todos, "recortes_segmento": [(0.0, 1.0)],
                             "pesos_classe": True},
        "D C+ruido": {"recortes": todos, "recortes_segmento": [(0.0, 1.0)],
                      "pesos_classe": True, "ruido": 0.02, "ruido_copias": 2},
        "F centrado no app": {"recortes": [(0.0, 1.0), (0.0, 0.9), (0.1, 1.0)],
                              "recortes_segmento": seg3, "pesos_classe": True,
                              "ruido": 0.02, "ruido_copias": 2},
        "G so segmento": {"recortes": [], "recortes_segmento": seg3,
                          "pesos_classe": True, "ruido": 0.02, "ruido_copias": 2},
    }

    for nome, cfg in configs.items():
        conf, aceito = avalia(cfg, seqs, rotulos, classes, sementes)
        total = conf.sum()
        print("\n=== %s ===" % nome, flush=True)
        print("  acerto %.1f%%   aceito pelo app %.1f%%"
              % (np.trace(conf) / total * 100, aceito.sum() / total * 100))
        pior = 100.0
        for i, cl in enumerate(classes):
            n = conf[i].sum()
            if n:
                a = aceito[i] / n * 100
                pior = min(pior, a)
                print("      %-12s acerto %5.1f%%  aceito %5.1f%%"
                      % (cl, conf[i, i] / n * 100, a))
        print("  pior classe (aceito): %.1f%%" % pior)
        falsos = [(conf[:, j].sum() - conf[j, j], classes[j]) for j in range(len(classes))]
        ruins = [(n, c) for n, c in falsos if n]
        if ruins:
            print("  falsos positivos: " + ", ".join(
                "%s %d" % (c, n) for n, c in sorted(ruins, reverse=True)))


if __name__ == "__main__":
    sys.exit(main())
