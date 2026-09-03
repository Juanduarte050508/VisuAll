"""Avaliacao honesta: separa TREINO e TESTE por CLIPE, nunca por recorte.

O treinar_corpo.py gera 3 recortes do mesmo clipe e depois embaralha tudo no
train_test_split -> pedacos do mesmo video caem nos dois lados e o acerto sai
inflado. Aqui o clipe inteiro fica de um lado so.
"""
import sys
from pathlib import Path

import numpy as np

CACHE = Path(__file__).parent / "cache_corpo.npz"
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "treino"))
import treinar_corpo as tc  # noqa: E402

RECORTES = [(0.0, 1.0), (0.0, 0.9), (0.1, 1.0)]


def monta(seqs, rotulos, indices, recortes):
    X, y = [], []
    for i in indices:
        quadros = seqs[i]
        for inicio, fim in recortes:
            a = int(len(quadros) * inicio)
            b = int(len(quadros) * fim)
            if b - a >= 10:
                X.append(np.array(tc.reamostra(list(quadros[a:b])), dtype=np.float32))
                y.append(rotulos[i])
    return np.array(X, dtype=np.float32), np.array(y)


def treina(X_tr, y_tr_idx, n_classes, epocas=60, semente=42):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(semente)
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(tc.JANELA, tc.N_FEATURES)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    m.fit(X_tr, y_tr_idx, epochs=epocas, batch_size=16, verbose=0)
    return m


def main():
    d = np.load(CACHE, allow_pickle=True)
    seqs, rotulos = d["seqs"], d["rotulos"]
    classes = sorted(set(rotulos))
    print("clipes: %d | classes: %s\n" % (len(seqs), " ".join(classes)))

    from sklearn.model_selection import StratifiedKFold

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # confusao acumulada sobre os clipes de teste de cada dobra
    conf = np.zeros((len(classes), len(classes)), dtype=int)
    # confianca do app: so conta acima de 0.85 (BODY_CONFIDENCE)
    LIMIAR = 0.85
    abaixo = np.zeros(len(classes), dtype=int)

    for dobra, (tr, te) in enumerate(skf.split(np.zeros(len(seqs)), rotulos), 1):
        X_tr, y_tr = monta(seqs, rotulos, tr, RECORTES)
        # teste: SO o clipe inteiro, que e o que o app ve de verdade
        X_te, y_te = monta(seqs, rotulos, te, [(0.0, 1.0)])
        y_tr_idx = np.array([classes.index(g) for g in y_tr])
        m = treina(X_tr, y_tr_idx, len(classes))
        probs = m.predict(X_te, verbose=0)
        pred = probs.argmax(axis=1)
        for verdadeiro, previsto, p in zip(y_te, pred, probs):
            iv = classes.index(verdadeiro)
            conf[iv, previsto] += 1
            if p[previsto] < LIMIAR:
                abaixo[iv] += 1
        acerto = float((np.array([classes.index(g) for g in y_te]) == pred).mean())
        print("  dobra %d: %d clipes de teste, acerto %.1f%%"
              % (dobra, len(y_te), acerto * 100))

    total = conf.sum()
    certos = np.trace(conf)
    print("\n=== ACERTO REAL (clipe nunca visto): %.1f%%  (%d/%d) ==="
          % (certos / total * 100, certos, total))

    print("\nMatriz de confusao (linha = verdadeiro, coluna = previsto):")
    print("%-12s %s   | acerto | abaixo de %.2f"
          % ("", " ".join("%-5s" % c[:5] for c in classes), LIMIAR))
    for i, c in enumerate(classes):
        linha = conf[i]
        acc = linha[i] / linha.sum() * 100 if linha.sum() else 0
        print("%-12s %s   | %5.1f%% | %d/%d"
              % (c, " ".join("%-5d" % v for v in linha), acc,
                 abaixo[i], linha.sum()))

    print("\nPrincipais confusoes:")
    pares = [(conf[i, j], classes[i], classes[j])
             for i in range(len(classes)) for j in range(len(classes)) if i != j]
    for n, a, b in sorted(pares, reverse=True)[:8]:
        if n:
            print("  %-12s virou %-12s %d vezes" % (a, b, n))


if __name__ == "__main__":
    sys.exit(main())

