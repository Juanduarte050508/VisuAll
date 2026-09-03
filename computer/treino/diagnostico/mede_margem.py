"""Qual margem separa acerto de erro nos gestos corporais?

O app aceita um gesto so pela confianca, sem exigir distancia do 2o colocado --
diferente das letras, que exigem as duas coisas. O proprio comentario do
BODY_CONFIDENCE diz que o caminho, se voltar a reconhecer facil demais, e
"exigir margem como as letras fazem".

Este script nao escolhe a margem no chute: roda a mesma validacao honesta do
avalia.py (clipe nunca visto) e mostra, pra cada valor candidato, quantos ERROS
seriam barrados e quantos ACERTOS seriam perdidos junto.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from avalia import CACHE, RECORTES, monta, treina  # noqa: E402

CONFIANCA = 0.85


def main():
    d = np.load(CACHE, allow_pickle=True)
    seqs, rotulos = d["seqs"], d["rotulos"]
    classes = sorted(set(rotulos))

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    registros = []  # (certo?, confianca, margem, verdadeiro, previsto)
    for tr, te in skf.split(np.zeros(len(seqs)), rotulos):
        X_tr, y_tr = monta(seqs, rotulos, tr, RECORTES)
        X_te, y_te = monta(seqs, rotulos, te, [(0.0, 1.0)])
        m = treina(X_tr, np.array([classes.index(g) for g in y_tr]), len(classes))
        probs = m.predict(X_te, verbose=0)
        for verdadeiro, p in zip(y_te, probs):
            ordem = np.sort(p)[::-1]
            i = int(p.argmax())
            registros.append((classes[i] == verdadeiro, float(ordem[0]),
                              float(ordem[0] - ordem[1]), verdadeiro, classes[i]))

    certos = [r for r in registros if r[0]]
    errados = [r for r in registros if not r[0]]
    print("clipes avaliados: %d  (%d certos, %d errados)\n"
          % (len(registros), len(certos), len(errados)))

    def resumo(nome, grupo, idx):
        if not grupo:
            print("  %-8s (nenhum)" % nome)
            return
        v = sorted(r[idx] for r in grupo)
        print("  %-8s min %.3f | p25 %.3f | mediana %.3f | max %.3f"
              % (nome, v[0], v[len(v) // 4], v[len(v) // 2], v[-1]))

    print("CONFIANCA:")
    resumo("acertos", certos, 1)
    resumo("erros", errados, 1)
    print("\nMARGEM (1o - 2o):")
    resumo("acertos", certos, 2)
    resumo("erros", errados, 2)

    print("\nEfeito de exigir margem (mantendo confianca >= %.2f):" % CONFIANCA)
    print("  %-8s %-22s %-22s" % ("margem", "erros barrados", "acertos perdidos"))
    base_ok = [r for r in certos if r[1] >= CONFIANCA]
    base_err = [r for r in errados if r[1] >= CONFIANCA]
    for margem in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        barra = [r for r in base_err if r[2] < margem]
        perde = [r for r in base_ok if r[2] < margem]
        print("  %-8.2f %-22s %-22s"
              % (margem,
                 "%d de %d (%.0f%%)" % (len(barra), len(base_err),
                                        100.0 * len(barra) / max(len(base_err), 1)),
                 "%d de %d (%.0f%%)" % (len(perde), len(base_ok),
                                        100.0 * len(perde) / max(len(base_ok), 1))))

    print("\nErros que passariam de confianca (os que a margem precisa pegar):")
    for certo, conf, marg, v, prev in sorted(base_err, key=lambda r: -r[1])[:10]:
        print("  %-11s virou %-11s conf %.2f  margem %.2f" % (v, prev, conf, marg))


if __name__ == "__main__":
    sys.exit(main())
