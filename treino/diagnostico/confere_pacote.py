"""Confere o .tflite que foi PRA DENTRO do APK, alimentado como o app alimenta.

Nao mede acerto: o modelo viu estes clipes no treino, entao o numero sai alto de
qualquer jeito. Serve pra pegar erro de encanamento -- recorte, ordem dos
rotulos, export -- que e o que quebrou da ultima vez: o modelo tinha sido
treinado com um limiar de corte e o app usava outro.
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from experimento import CACHE, segmento_do_app  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "treino"))
import treinar_corpo as tc  # noqa: E402

ATIVO = REPO / "mobile/app/src/main/assets/gestos/geral"


def main():
    rotulos_modelo = [l.strip() for l in (ATIVO / "labels.txt").read_text().splitlines() if l.strip()]
    interp = tf.lite.Interpreter(model_path=str(ATIVO / "model.tflite"))
    interp.allocate_tensors()
    ent, sai = interp.get_input_details()[0], interp.get_output_details()[0]

    d = np.load(CACHE, allow_pickle=True)
    seqs, rotulos = d["seqs"], d["rotulos"]

    conf = {v: {p: 0 for p in rotulos_modelo} for v in rotulos_modelo}
    sem_segmento = {v: 0 for v in rotulos_modelo}

    for seq, verdadeiro in zip(seqs, rotulos):
        seg = segmento_do_app(list(seq))
        if seg is None:
            sem_segmento[verdadeiro] += 1
            continue
        x = np.array(tc.reamostra(seg), dtype=np.float32)[None, ...]
        interp.set_tensor(ent["index"], x)
        interp.invoke()
        p = interp.get_tensor(sai["index"])[0]
        conf[verdadeiro][rotulos_modelo[int(p.argmax())]] += 1

    print("rotulos do pacote:", " ".join(rotulos_modelo))
    print("\nlinha = verdadeiro, coluna = previsto (segmento do app)\n")
    print("%-12s %s  | sem segmento" % ("", " ".join("%-5s" % r[:5] for r in rotulos_modelo)))
    for v in rotulos_modelo:
        linha = " ".join("%-5d" % conf[v][p] for p in rotulos_modelo)
        print("%-12s %s  | %d" % (v, linha, sem_segmento[v]))

    total = sum(sum(c.values()) for c in conf.values())
    certos = sum(conf[v][v] for v in rotulos_modelo)
    print("\nsegmentos classificados: %d   coerentes: %d (%.1f%%)" % (
        total, certos, 100.0 * certos / max(total, 1)))


if __name__ == "__main__":
    sys.exit(main())
