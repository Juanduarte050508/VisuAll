"""O app ve as maos trocadas de lado em relacao ao treino?

O codigo que poe cada mao no seu slot e igual nos dois lados (rotulo do
MediaPipe, e avgX como desempate). Mas o rotulo depende da IMAGEM: se o app
manda o quadro espelhado e o video foi gravado sem espelho, a MESMA mao fisica
recebe rotulos opostos -- e as duas maos trocam de slot sem nenhum codigo estar
errado.

Num sinal de uma mao so isso e inofensivo. Num sinal de duas maos assimetrico
como o COMPUTADOR, e fatal.

Compara 4 leituras do MESMO clipe:
  normal   -> como o treino viu
  trocado  -> maos trocadas de slot
  espelho  -> x espelhado
  ambos    -> espelho de verdade (x espelhado E maos trocadas)
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from experimento import CACHE, segmento_do_app  # noqa: E402

COMPUTER = Path(__file__).resolve().parents[2]
REPO = COMPUTER.parent
sys.path.insert(0, str(COMPUTER / "treino"))
import treinar_corpo as tc  # noqa: E402

ATIVO = REPO / "mobile/app/src/main/assets/gestos/geral"
N_POSE, N_MAO = 33, 21
A, B = N_POSE, N_POSE + N_MAO          # inicio de cada slot de mao


def troca_maos(frame):
    f = frame.copy()
    f[A * 3:B * 3], f[B * 3:(B + N_MAO) * 3] = \
        frame[B * 3:(B + N_MAO) * 3].copy(), frame[A * 3:B * 3].copy()
    return f


def espelha(frame):
    """Espelho em quadro JA NORMALIZADO: x e centrado nos ombros, entao
    espelhar e NEGAR, nao fazer 1-x. Errei isso na primeira versao e as linhas
    de espelho sairam sem sentido."""
    f = frame.copy()
    f[0::3] = np.where(f[0::3] != 0, -f[0::3], 0.0)
    return f


def main():
    rot = [l.strip() for l in (ATIVO / "labels.txt").read_text().splitlines() if l.strip()]
    interp = tf.lite.Interpreter(model_path=str(ATIVO / "model.tflite"))
    interp.allocate_tensors()
    ent, sai = interp.get_input_details()[0], interp.get_output_details()[0]

    def preve(seg):
        x = np.array(tc.reamostra(seg), dtype=np.float32)[None, ...]
        interp.set_tensor(ent["index"], x)
        interp.invoke()
        p = interp.get_tensor(sai["index"])[0]
        return rot[int(p.argmax())], float(p.max())

    d = np.load(CACHE, allow_pickle=True)
    seqs, rotulos = d["seqs"], d["rotulos"]

    variantes = {
        "normal":  lambda f: f,
        "trocado": troca_maos,
        "espelho": espelha,
        "ambos":   lambda f: troca_maos(espelha(f)),
    }

    for gesto in sorted(set(rotulos)):
        idx = [i for i, r in enumerate(rotulos) if r == gesto]
        print("\n=== %s (%d clipes) ===" % (gesto, len(idx)))
        for nome, fn in variantes.items():
            saidas = {}
            for i in idx:
                seg = segmento_do_app(list(seqs[i]))
                if seg is None:
                    continue
                palavra, _ = preve([fn(np.asarray(f)) for f in seg])
                saidas[palavra] = saidas.get(palavra, 0) + 1
            total = sum(saidas.values())
            certo = saidas.get(gesto, 0)
            resumo = "  ".join("%s:%d" % (k, v) for k, v in
                               sorted(saidas.items(), key=lambda kv: -kv[1]))
            print("  %-8s %3d/%-3d  %s" % (nome, certo, total, resumo))


if __name__ == "__main__":
    sys.exit(main())
