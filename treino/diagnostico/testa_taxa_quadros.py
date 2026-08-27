"""O app roda a ~13 fps; os videos de treino tem 30. Isso quebra a segmentacao?

Tudo na maquina de captura e contado em QUADROS, e quadro nao e uma unidade de
tempo fixa. Medido no aparelho (diag4.log): 60 quadros levaram 4,7 s, ou seja
~13 fps. Nos videos, 60 quadros sao 2,0 s.

Consequencias esperadas:
  - a janela de movimento cobre 2,3x mais tempo -> movimento medido maior
  - o teto de 60 quadros deixa a captura correr 2,3x mais tempo -> sinais
    colados um no outro, janela diluida, confianca despencando

Compara 3 mundos, com o modelo que esta no APK:
  30fps        -> como o treino ve (referencia)
  13fps atual  -> o app de hoje, se a hipotese estiver certa deve desabar
  13fps ajust. -> limiares convertidos pro tempo real do aparelho
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from experimento import CACHE, movimento, tem_mao  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "treino"))
import treinar_corpo as tc  # noqa: E402

ATIVO = REPO / "mobile/app/src/main/assets/gestos/geral"
FPS_TREINO = 30.0
CONFIANCA = 0.85


def reamostra_taxa(quadros, fps):
    """Simula a camera do aparelho: pega ~fps quadros por segundo do clipe."""
    passo = FPS_TREINO / fps
    idx = np.arange(0, len(quadros), passo).astype(int)
    return [quadros[i] for i in idx if i < len(quadros)]


def segmenta(quadros, mov_window, start, end, start_f, end_f, maxf, escala):
    buffer, capturando, gesto = [], False, []
    ini_cnt = fim_cnt = 0
    for frame in quadros:
        buffer.append(frame)
        if len(buffer) > mov_window:
            buffer.pop(0)
        mov = movimento(buffer) * escala
        if not capturando:
            if tem_mao(frame) and len(buffer) >= mov_window and mov > start:
                ini_cnt += 1
                if ini_cnt >= start_f:
                    capturando, gesto, fim_cnt = True, list(buffer), 0
            else:
                ini_cnt = 0
        else:
            gesto.append(frame)
            fim_cnt = fim_cnt + 1 if mov < end else 0
            if fim_cnt >= end_f or len(gesto) >= maxf:
                return (gesto, "teto" if len(gesto) >= maxf else "parada") \
                    if len(gesto) >= 10 else (None, "curto")
    return (gesto, "clipe") if capturando and len(gesto) >= 10 else (None, "curto")


def main():
    rot = [l.strip() for l in (ATIVO / "labels.txt").read_text().splitlines() if l.strip()]
    interp = tf.lite.Interpreter(model_path=str(ATIVO / "model.tflite"))
    interp.allocate_tensors()
    ent, sai = interp.get_input_details()[0], interp.get_output_details()[0]

    d = np.load(CACHE, allow_pickle=True)
    seqs, rotulos = d["seqs"], d["rotulos"]

    fps_app = 13.0
    f = fps_app / FPS_TREINO      # 0.433
    mundos = [
        ("30fps referencia", 30.0, dict(mov_window=5, start=0.050, end=0.030,
                                        start_f=3, end_f=5, maxf=60, escala=1.0)),
        ("13fps app hoje", fps_app, dict(mov_window=5, start=0.050, end=0.030,
                                         start_f=3, end_f=5, maxf=60, escala=1.0)),
        ("13fps ajustado", fps_app, dict(mov_window=5, start=0.050, end=0.030,
                                         start_f=2, end_f=2,
                                         maxf=max(10, int(round(60 * f))),
                                         escala=f)),
    ]

    for nome, fps, cfg in mundos:
        certos = aceitos = total = tetos = 0
        confs = []
        por_gesto = {}
        for seq, verdadeiro in zip(seqs, rotulos):
            seg, motivo = segmenta(reamostra_taxa(list(seq), fps), **cfg)
            if seg is None:
                continue
            x = np.array(tc.reamostra(seg), dtype=np.float32)[None, ...]
            interp.set_tensor(ent["index"], x)
            interp.invoke()
            p = interp.get_tensor(sai["index"])[0]
            palavra, conf = rot[int(p.argmax())], float(p.max())
            total += 1
            tetos += motivo == "teto"
            confs.append(conf)
            ok = palavra == verdadeiro
            certos += ok
            aceitos += ok and conf >= CONFIANCA
            g = por_gesto.setdefault(verdadeiro, [0, 0])
            g[0] += ok and conf >= CONFIANCA
            g[1] += 1
        print("\n=== %s ===" % nome)
        print("  segmentos %d | rotulo certo %.0f%% | ACEITO (>=0.85) %.0f%% "
              "| conf mediana %.3f | teto %.0f%%" % (
                  total, 100.0 * certos / total, 100.0 * aceitos / total,
                  float(np.median(confs)), 100.0 * tetos / total))
        for g in sorted(por_gesto):
            a, n = por_gesto[g]
            print("      %-11s aceito %2d/%-3d (%.0f%%)" % (g, a, n, 100.0 * a / n))


if __name__ == "__main__":
    sys.exit(main())
