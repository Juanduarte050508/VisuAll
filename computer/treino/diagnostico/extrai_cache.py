"""Extrai os landmarks de cada clipe UMA vez e guarda em cache.

Guarda a sequencia inteira por clipe (sem recortes), pra permitir avaliar
com separacao por clipe depois sem reprocessar video.
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve()
COMPUTER = Path(__file__).resolve().parents[2]
REPO = COMPUTER.parent
sys.path.insert(0, str(COMPUTER / "treino"))

import treinar_corpo as tc  # noqa: E402
import mediapipe as mp      # noqa: E402

SAIDA = Path(__file__).parent / "cache_corpo.npz"


def main():
    gestos = sorted([p.name for p in tc.DATA.iterdir()
                     if p.is_dir() and any(p.glob("*.mp4"))])

    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=0,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    maos = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                    model_complexity=0,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

    seqs, rotulos, nomes = [], [], []
    try:
        for gesto in gestos:
            clipes = sorted(tc.DATA.joinpath(gesto).glob("*.mp4"))
            ok = 0
            for clipe in clipes:
                quadros = tc.extrai_video(clipe, pose, maos)
                if len(quadros) < 10:
                    print("    %s: so %d quadros com corpo - ignorado"
                          % (clipe.name, len(quadros)), flush=True)
                    continue
                seqs.append(np.array(quadros, dtype=np.float32))
                rotulos.append(gesto)
                nomes.append("%s/%s" % (gesto, clipe.name))
                ok += 1
            print("  %-12s %2d/%d clipes usados" % (gesto, ok, len(clipes)), flush=True)
    finally:
        pose.close()
        maos.close()

    np.savez_compressed(
        SAIDA,
        seqs=np.array(seqs, dtype=object),
        rotulos=np.array(rotulos),
        nomes=np.array(nomes),
    )
    print("\nsalvo: %s  (%d clipes)" % (SAIDA, len(seqs)))


if __name__ == "__main__":
    sys.exit(main())

