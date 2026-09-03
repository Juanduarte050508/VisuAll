"""O app espelha a camera frontal. O treino nao espelha. Isso muda o resultado?

O app roda prepararBitmap com postScale(-1,1) na camera frontal, por causa de um
comentario que fala do dataset ANTIGO ("gravado com a webcam espelhada"). Mas o
gravar.py salva o quadro CRU ("preview espelhado; o que e salvo e o quadro cru")
e o treinar_corpo.py nao espelha nada.

Se isso importa, o efeito e sistematico: o MediaPipe rotula a mao por
"Left"/"Right", e espelhar troca os rotulos -- as duas maos trocam de slot no
vetor de 225. Este script mede, em vez de supor: extrai os mesmos clipes com e
sem flip e roda o model.tflite que esta no app agora.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

COMPUTER = Path(__file__).resolve().parents[2]
REPO = COMPUTER.parent
sys.path.insert(0, str(COMPUTER / "treino"))
import treinar_corpo as tc  # noqa: E402
import mediapipe as mp      # noqa: E402

POR_CLASSE = 6


def extrai(caminho, pose, hands, espelhar):
    cap = cv2.VideoCapture(str(caminho))
    quadros = []
    while True:
        ok, imagem = cap.read()
        if not ok:
            break
        imagem = cv2.resize(imagem, tc.TAMANHO_4_3)
        if espelhar:
            imagem = cv2.flip(imagem, 1)   # o que o app faz na camera frontal
        rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        r_pose = pose.process(rgb)
        r_maos = hands.process(rgb)

        frame = np.zeros(tc.N_FEATURES, dtype=np.float32)
        tem_pose = bool(r_pose.pose_landmarks)
        if tem_pose:
            for i, lm in enumerate(r_pose.pose_landmarks.landmark[:tc.N_POSE]):
                frame[i * 3:i * 3 + 3] = (lm.x, lm.y, lm.z)
        if r_maos.multi_hand_landmarks:
            for idx, marcas in enumerate(r_maos.multi_hand_landmarks):
                rotulo = ""
                if r_maos.multi_handedness and idx < len(r_maos.multi_handedness):
                    rotulo = r_maos.multi_handedness[idx].classification[0].label
                if rotulo.lower() == "left":
                    desloc = tc.N_POSE
                elif rotulo.lower() == "right":
                    desloc = tc.N_POSE + tc.N_MAO
                else:
                    media_x = float(np.mean([p.x for p in marcas.landmark]))
                    desloc = tc.N_POSE if media_x < 0.5 else tc.N_POSE + tc.N_MAO
                for i, lm in enumerate(marcas.landmark[:tc.N_MAO]):
                    b = (desloc + i) * 3
                    frame[b:b + 3] = (lm.x, lm.y, lm.z)
        if tem_pose:
            quadros.append(tc.normaliza_corpo(frame))
    cap.release()
    return quadros


def main():
    import tensorflow as tf

    labels = [l.strip() for l in (tc.ASSETS / "labels.txt").read_text().splitlines() if l.strip()]
    interp = tf.lite.Interpreter(model_path=str(tc.ASSETS / "model.tflite"))
    interp.allocate_tensors()
    ent = interp.get_input_details()[0]
    sai = interp.get_output_details()[0]

    def preve(quadros):
        x = np.array([tc.reamostra(quadros)], dtype=np.float32)
        interp.set_tensor(ent["index"], x)
        interp.invoke()
        p = interp.get_tensor(sai["index"])[0]
        i = int(p.argmax())
        return labels[i], float(p[i])

    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=0,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)

    gestos = sorted([p.name for p in tc.DATA.iterdir() if p.is_dir()])
    acerto = {"normal": 0, "espelhado": 0}
    total = 0
    virou = {}

    for gesto in gestos:
        clipes = sorted(tc.DATA.glob(gesto + "/*.mp4"))[:POR_CLASSE]
        for c in clipes:
            qn = extrai(c, pose, hands, False)
            qe = extrai(c, pose, hands, True)
            if len(qn) < 10 or len(qe) < 10:
                continue
            pn, cn = preve(qn)
            pe, ce = preve(qe)
            total += 1
            if pn == gesto:
                acerto["normal"] += 1
            if pe == gesto:
                acerto["espelhado"] += 1
            else:
                virou.setdefault((gesto, pe), []).append(ce)
            print("  %-11s %-11s %.2f  |  espelhado -> %-11s %.2f  %s"
                  % (gesto, pn, cn, pe, ce, "" if pe == gesto else "<-- ERRO"))
    pose.close()
    hands.close()

    print("\n=== %d clipes ===" % total)
    print("  SEM espelhar (como o treino ve):  %d/%d  = %.0f%%"
          % (acerto["normal"], total, 100.0 * acerto["normal"] / max(total, 1)))
    print("  ESPELHADO (como o app ve):        %d/%d  = %.0f%%"
          % (acerto["espelhado"], total, 100.0 * acerto["espelhado"] / max(total, 1)))
    if virou:
        print("\nErros com a imagem espelhada (confianca media):")
        for (v, p), confs in sorted(virou.items(), key=lambda kv: -len(kv[1])):
            print("  %-11s virou %-11s %dx  conf media %.2f"
                  % (v, p, len(confs), sum(confs) / len(confs)))


if __name__ == "__main__":
    sys.exit(main())
