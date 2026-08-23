"""
Treina e exporta o modelo de GESTOS CORPORAIS do app (.tflite).

Entrada:  data/raw_body_videos/<GESTO>/*.mp4   (gravados no modo "corpo")
Saida:    mobile/app/src/main/assets/gestos/geral/model.tflite
          mobile/app/src/main/assets/gestos/geral/labels.txt

CONTRATO com o app (BodyGestureEngine.kt) -- nada disto pode mudar:

  1. entrada .......... [1, 30, 225] float32
                        30 quadros x 225 numeros
  2. os 225 numeros ... 75 pontos x (x, y, z), nesta ordem exata:
                          pontos  0..32  -> corpo/pose   (33 pontos)
                          pontos 33..53  -> mao ESQUERDA (21 pontos)
                          pontos 54..74  -> mao DIREITA  (21 pontos)
                        (BodyGestureEngine.extractFrame + writeBodyPoint)
  3. saida ............ [1, n_gestos], uma probabilidade por linha do labels.txt
  4. Select TF Ops .... o LSTM nao tem kernel nativo no TFLite; o app registra
                        um FlexDelegate justamente pra isso, entao a conversao
                        PRECISA habilitar SELECT_TF_OPS.

Duas contas tem que bater exatamente com o Kotlin, senao o modelo aprende num
formato e e usado em outro -- sem dar erro, so errando mais:

  normaliza_corpo() <-> LibrasMath.normalizeBodyFrame
  reamostra()       <-> LibrasMath.resample / resampleIndex

Sobre a proporcao 4:3: o app multiplica o x por `aspectX = 0.75 * (largura/altura)`
"pra corrigir o x para a proporcao 4:3 do treino" (comentario em
LibrasAnalyzer.kt:329). Em 4:3 essa conta da exatamente 1.0 -- por isso aqui os
quadros sao redimensionados pra 4:3 e o x entra sem fator nenhum, que e o mesmo
que o pipeline das letras ja faz.

Uso:
    python treino/treinar_corpo.py
    python treino/treinar_corpo.py --epocas 80
"""
import argparse
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data" / "raw_body_videos"
ASSETS = RAIZ / "mobile" / "app" / "src" / "main" / "assets" / "gestos" / "geral"

N_POSE, N_MAO = 33, 21
N_PONTOS = N_POSE + 2 * N_MAO          # 75
N_FEATURES = N_PONTOS * 3              # 225
JANELA = 30
ESCALA_MINIMA_OMBROS = 0.0001          # LibrasMath.ESCALA_MINIMA_OMBROS
OMBRO_ESQ, OMBRO_DIR = 11, 12          # BODY_POINT_LEFT/RIGHT_SHOULDER
TAMANHO_4_3 = (320, 240)


def normaliza_corpo(frame):
    """Gemeo de LibrasMath.normalizeBodyFrame.

    Centraliza x,y no meio dos ombros e divide pela distancia 3D entre eles
    (o dz ENTRA na conta). O z fica cru, como sai do MediaPipe.
    """
    # float32 em TODA operacao, de proposito. O Kotlin faz esta conta inteira
    # em Float, entao e assim que o gemeo tem que calcular. E nao e so
    # fidelidade: com um float comum do Python no lugar (2.0, ou float(...)),
    # o resultado passa a depender da VERSAO do numpy -- no numpy 1.x um
    # escalar np.float32 dividido por float do Python sobe pra float64, e no
    # numpy 2 (NEP 50) fica em float32. As duas contas diferem por alguns ULP,
    # o bastante pra mudar a 6a casa decimal e quebrar o contrato em
    # tests/fixtures/landmark_contract.json dependendo da maquina.
    saida = np.asarray(frame, dtype=np.float32).copy()
    le, ld = OMBRO_ESQ * 3, OMBRO_DIR * 3
    dois = np.float32(2.0)
    centro_x = (saida[le] + saida[ld]) / dois
    centro_y = (saida[le + 1] + saida[ld + 1]) / dois
    dx = saida[le] - saida[ld]
    dy = saida[le + 1] - saida[ld + 1]
    dz = saida[le + 2] - saida[ld + 2]
    escala = np.float32(np.sqrt(dx * dx + dy * dy + dz * dz))
    # Pose degenerada (pessoa de lado, ombro fora do quadro) daria uma escala
    # quase zero e explodiria as features -- aí não normaliza.
    if escala <= ESCALA_MINIMA_OMBROS:
        escala = np.float32(1.0)
    for ponto in range(N_PONTOS):
        base = ponto * 3
        saida[base] = (saida[base] - centro_x) / escala
        saida[base + 1] = (saida[base + 1] - centro_y) / escala
    return saida


def reamostra(frames, quantidade=JANELA):
    """Gemeo de LibrasMath.resample: escolhe quadros por indice, nao interpola."""
    if len(frames) == quantidade:
        return list(frames)
    return [frames[int((len(frames) - 1) * i / (quantidade - 1))]
            for i in range(quantidade)]


def extrai_video(caminho, pose, hands):
    """Le um clipe e devolve a lista de quadros ja normalizados (225 cada)."""
    import cv2

    cap = cv2.VideoCapture(str(caminho))
    quadros = []
    while True:
        ok, imagem = cap.read()
        if not ok:
            break
        # 4:3 -> o fator aspectX do app vale 1.0 nesta proporcao (ver docstring)
        imagem = cv2.resize(imagem, TAMANHO_4_3)
        rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        r_pose = pose.process(rgb)
        r_maos = hands.process(rgb)

        frame = np.zeros(N_FEATURES, dtype=np.float32)
        tem_pose = bool(r_pose.pose_landmarks)
        if tem_pose:
            for i, lm in enumerate(r_pose.pose_landmarks.landmark[:N_POSE]):
                frame[i * 3:i * 3 + 3] = (lm.x, lm.y, lm.z)

        if r_maos.multi_hand_landmarks:
            for idx, marcas in enumerate(r_maos.multi_hand_landmarks):
                # Mesmo criterio do Kotlin: usa o rotulo do MediaPipe e, sem
                # ele, decide pelo lado da tela (avgX < 0.5 = esquerda).
                rotulo = ""
                if r_maos.multi_handedness and idx < len(r_maos.multi_handedness):
                    rotulo = r_maos.multi_handedness[idx].classification[0].label
                if rotulo.lower() == "left":
                    deslocamento = N_POSE
                elif rotulo.lower() == "right":
                    deslocamento = N_POSE + N_MAO
                else:
                    media_x = float(np.mean([p.x for p in marcas.landmark]))
                    deslocamento = N_POSE if media_x < 0.5 else N_POSE + N_MAO
                for i, lm in enumerate(marcas.landmark[:N_MAO]):
                    base = (deslocamento + i) * 3
                    frame[base:base + 3] = (lm.x, lm.y, lm.z)

        # Sem pose nao da pra normalizar (a escala vem dos ombros): descarta.
        if tem_pose:
            quadros.append(normaliza_corpo(frame))
    cap.release()
    return quadros


def monta_dataset(recortes):
    """Le todos os clipes e devolve X [N, 30, 225] e y [N]."""
    import mediapipe as mp

    gestos = sorted([p.name for p in DATA.iterdir()
                     if p.is_dir() and any(p.glob("*.mp4"))])
    if not gestos:
        print("Nenhum clipe em %s" % DATA)
        print("Grave no modo 'corpo' do Gravar.bat (TAB ate aparecer 'corpo').")
        return None, None, []

    X, y = [], []
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=0,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    maos = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                    model_complexity=0,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
    try:
        for gesto in gestos:
            clipes = sorted(DATA.joinpath(gesto).glob("*.mp4"))
            antes = len(X)
            for clipe in clipes:
                quadros = extrai_video(clipe, pose, maos)
                if len(quadros) < 10:      # BODY_MIN_FRAMES
                    print("    %s: só %d quadros com corpo visível — ignorado"
                          % (clipe.name, len(quadros)))
                    continue
                # O clipe inteiro é a amostra fiel. Os recortes são variação
                # temporal barata (mesmo gesto, começando/terminando um pouco
                # antes ou depois) — ajuda quando há poucos clipes.
                for inicio, fim in recortes:
                    a = int(len(quadros) * inicio)
                    b = int(len(quadros) * fim)
                    if b - a >= 10:
                        X.append(np.array(reamostra(quadros[a:b]), dtype=np.float32))
                        y.append(gesto)
            print("  %-12s %2d clipes -> %d amostras"
                  % (gesto, len(clipes), len(X) - antes))
    finally:
        pose.close()
        maos.close()

    if not X:
        return None, None, gestos
    return np.array(X, dtype=np.float32), np.array(y), gestos


def carrega_professor(caminho):
    """Abre o modelo que JA esta no app pra usar como professor.

    Devolve uma funcao que roda inferencia, ou None se o arquivo nao existir /
    nao for um modelo de verdade (ex.: ponteiro do Git LFS, de 132 bytes).
    """
    import tensorflow as tf

    caminho = Path(caminho)
    if not caminho.exists() or caminho.stat().st_size < 10000:
        return None
    try:
        it = tf.lite.Interpreter(model_path=str(caminho))
        entrada = it.get_input_details()[0]
        it.resize_tensor_input(entrada["index"], [1, JANELA, N_FEATURES])
        it.allocate_tensors()
        entrada = it.get_input_details()[0]
        saida = it.get_output_details()[0]
    except Exception as erro:
        print("  (não consegui abrir o modelo antigo: %s)" % erro)
        return None

    def prever(x):
        it.set_tensor(entrada["index"], x.astype(np.float32))
        it.invoke()
        return it.get_tensor(saida["index"])[0]

    return prever


def gera_ensaio(professor, labels_antigos, por_classe, confianca=0.80,
                max_tentativas=6000, semente=0):
    """Reconstroi exemplos do que o modelo ANTIGO sabe, sem ter os videos dele.

    O problema: treinar do zero com os seus videos faz o modelo esquecer os
    gestos que voce nao gravou -- e os videos originais nao existem mais.

    A saida: gerar muitas sequencias sinteticas, perguntar ao modelo antigo o
    que ele ve em cada uma, e guardar aquelas em que ele responde com
    confianca alta. Esses pares (sequencia -> resposta do modelo antigo) viram
    exemplos de treino ao lado dos seus videos: o modelo novo aprende os
    gestos novos E imita o antigo no resto. Chama-se destilacao.
    """
    rs = np.random.RandomState(semente)
    baldes = {rotulo: [] for rotulo in labels_antigos}
    alvo_total = por_classe * len(labels_antigos)

    for tentativa in range(max_tentativas):
        modo = tentativa % 4
        if modo == 0:                                  # ruído puro
            x = rs.normal(0, rs.uniform(0.05, 1.5), (1, JANELA, N_FEATURES))
        elif modo == 1:                                # movimento suave
            t = np.linspace(0, rs.uniform(1, 4) * np.pi, JANELA)[:, None]
            x = (np.sin(t + rs.uniform(0, 6.28, (1, N_FEATURES)))
                 * rs.uniform(0.1, 1.2))[None]
        elif modo == 2:                                # quase parado, com deriva
            base = rs.normal(0, 0.6, (1, 1, N_FEATURES))
            x = base + np.linspace(0, 1, JANELA)[None, :, None] * \
                rs.normal(0, 0.4, (1, 1, N_FEATURES))
        else:                                          # rampa + ruído
            x = np.linspace(-1, 1, JANELA)[None, :, None] * \
                rs.normal(0, 0.8, (1, 1, N_FEATURES)) + \
                rs.normal(0, 0.2, (1, JANELA, N_FEATURES))

        probs = professor(x)
        indice = int(probs.argmax())
        if probs[indice] < confianca or indice >= len(labels_antigos):
            continue
        rotulo = labels_antigos[indice]
        if len(baldes[rotulo]) < por_classe:
            baldes[rotulo].append(x[0].astype(np.float32))
        if sum(len(v) for v in baldes.values()) >= alvo_total:
            break

    X = [amostra for rotulo in labels_antigos for amostra in baldes[rotulo]]
    y = [rotulo for rotulo in labels_antigos for _ in baldes[rotulo]]
    return (np.array(X, dtype=np.float32) if X else None,
            np.array(y) if y else None,
            {rotulo: len(v) for rotulo, v in baldes.items()})


def exporta_tflite(modelo, destino):
    import tensorflow as tf

    conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
    # O LSTM não tem kernel nativo no TFLite. O app registra um FlexDelegate
    # exatamente por isso (BodyGestureEngine.ensureLoaded) — sem estas duas
    # linhas o modelo nem converte, ou converte e o app não consegue rodar.
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                      tf.lite.OpsSet.SELECT_TF_OPS]
    conv._experimental_lower_tensor_list_ops = False
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(conv.convert())
    return destino


def valida(caminho, n_gestos):
    """Roda o .tflite como o app roda, antes de você descobrir no celular."""
    import tensorflow as tf

    it = tf.lite.Interpreter(model_path=str(caminho))
    entrada = it.get_input_details()[0]
    it.resize_tensor_input(entrada["index"], [1, JANELA, N_FEATURES])
    it.allocate_tensors()
    entrada = it.get_input_details()[0]
    saida = it.get_output_details()[0]

    if list(entrada["shape"]) != [1, JANELA, N_FEATURES]:
        print("  FALHOU: entrada devia ser [1, %d, %d], veio %s"
              % (JANELA, N_FEATURES, list(entrada["shape"])))
        return False
    if list(saida["shape"]) != [1, n_gestos]:
        print("  FALHOU: saída devia ser [1, %d], veio %s"
              % (n_gestos, list(saida["shape"])))
        return False

    it.set_tensor(entrada["index"], np.zeros((1, JANELA, N_FEATURES), np.float32))
    it.invoke()
    probs = it.get_tensor(saida["index"])
    soma = float(probs.sum())
    if not (0.98 <= soma <= 1.02):
        print("  FALHOU: as probabilidades somam %.3f (deviam somar 1)." % soma)
        return False

    print("  validado: entrada %s, saída %s, soma=%.3f"
          % (list(entrada["shape"]), list(saida["shape"]), soma))
    return True


def main():
    ap = argparse.ArgumentParser(description="Treina o modelo de gestos corporais.")
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--do-zero", action="store_true",
                    help="treina SÓ com os seus vídeos, deixando o app esquecer "
                         "os gestos que você não gravou")
    ap.add_argument("--sem-recortes", action="store_true",
                    help="usa só o clipe inteiro, sem variação temporal")
    args = ap.parse_args()

    if not DATA.exists():
        print("Não existe %s — grave no modo 'corpo' do Gravar.bat primeiro." % DATA)
        return 1

    recortes = [(0.0, 1.0)] if args.sem_recortes else [(0.0, 1.0), (0.0, 0.9), (0.1, 1.0)]

    print("Lendo os clipes (isto demora: cada quadro passa por pose + 2 mãos)...")
    X, y, gestos = monta_dataset(recortes)
    if X is None:
        return 1

    labels_path = ASSETS / "labels.txt"
    atuais = []
    if labels_path.exists():
        atuais = [l.strip() for l in labels_path.read_text(encoding="utf-8").splitlines()
                  if l.strip()]

    faltando = sorted(set(atuais) - set(gestos))

    # ── Acrescentar ao modelo antigo em vez de substituí-lo ──────────────
    # Treinar só com os vídeos de hoje faria o app esquecer os gestos que você
    # não gravou. Como os vídeos originais não existem mais, o jeito de
    # preservá-los é perguntar ao próprio modelo que já está no app: ele vira
    # professor e "reconta" o que sabe, em exemplos que entram no treino ao
    # lado dos seus vídeos.
    if faltando and not args.do_zero:
        print("\n  Gestos que você não gravou: %s" % " ".join(faltando))
        print("  Vou preservá-los aprendendo com o modelo que já está no app,")
        print("  em vez de substituí-lo. Isso leva um minuto...")

        professor = carrega_professor(ASSETS / "model.tflite")
        if professor is None:
            print("\n  " + "!" * 58)
            print("  PAREI: não há modelo antigo utilizável pra preservar.")
            print("  " + "!" * 58)
            print("  O arquivo %s" % (ASSETS / "model.tflite").relative_to(RAIZ))
            print("  não existe ou é um ponteiro do Git LFS (132 bytes) em vez do")
            print("  modelo. Rode 'git lfs pull' no repo, ou grave também estes")
            print("  gestos: %s" % " ".join(faltando))
            print("  (ou use --do-zero pra treinar só com o que você gravou)")
            return 1

        # Quantidade parecida com a dos seus dados, pra nenhum lado dominar.
        por_classe = max(20, int(np.median([int((y == g).sum()) for g in gestos])))
        X_ensaio, y_ensaio, contagem = gera_ensaio(professor, atuais, por_classe)

        if X_ensaio is None or any(contagem.get(g, 0) < 5 for g in faltando):
            print("\n  Não consegui exemplos suficientes do modelo antigo:")
            for g in faltando:
                print("     %-12s %d exemplos" % (g, contagem.get(g, 0)))
            print("  Grave estes gestos, ou use --do-zero se aceitar perdê-los.")
            return 1

        # Só entram os gestos que você NÃO gravou: onde há vídeo real, o vídeo
        # real é melhor que a reconstrução.
        manter = np.isin(y_ensaio, faltando)
        X_ensaio, y_ensaio = X_ensaio[manter], y_ensaio[manter]
        print("  exemplos recuperados do modelo antigo:")
        for g in faltando:
            print("     %-12s %d" % (g, int((y_ensaio == g).sum())))

        X = np.concatenate([X, X_ensaio])
        y = np.concatenate([y, y_ensaio])
        gestos = sorted(set(gestos) | set(faltando))

    elif faltando and args.do_zero:
        print("\n  --do-zero: o app vai PERDER %s" % " ".join(faltando))

    classes = sorted(set(y))
    y_idx = np.array([classes.index(g) for g in y])
    print("\n  %d amostras | %d gestos: %s" % (len(X), len(classes), " ".join(classes)))
    for c in classes:
        print("     %-12s %d" % (c, int((y == c).sum())))

    if len(classes) < 2:
        print("\n  ERRO: precisa de pelo menos 2 gestos diferentes.")
        return 1

    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    minimo = min(int((y_idx == i).sum()) for i in range(len(classes)))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_idx, test_size=0.2, random_state=42,
        stratify=y_idx if minimo >= 5 else None)

    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(JANELA, N_FEATURES)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(classes), activation="softmax"),
    ])
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    print("\n  treinando LSTM (%d épocas)...\n" % args.epocas)
    modelo.fit(X_tr, y_tr, validation_data=(X_te, y_te),
               epochs=args.epocas, batch_size=16, verbose=2)

    perda, acerto = modelo.evaluate(X_te, y_te, verbose=0)
    print("\n  acerto no teste: %.1f%%" % (acerto * 100))

    # Guarda o modelo atual ANTES de sobrescrever. Treinar aqui NAO acrescenta
    # ao que ja existe: substitui por um que sabe so o que estava nos seus
    # videos. Se sair pior, RestaurarModelo.bat devolve o anterior.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from backup import faz_backup
    guardado = faz_backup([ASSETS / "model.tflite", labels_path], "gestos")
    if guardado:
        print("\n  modelo anterior guardado em: %s" % guardado.relative_to(RAIZ))

    destino = ASSETS / "model.tflite"
    exporta_tflite(modelo, destino)
    labels_path.write_text("\n".join(classes) + "\n", encoding="utf-8")
    print("  gerado: %s (%.0f KB)"
          % (destino.relative_to(RAIZ), destino.stat().st_size / 1024))
    print("  gerado: %s -> %s" % (labels_path.relative_to(RAIZ), " ".join(classes)))

    if not valida(destino, len(classes)):
        return 1

    print("\nPróximo passo: Android Studio -> pasta 'mobile' -> Run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
