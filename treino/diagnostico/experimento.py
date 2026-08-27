"""Compara configuracoes de treino com avaliacao honesta (separada por clipe).

Avalia de dois jeitos:
  clipe   -> o clipe inteiro reamostrado, que e o que o treino usa hoje
  app     -> o trecho que a maquina de estado do app REALMENTE captura
             (comeca no movimento, termina na parada) -- e o que vale no celular
"""
import sys
from pathlib import Path

import numpy as np

CACHE = Path(__file__).parent / "cache_corpo.npz"
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "treino"))
import treinar_corpo as tc  # noqa: E402

# BodyGestureEngine / LibrasAnalyzer
MOV_WINDOW, START_MOTION, END_MOTION = 5, 0.050, 0.030
START_FRAMES, END_FRAMES, MIN_FRAMES, MAX_FRAMES = 3, 5, 10, 60
CONFIANCA = 0.85
P_MAO_INI, P_TOTAL = 33, 75


def movimento(buffer):
    """Gemeo de BodyGestureEngine.bodyMotion: std dos pontos das MAOS (x,y)."""
    if len(buffer) < 3:
        return 0.0
    arr = np.array(buffer, dtype=np.float32)
    total, n = 0.0, 0
    for ponto in range(P_MAO_INI, P_TOTAL):
        for coord in (0, 1):
            total += float(arr[:, ponto * 3 + coord].std())
            n += 1
    return total / n if n else 0.0


def tem_mao(frame):
    return bool(np.any(frame[P_MAO_INI * 3:P_TOTAL * 3]))


def segmento_do_app(quadros):
    """Roda a maquina de estado do app sobre o clipe e devolve o trecho capturado."""
    buffer, capturando, gesto = [], False, []
    ini_cnt = fim_cnt = 0
    for frame in quadros:
        buffer.append(frame)
        if len(buffer) > MOV_WINDOW:
            buffer.pop(0)
        mov = movimento(buffer)
        if not capturando:
            if tem_mao(frame) and len(buffer) >= MOV_WINDOW and mov > START_MOTION:
                ini_cnt += 1
                if ini_cnt >= START_FRAMES:
                    capturando, gesto, fim_cnt = True, list(buffer), 0
            else:
                ini_cnt = 0
        else:
            gesto.append(frame)
            fim_cnt = fim_cnt + 1 if mov < END_MOTION else 0
            if fim_cnt >= END_FRAMES or len(gesto) >= MAX_FRAMES:
                return gesto if len(gesto) >= MIN_FRAMES else None
    return gesto if capturando and len(gesto) >= MIN_FRAMES else None


def janela(quadros, inicio, fim):
    a, b = int(len(quadros) * inicio), int(len(quadros) * fim)
    if b - a < 10:
        return None
    return np.array(tc.reamostra(list(quadros[a:b])), dtype=np.float32)


def monta_treino(seqs, rotulos, indices, cfg, rs):
    X, y = [], []
    for i in indices:
        quadros = seqs[i]
        for inicio, fim in cfg["recortes"]:
            v = janela(quadros, inicio, fim)
            if v is not None:
                X.append(v); y.append(rotulos[i])
        if cfg.get("usa_segmento_app"):
            seg = segmento_do_app(quadros)
            if seg is not None:
                X.append(np.array(tc.reamostra(seg), dtype=np.float32))
                y.append(rotulos[i])
        for _ in range(cfg.get("ruido_copias", 0)):
            base = janela(quadros, 0.0, 1.0)
            if base is not None:
                X.append(base + rs.normal(0, cfg["ruido"], base.shape).astype(np.float32))
                y.append(rotulos[i])
    return np.array(X, dtype=np.float32), np.array(y)


def monta_teste(seqs, rotulos, indices, modo):
    X, y = [], []
    for i in indices:
        quadros = seqs[i]
        if modo == "clipe":
            v = janela(quadros, 0.0, 1.0)
        else:
            seg = segmento_do_app(quadros)
            v = np.array(tc.reamostra(seg), dtype=np.float32) if seg is not None else None
        if v is not None:
            X.append(v); y.append(rotulos[i])
    return np.array(X, dtype=np.float32), np.array(y)


def treina(X, y_idx, n_classes, cfg, semente):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(semente)
    camadas = [tf.keras.layers.Input(shape=(tc.JANELA, tc.N_FEATURES))]
    if cfg.get("bidirecional"):
        camadas.append(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(cfg.get("unidades", 64))))
    else:
        camadas.append(tf.keras.layers.LSTM(cfg.get("unidades", 64)))
    camadas += [
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ]
    m = tf.keras.Sequential(camadas)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    peso = None
    if cfg.get("pesos_classe"):
        cont = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
        peso = {i: len(y_idx) / (n_classes * c) if c else 0.0
                for i, c in enumerate(cont)}
    m.fit(X, y_idx, epochs=cfg.get("epocas", 60), batch_size=16, verbose=0,
          class_weight=peso)
    return m


def avalia(cfg, seqs, rotulos, classes, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    rs = np.random.RandomState(0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    conf = {m: np.zeros((len(classes), len(classes)), int) for m in ("clipe", "app")}
    aceito = {m: np.zeros(len(classes), int) for m in ("clipe", "app")}

    for tr, te in skf.split(np.zeros(len(seqs)), rotulos):
        X_tr, y_tr = monta_treino(seqs, rotulos, tr, cfg, rs)
        y_idx = np.array([classes.index(g) for g in y_tr])
        m = treina(X_tr, y_idx, len(classes), cfg, 42)
        for modo in ("clipe", "app"):
            X_te, y_te = monta_teste(seqs, rotulos, te, modo)
            if not len(X_te):
                continue
            probs = m.predict(X_te, verbose=0)
            pred = probs.argmax(axis=1)
            for verdadeiro, previsto, p in zip(y_te, pred, probs):
                iv = classes.index(verdadeiro)
                conf[modo][iv, previsto] += 1
                # sucesso no app = classe certa E acima do limiar de confianca
                if previsto == iv and p[previsto] >= CONFIANCA:
                    aceito[modo][iv] += 1
    return conf, aceito


def mostra(nome, conf, aceito, classes):
    for modo in ("clipe", "app"):
        c, a = conf[modo], aceito[modo]
        total = c.sum()
        if not total:
            continue
        print("  [%s] acerto %.1f%%  | aceito pelo app (>=%.2f) %.1f%%"
              % (modo, np.trace(c) / total * 100, CONFIANCA, a.sum() / total * 100))
        for i, cl in enumerate(classes):
            n = c[i].sum()
            if n:
                print("      %-12s acerto %5.1f%%  aceito %5.1f%%"
                      % (cl, c[i, i] / n * 100, a[i] / n * 100))


def main():
    d = np.load(CACHE, allow_pickle=True)
    seqs, rotulos = d["seqs"], d["rotulos"]
    classes = sorted(set(rotulos))

    base_recortes = [(0.0, 1.0), (0.0, 0.9), (0.1, 1.0)]
    mais_recortes = [(0.0, 1.0), (0.0, 0.9), (0.1, 1.0), (0.05, 0.95),
                     (0.0, 0.8), (0.2, 1.0), (0.1, 0.9)]

    configs = {
        "A atual (base)": {"recortes": base_recortes},
        "B + segmento do app": {"recortes": base_recortes, "usa_segmento_app": True},
        "C + recortes e pesos": {"recortes": mais_recortes, "usa_segmento_app": True,
                                 "pesos_classe": True},
        "D + ruido": {"recortes": mais_recortes, "usa_segmento_app": True,
                      "pesos_classe": True, "ruido": 0.02, "ruido_copias": 2},
        "E D + bidirecional": {"recortes": mais_recortes, "usa_segmento_app": True,
                               "pesos_classe": True, "ruido": 0.02, "ruido_copias": 2,
                               "bidirecional": True, "unidades": 96, "epocas": 80},
    }

    for nome, cfg in configs.items():
        print("\n=== %s ===" % nome, flush=True)
        conf, aceito = avalia(cfg, seqs, rotulos, classes)
        mostra(nome, conf, aceito, classes)


if __name__ == "__main__":
    sys.exit(main())

