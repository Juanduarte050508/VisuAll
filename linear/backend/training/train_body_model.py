"""
Treina o modelo de gestos corporais (LSTM) e exporta pro formato que o app
Android carrega.

Não existia treino publicado pra esse modelo neste repositório (o
body_model.tflite atual veio de um pipeline externo — ver docstring de
extract_from_videos_corpo.py). Esta é a primeira versão treinável a partir
de dados gravados pelo próprio time, via treinamento/Capturar.bat.

Entrada:
  - VisuAll/data/dataset_corpo.npz  (gerado por extract_from_videos_corpo.py)

Saídas:
  - VisuAll/models/body_model.keras
  - VisuAll/models/body_classes.pkl
  - VisuAll/mobile/app/src/main/assets/gestos/geral/model.tflite
  - VisuAll/mobile/app/src/main/assets/gestos/geral/labels.txt

Arquitetura (nova, projetada pra este projeto — não é port de nada
existente, já que o modelo original nunca teve o treino publicado aqui):
LSTM simples + dropout + softmax. O TFLite exportado precisa do Select TF
Ops (FlexDelegate) porque o kernel de LSTM não é 100% coberto pelos ops
nativos do TFLite — o app Android já espera isso
(org.tensorflow.lite.flex.FlexDelegate em BodyGestureEngine.kt).
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from training_common import (
    BODY_LABELS,
    DATA_DIR,
    MOBILE_ASSETS_DIR,
    MODELS_DIR,
    balance_by_class,
    encode_labels_in_mobile_order,
    ensure_dirs,
    save_labels,
    save_pickle,
)

try:
    import tensorflow as tf
except ImportError as exc:
    raise RuntimeError(
        "Para treinar o modelo de corpo, instale o TensorFlow: pip install tensorflow"
    ) from exc


JANELA = 30
FEATURES = 225
MAX_POR_CLASSE = 300  # gestos corporais são mais caros de gravar que letras


def carregar_dataset():
    caminho = DATA_DIR / "dataset_corpo.npz"
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} não existe. Grave clipes com treinamento/Capturar.bat e rode "
            "extract_from_videos_corpo.py antes de treinar."
        )
    dados = np.load(caminho, allow_pickle=True)
    X = dados["X"].astype(np.float32)
    y = np.array([str(v).upper() for v in dados["y"]])
    if X.ndim != 3 or X.shape[1:] != (JANELA, FEATURES):
        raise ValueError(f"{caminho} tem shape {X.shape}; esperado [N, {JANELA}, {FEATURES}]")
    return X, y


def construir_modelo(n_classes):
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(JANELA, FEATURES)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def exportar_tflite(modelo, caminho):
    conversor = tf.lite.TFLiteConverter.from_keras_model(modelo)
    # LSTM precisa de Select TF Ops além dos builtins do TFLite — sem isso a
    # conversão falha ou o .tflite resultante não roda no FlexDelegate que o
    # app já usa.
    conversor.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = conversor.convert()
    Path(caminho).write_bytes(tflite_model)


def main():
    ensure_dirs()
    X, y = carregar_dataset()
    print(f"Amostras carregadas: {len(X)} | Shape: {X.shape}")
    print(f"Distribuicao: { {label: int((y == label).sum()) for label in BODY_LABELS} }")

    faltando = [label for label in BODY_LABELS if label not in set(y)]
    if faltando:
        print(f"\n⚠️  Sem NENHUMA amostra para: {faltando}")
        print("   O modelo não vai reconhecer esses gestos até você gravar clipes pra eles.")

    if len(X) < len(BODY_LABELS) * 4:
        print(f"\n⚠️  Poucas amostras no total ({len(X)}). O ideal é pelo menos ~20 "
              "clipes por gesto (ver treinamento/README.md). Treinando mesmo assim.")

    X_bal, y_bal = balance_by_class(X, y, max_per_class=MAX_POR_CLASSE)
    y_enc = encode_labels_in_mobile_order(y_bal, BODY_LABELS)

    print(f"Apos balanceamento: { {label: int((y_bal == label).sum()) for label in BODY_LABELS} }")
    print(f"Ordem mobile: {BODY_LABELS}")

    # Com poucas amostras por classe, stratify pode falhar (classe com 1
    # amostra não dá pra separar em treino+teste); cai pra split sem
    # stratify nesse caso em vez de travar o treino inteiro.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_bal, y_enc, test_size=0.2, random_state=42, stratify=y_enc,
        )
    except ValueError as e:
        print(f"\n⚠️  Não deu pra estratificar o split ({e}); usando split simples.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_bal, y_enc, test_size=0.2, random_state=42,
        )

    print("\nTreinando LSTM de corpo...")
    modelo = construir_modelo(len(BODY_LABELS))
    modelo.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=80,
        batch_size=16,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=12, restore_best_weights=True
            )
        ],
        verbose=2,
    )

    y_pred = np.argmax(modelo.predict(X_test), axis=1)
    present_labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    print(
        "\n"
        + classification_report(
            y_test,
            y_pred,
            labels=present_labels,
            target_names=[BODY_LABELS[i] for i in present_labels],
            zero_division=0,
        )
    )

    # gestos/geral/ -- convencao introduzida em paralelo por outro commit
    # (BodyGestureEngine.kt agora le daqui, nao mais de body_model.tflite
    # solto na raiz de assets/).
    body_assets = MOBILE_ASSETS_DIR / "gestos" / "geral"
    body_assets.mkdir(parents=True, exist_ok=True)

    modelo.save(MODELS_DIR / "body_model.keras")
    save_pickle(MODELS_DIR / "body_classes.pkl", BODY_LABELS)
    save_labels(body_assets / "labels.txt", BODY_LABELS)
    exportar_tflite(modelo, body_assets / "model.tflite")

    print(f"\nModelo Python salvo em: {MODELS_DIR / 'body_model.keras'}")
    print(f"Classes Python salvas em: {MODELS_DIR / 'body_classes.pkl'}")
    print(f"Modelo Android salvo em: {body_assets / 'model.tflite'}")
    print(f"Labels Android salvos em: {body_assets / 'labels.txt'}")


if __name__ == "__main__":
    main()
