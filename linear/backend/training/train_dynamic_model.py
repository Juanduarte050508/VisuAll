"""
Treina o modelo de letras dinamicas e gera os artefatos do backend Python e do
app Android.

Entradas aceitas:
  - VisuAll/data/dataset_dynamic.npz
  - VisuAll/data/visuall_libras_dynamic_phone_dataset.csv
  - VisuAll/data/dynamic_external_dataset.csv

Saidas:
  - VisuAll/models/dynamic_model.pkl
  - VisuAll/models/dynamic_classes.pkl
  - VisuAll/mobile/app/src/main/assets/dynamic_model.onnx
  - VisuAll/mobile/app/src/main/assets/dynamic_labels.txt
"""
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from training_common import (
    DATA_DIR,
    DYNAMIC_LABELS,
    MOBILE_ASSETS_DIR,
    MODELS_DIR,
    balance_by_class,
    encode_labels_in_mobile_order,
    ensure_dirs,
    export_onnx_model,
    load_datasets,
    save_labels,
    save_pickle,
)


FEATURES = 420
MAX_POR_CLASSE = 500


ensure_dirs()

X, y = load_datasets(
    npz_path=DATA_DIR / "dataset_dynamic.npz",
    csv_paths=[
        DATA_DIR / "visuall_libras_dynamic_phone_dataset.csv",
        DATA_DIR / "dynamic_external_dataset.csv",
    ],
    expected_features=FEATURES,
    allowed_labels=DYNAMIC_LABELS,
)

print(f"Amostras carregadas: {len(X)} | Features: {X.shape[1]}")
print(f"Distribuicao: { {label: int((y == label).sum()) for label in DYNAMIC_LABELS} }")

X, y = balance_by_class(X, y, max_per_class=MAX_POR_CLASSE)
y_enc = encode_labels_in_mobile_order(y, DYNAMIC_LABELS)

print(f"Apos balanceamento: { {label: int((y == label).sum()) for label in DYNAMIC_LABELS} }")
print(f"Ordem mobile: {DYNAMIC_LABELS}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc,
)

print("Treinando MLP dinamico...")
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=True,
)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
present_labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
print(
    "\n"
    + classification_report(
        y_test,
        y_pred,
        labels=present_labels,
        target_names=[DYNAMIC_LABELS[i] for i in present_labels],
        zero_division=0,
    )
)

save_pickle(MODELS_DIR / "dynamic_model.pkl", mlp)
save_pickle(MODELS_DIR / "dynamic_classes.pkl", DYNAMIC_LABELS)
save_labels(MOBILE_ASSETS_DIR / "dynamic_labels.txt", DYNAMIC_LABELS)
export_onnx_model(mlp, MOBILE_ASSETS_DIR / "dynamic_model.onnx", FEATURES)

print(f"Modelo Python salvo em: {MODELS_DIR / 'dynamic_model.pkl'}")
print(f"Classes Python salvas em: {MODELS_DIR / 'dynamic_classes.pkl'}")
print(f"Modelo Android salvo em: {MOBILE_ASSETS_DIR / 'dynamic_model.onnx'}")
print(f"Labels Android salvos em: {MOBILE_ASSETS_DIR / 'dynamic_labels.txt'}")
