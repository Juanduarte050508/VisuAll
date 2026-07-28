from pathlib import Path
import csv
import pickle

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MOBILE_ASSETS_DIR = ROOT / "mobile" / "app" / "src" / "main" / "assets"

STATIC_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "I", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "Y",
]
DYNAMIC_LABELS = ["H", "J", "K", "X", "Z"]
# Ordem alfabética = mesma ordem de mobile/app/src/main/assets/body_labels.txt.
# NEUTRO é uma classe real (não "sem gesto"): o modelo precisa de exemplos de
# gente parada/sem sinalizar pra não confundir estar-parado com um dos sinais.
BODY_LABELS = ["AJUDAR", "COMPUTADOR", "CONVERSAR", "NEUTRO", "PESSOA", "SURDO"]


def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MOBILE_ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def load_npz(path, expected_features):
    path = Path(path)
    if not path.exists():
        return None, None
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = np.array([str(v).upper() for v in data["y"]])
    if X.ndim != 2 or X.shape[1] != expected_features:
        raise ValueError(f"{path} tem shape {X.shape}; esperado [N, {expected_features}]")
    return X, y


def load_csv(path, expected_features, allowed_labels):
    path = Path(path)
    if not path.exists():
        return None, None

    X, y = [], []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        feature_names = [f"f{i}" for i in range(expected_features)]
        missing = [name for name in feature_names if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} nao tem todas as features: faltando {missing[:5]}")
        if "label" not in reader.fieldnames:
            raise ValueError(f"{path} nao tem coluna 'label'")

        for row in reader:
            label = row["label"].strip().upper()
            if label not in allowed_labels:
                continue
            X.append([float(row[name]) for name in feature_names])
            y.append(label)

    if not X:
        return None, None
    return np.array(X, dtype=np.float32), np.array(y)


def load_datasets(npz_path, csv_paths, expected_features, allowed_labels):
    chunks_X, chunks_y = [], []

    X, y = load_npz(npz_path, expected_features)
    if X is not None:
        chunks_X.append(X)
        chunks_y.append(y)

    for csv_path in csv_paths:
        X, y = load_csv(csv_path, expected_features, allowed_labels)
        if X is not None:
            chunks_X.append(X)
            chunks_y.append(y)

    if not chunks_X:
        raise FileNotFoundError(
            "Nenhum dataset encontrado. Gere um .npz ou exporte CSVs pelo app."
        )

    X = np.concatenate(chunks_X).astype(np.float32)
    y = np.concatenate(chunks_y)
    keep = np.isin(y, allowed_labels)
    return X[keep], y[keep]


def encode_labels_in_mobile_order(y, label_order):
    label_to_index = {label: index for index, label in enumerate(label_order)}
    missing = [label for label in label_order if label not in set(y)]
    if missing:
        print(f"Atencao: sem amostras para labels: {missing}")
    return np.array([label_to_index[label] for label in y], dtype=np.int64)


def balance_by_class(X, y, max_per_class, seed=42):
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        selected.extend(idx.tolist())
    selected = np.array(selected, dtype=np.int64)
    rng.shuffle(selected)
    return X[selected], y[selected]


def save_pickle(path, obj):
    with Path(path).open("wb") as f:
        pickle.dump(obj, f)


def save_labels(path, labels):
    Path(path).write_text("\n".join(labels) + "\n", encoding="utf-8")


def export_onnx_model(model, path, input_features):
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:
        raise RuntimeError(
            "Para exportar ONNX, instale as dependencias: pip install skl2onnx onnx"
        ) from exc

    initial_type = [("landmarks_input", FloatTensorType([None, input_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        options={id(model): {"zipmap": False}},
        target_opset=12,
    )
    Path(path).write_bytes(onnx_model.SerializeToString())
