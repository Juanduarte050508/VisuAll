"""
Programa unificado para importar fotos/videos e treinar os modelos do VisuAll.

Uso rapido:

  python treinar_visuall.py status
  python treinar_visuall.py importar C:\\dataset_libras
  python treinar_visuall.py tudo C:\\dataset_libras

Estrutura recomendada de entrada:

  dataset_libras/
    A/ foto1.jpg foto2.png video_estatico.mp4
    B/ ...
    H/ movimento1.mp4
    J/ ...
    AJUDAR/ gesto1.mp4
    PESSOA/ ...

O programa descobre o tipo pelo label:

  - labels estaticos + fotos: treinamento/dados/raw_images/<LABEL>/
  - labels estaticos + videos: treinamento/dados/raw_static_videos/<LABEL>/
  - labels dinamicos + videos: treinamento/dados/raw_videos/<LABEL>/
  - labels corporais + videos: treinamento/dados/raw_body_videos/<LABEL>/

Depois ele gera datasets e modelos nos formatos que o app Android espera.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT if (ROOT / "mobile").exists() else ROOT / "VisuAll"
DATA_DIR = ROOT / "treinamento" / "dados"
PARTIAL_MODELS_DIR = ROOT / "treinamento" / "modelos_parciais"
INDIVIDUAL_MODELS_DIR = ROOT / "treinamento" / "modelos_individuais"
MOBILE_ASSETS = PROJECT / "mobile" / "app" / "src" / "main" / "assets"
STATIC_ASSETS = MOBILE_ASSETS / "letras_estaticas"
DYNAMIC_ASSETS = MOBILE_ASSETS / "letras_dinamicas"
GESTURE_ASSETS = MOBILE_ASSETS / "gestos"

# training_common.py mora aqui do lado (treinamento/). Ficava em
# linear/backend/training/ junto de dois treinadores que este arquivo
# substituiu; com eles removidos, nao fazia mais sentido o codigo de treino
# ficar espalhado em duas pastas.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from training_common import (  # noqa: E402
    DYNAMIC_LABELS,
    MODELS_DIR,
    STATIC_LABELS,
    balance_by_class,
    encode_labels_in_mobile_order,
    ensure_dirs,
    export_onnx_model,
    save_labels,
    save_pickle,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
BODY_LABELS = [
    line.strip().upper()
    for line in (GESTURE_ASSETS / "geral" / "labels.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
ALL_LABELS = set(STATIC_LABELS) | set(DYNAMIC_LABELS) | set(BODY_LABELS)
PARTIAL_MODELS_TO_KEEP = 5
# Rotulo especial: nao e letra nem gesto, e "isto NAO e sinal nenhum" (mao a
# toa, cocando a cabeca, gesticulando enquanto fala). Usado so como exemplo
# negativo no treino dos modelos individuais -- nunca vira classe de saida,
# entao nao entra em ALL_LABELS nem em nenhum labels.txt do app.
NEGATIVE_LABEL = "NADA"
NEGATIVE_DIR = DATA_DIR / "raw_negativos"


def safe_label_name(label: str) -> str:
    return "".join(ch for ch in label.upper() if ch.isalnum() or ch in ("_", "-"))


@dataclass
class ImportStats:
    copied: int = 0
    skipped: int = 0


def require_cv_stack():
    try:
        import cv2  # noqa: F401
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks.python.core.base_options import BaseOptions  # noqa: F401
        from mediapipe.tasks.python.vision import HandLandmarker  # noqa: F401
        from mediapipe.tasks.python.vision import HandLandmarkerOptions  # noqa: F401
        from mediapipe.tasks.python.vision import PoseLandmarker  # noqa: F401
        from mediapipe.tasks.python.vision import PoseLandmarkerOptions  # noqa: F401
        from mediapipe.tasks.python.vision import RunningMode  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"Faltam dependencias de extracao. Rode: pip install -r {PROJECT / 'requirements.txt'}"
        ) from exc


def require_sklearn_stack():
    try:
        from sklearn.metrics import classification_report  # noqa: F401
        from sklearn.model_selection import train_test_split  # noqa: F401
        from sklearn.neural_network import MLPClassifier  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"Faltam dependencias de treino. Rode: pip install -r {PROJECT / 'requirements.txt'}"
        ) from exc


def require_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Para treinar gestos corporais, instale TensorFlow: pip install tensorflow"
        ) from exc


def normalize_label(raw: str) -> str:
    return raw.strip().upper()


def media_kind(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS:
        return "image"
    if suffix in VIDEO_EXTS:
        return "video"
    return None


def target_dir_for(label: str, kind: str) -> Path | None:
    if label in STATIC_LABELS and kind == "image":
        return DATA_DIR / "raw_images" / label
    if label in STATIC_LABELS and kind == "video":
        return DATA_DIR / "raw_static_videos" / label
    if label in DYNAMIC_LABELS and kind == "video":
        return DATA_DIR / "raw_videos" / label
    if label in BODY_LABELS and kind == "video":
        return DATA_DIR / "raw_body_videos" / label
    return None


def unique_destination(directory: Path, source: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / source.name
    if not candidate.exists():
        return candidate
    stem = source.stem
    suffix = source.suffix
    counter = 2
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def import_one_file(path: Path, label: str, stats: ImportStats) -> None:
    kind = media_kind(path)
    if kind is None:
        stats.skipped += 1
        return

    destination_dir = target_dir_for(label, kind)
    if destination_dir is None:
        stats.skipped += 1
        return

    destination = unique_destination(destination_dir, path)
    shutil.copy2(path, destination)
    stats.copied += 1


def import_media(source: Path, forced_label: str | None = None) -> ImportStats:
    stats = ImportStats()
    source = source.resolve()
    if not source.exists():
        raise SystemExit(f"Entrada nao encontrada: {source}")

    if source.is_file():
        if not forced_label:
            raise SystemExit("Para importar um arquivo isolado, informe --label.")
        label = normalize_label(forced_label)
        if label not in ALL_LABELS:
            raise SystemExit(f"Label desconhecido: {label}")
        import_one_file(source, label, stats)
        return stats

    if forced_label:
        label = normalize_label(forced_label)
        if label not in ALL_LABELS:
            raise SystemExit(f"Label desconhecido: {label}")
        for path in source.rglob("*"):
            if path.is_file():
                import_one_file(path, label, stats)
        return stats

    for label_dir in source.iterdir():
        if not label_dir.is_dir():
            continue
        label = normalize_label(label_dir.name)
        if label not in ALL_LABELS:
            stats.skipped += len([p for p in label_dir.rglob("*") if p.is_file()])
            continue
        for path in label_dir.rglob("*"):
            if path.is_file():
                import_one_file(path, label, stats)
    return stats


def normalize_hand_landmarks(points: list[tuple[float, float]]) -> list[float]:
    base_x, base_y = points[0]
    values: list[float] = []
    for x, y in points:
        values.extend([x - base_x, y - base_y])
    max_value = max(abs(v) for v in values) or 1.0
    return [v / max_value for v in values]


def mp_image_from_bgr(frame):
    import cv2
    import mediapipe as mp

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def create_hand_landmarker(running_mode, num_hands: int = 1):
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker
    from mediapipe.tasks.python.vision import HandLandmarkerOptions

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MOBILE_ASSETS / "hand_landmarker.task")),
        running_mode=running_mode,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def create_pose_landmarker(running_mode):
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker
    from mediapipe.tasks.python.vision import PoseLandmarkerOptions

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MOBILE_ASSETS / "pose_landmarker_lite.task")),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def first_hand_points(hand_result) -> list[tuple[float, float]] | None:
    if not hand_result.hand_landmarks:
        return None
    return [(lm.x, lm.y) for lm in hand_result.hand_landmarks[0]]


def extract_static_dataset(frame_stride: int = 5) -> None:
    require_cv_stack()
    import cv2
    from mediapipe.tasks.python.vision import RunningMode

    print("Extraindo letras estaticas...")
    rows: list[tuple[int, str, str, list[float]]] = []
    hands_image = create_hand_landmarker(RunningMode.IMAGE, num_hands=1)
    hands_video = create_hand_landmarker(RunningMode.VIDEO, num_hands=1)
    timestamp_ms = 0

    for label in STATIC_LABELS:
        image_dir = DATA_DIR / "raw_images" / label
        image_files = sorted(image_dir.glob("*")) if image_dir.exists() else []
        video_dir = DATA_DIR / "raw_static_videos" / label
        video_files = sorted(video_dir.glob("*")) if video_dir.exists() else []
        if image_files or video_files:
            print(f"  {label}: {len(image_files)} fotos, {len(video_files)} videos")
        for image_path in image_files:
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            result = hands_image.detect(mp_image_from_bgr(frame))
            points = first_hand_points(result)
            if points is None:
                continue
            rows.append((image_path.stat().st_mtime_ns, label, "external_image", normalize_hand_landmarks(points)))

        for video_path in video_files:
            if video_path.suffix.lower() not in VIDEO_EXTS:
                continue
            cap = cv2.VideoCapture(str(video_path))
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % frame_stride != 0:
                    frame_index += 1
                    continue
                result = hands_video.detect_for_video(mp_image_from_bgr(frame), timestamp_ms)
                timestamp_ms += 1
                points = first_hand_points(result)
                if points is not None:
                    rows.append((
                        video_path.stat().st_mtime_ns + frame_index,
                        label,
                        "external_static_video",
                        normalize_hand_landmarks(points),
                    ))
                frame_index += 1
            cap.release()

    hands_image.close()
    hands_video.close()
    save_hand_dataset(rows, DATA_DIR / "dataset_static.npz", DATA_DIR / "static_external_dataset.csv", 42)


def extract_dynamic_dataset(window: int = 10, step: int = 1) -> None:
    require_cv_stack()
    import cv2
    from mediapipe.tasks.python.vision import RunningMode

    print("Extraindo letras dinamicas...")
    rows: list[tuple[int, str, str, list[float]]] = []
    hands = create_hand_landmarker(RunningMode.VIDEO, num_hands=1)
    timestamp_ms = 0

    for label in DYNAMIC_LABELS:
        video_dir = DATA_DIR / "raw_videos" / label
        video_files = sorted(video_dir.glob("*")) if video_dir.exists() else []
        if video_files:
            print(f"  {label}: {len(video_files)} videos")
        for video_index, video_path in enumerate(video_files, start=1):
            if video_path.suffix.lower() not in VIDEO_EXTS:
                continue
            if video_index == 1 or video_index % 10 == 0 or video_index == len(video_files):
                print(f"    {label}: video {video_index}/{len(video_files)}")
            cap = cv2.VideoCapture(str(video_path))
            frames: list[list[float]] = []
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = hands.detect_for_video(mp_image_from_bgr(frame), timestamp_ms)
                timestamp_ms += 1
                points = first_hand_points(result)
                if points is not None:
                    frames.append(normalize_hand_landmarks(points))
                else:
                    frames = []
                frame_index += 1
            cap.release()

            for index in range(0, max(0, len(frames) - window + 1), step):
                features = np.array(frames[index:index + window], dtype=np.float32).flatten().tolist()
                rows.append((video_path.stat().st_mtime_ns + index, label, "external_dynamic_video", features))

    hands.close()
    save_hand_dataset(rows, DATA_DIR / "dataset_dynamic.npz", DATA_DIR / "dynamic_external_dataset.csv", 420)


def extract_negative_dataset(frame_stride: int = 5, window: int = 10, step: int = 2) -> None:
    """Extrai os clipes de "nao e sinal nenhum" em dois datasets de uma vez.

    O mesmo clipe serve pros dois casos: cada quadro isolado vira exemplo
    negativo estatico (42 features) e cada janela de 10 quadros vira exemplo
    negativo dinamico (420 features). Nao tem label -- e so um monte de "isto
    aqui nao e letra".
    """
    require_cv_stack()
    import cv2
    from mediapipe.tasks.python.vision import RunningMode

    videos = sorted(
        path for path in NEGATIVE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    ) if NEGATIVE_DIR.exists() else []

    if not videos:
        print("Sem clipes negativos (dados/raw_negativos/). Pulando.")
        return

    print(f"Extraindo negativos ({len(videos)} clipes)...")
    hands = create_hand_landmarker(RunningMode.VIDEO, num_hands=1)
    timestamp_ms = 0
    estaticos: list[list[float]] = []
    dinamicos: list[list[float]] = []

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        frames: list[list[float]] = []
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = hands.detect_for_video(mp_image_from_bgr(frame), timestamp_ms)
            timestamp_ms += 1
            points = first_hand_points(result)
            if points is not None:
                values = normalize_hand_landmarks(points)
                frames.append(values)
                # Sub-amostra os quadros pro dataset estatico: quadros
                # vizinhos de um mesmo clipe sao quase identicos.
                if frame_index % frame_stride == 0:
                    estaticos.append(values)
            else:
                # Sem mao no quadro: quebra a sequencia (uma janela dinamica
                # nao pode juntar dois trechos separados do video).
                frames = []
            frame_index += 1
        cap.release()

        for index in range(0, max(0, len(frames) - window + 1), step):
            dinamicos.append(
                np.array(frames[index:index + window], dtype=np.float32).flatten().tolist()
            )

    hands.close()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for values, features, path in (
        (estaticos, 42, DATA_DIR / "dataset_static_negativos.npz"),
        (dinamicos, 420, DATA_DIR / "dataset_dynamic_negativos.npz"),
    ):
        if not values:
            print(f"  nenhum negativo de {features} features gerado.")
            continue
        X = np.array(values, dtype=np.float32)
        np.savez(path, X=X, y=np.array([NEGATIVE_LABEL] * len(X)))
        print(f"  negativos salvos: {path} ({len(X)} amostras)")


def load_negative_pool(name: str, features: int) -> np.ndarray | None:
    path = DATA_DIR / f"dataset_{name}_negativos.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    if X.ndim != 2 or X.shape[1] != features:
        print(f"Ignorando {path}: shape {X.shape}, esperado [N, {features}].")
        return None
    return X


class ModeloInvalidoError(RuntimeError):
    """Modelo exportado nao bate com o que o app carrega."""


def verificar_onnx_exportado(caminho: Path, features: int, n_classes: int | None = None) -> None:
    """Confere se o .onnx recem-gerado e mesmo o que o app espera receber.

    Sem isso, um export com o nome ou o formato de entrada errado passa
    despercebido aqui e so aparece la na frente, com o app ja instalado no
    celular: a tela simplesmente para de reconhecer, sem mensagem nenhuma.
    O contrato conferido e o do LetraEngine.kt, que roda a sessao com
    mapOf("landmarks_input" to tensor) e le a saida [1, n_classes].
    """
    import onnx

    modelo = onnx.load(str(caminho))
    onnx.checker.check_model(modelo)

    entradas = modelo.graph.input
    if len(entradas) != 1:
        raise ModeloInvalidoError(
            f"{caminho.name}: esperava 1 entrada, tem {len(entradas)} "
            f"({[e.name for e in entradas]}). O app so alimenta uma."
        )

    entrada = entradas[0]
    if entrada.name != "landmarks_input":
        raise ModeloInvalidoError(
            f"{caminho.name}: entrada se chama '{entrada.name}', mas o app procura "
            "'landmarks_input' — ele nao conseguiria rodar este modelo."
        )

    dims = entrada.type.tensor_type.shape.dim
    if len(dims) != 2 or dims[1].dim_value != features:
        forma = [d.dim_value or d.dim_param or "?" for d in dims]
        raise ModeloInvalidoError(
            f"{caminho.name}: entrada tem forma {forma}, esperado [N, {features}]."
        )

    if n_classes is not None:
        # O app le a SEGUNDA saida (out[1]) como as probabilidades por classe.
        saidas = modelo.graph.output
        if len(saidas) < 2:
            raise ModeloInvalidoError(
                f"{caminho.name}: tem {len(saidas)} saida(s); o app le a segunda "
                "(probabilidades). Exportou sem zipmap=False?"
            )


def verificar_tflite_exportado(caminho: Path, janela: int, features: int, n_classes: int) -> None:
    """Mesma ideia do verificar_onnx_exportado, pro modelo de gestos.

    Confere o contrato do BodyGestureEngine.kt, que faz
    resizeInput(0, [1, BODY_WINDOW, BODY_FEATURES]) e le a saida [1, n_classes].
    """
    import tensorflow as tf

    interpretador = tf.lite.Interpreter(model_path=str(caminho))
    interpretador.resize_tensor_input(
        interpretador.get_input_details()[0]["index"], [1, janela, features]
    )
    interpretador.allocate_tensors()

    entrada = interpretador.get_input_details()[0]
    forma_entrada = list(entrada["shape"])
    if forma_entrada != [1, janela, features]:
        raise ModeloInvalidoError(
            f"{caminho.name}: entrada {forma_entrada}, esperado [1, {janela}, {features}]."
        )

    saida = interpretador.get_output_details()[0]
    forma_saida = list(saida["shape"])
    if len(forma_saida) != 2 or forma_saida[1] != n_classes:
        raise ModeloInvalidoError(
            f"{caminho.name}: saida {forma_saida}, esperado [1, {n_classes}] "
            f"(uma probabilidade por gesto do labels.txt)."
        )

    # Roda de verdade: o modelo usa LSTM, que precisa do Select TF Ops. Se o
    # export tiver saido sem isso, e aqui que estoura -- e nao no celular.
    interpretador.set_tensor(
        entrada["index"], np.zeros((1, janela, features), dtype=np.float32)
    )
    interpretador.invoke()


def save_hand_dataset(
    rows: list[tuple[int, str, str, list[float]]],
    npz_path: Path,
    csv_path: Path,
    features: int,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"Nenhuma amostra gerada para {npz_path.name}.")
        return

    X_new = np.array([row[3] for row in rows], dtype=np.float32)
    y_new = np.array([row[1] for row in rows])
    X_new, y_new, removed_new = filter_outlier_samples(X_new, y_new)
    if removed_new:
        print(f"Outliers ignorados na extracao atual de {npz_path.name}: {removed_new}")
    if len(X_new) == 0:
        print(f"Nenhuma amostra consistente gerada para {npz_path.name}.")
        return

    if npz_path.exists():
        old = np.load(npz_path, allow_pickle=True)
        X_old = old["X"].astype(np.float32)
        y_old = np.array([str(v).upper() for v in old["y"]])
        if X_old.ndim == 2 and X_old.shape[1] == features:
            X = np.concatenate([X_old, X_new])
            y = np.concatenate([y_old, y_new])
        else:
            print(f"Ignorando dataset antigo com shape inesperado: {npz_path} {X_old.shape}")
            X, y = X_new, y_new
    else:
        X, y = X_new, y_new

    X, y = deduplicate_samples(X, y)
    X, y, removed_total = filter_outlier_samples(X, y)
    if removed_total:
        print(f"Outliers ignorados no dataset acumulado de {npz_path.name}: {removed_total}")
    np.savez(npz_path, X=X, y=y)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "label", "source"] + [f"f{i}" for i in range(features)])
        for index, (label, values) in enumerate(zip(y, X, strict=True)):
            writer.writerow([index, label, "dataset_acumulado"] + values.astype(float).tolist())

    print(f"Dataset atualizado: {npz_path} ({len(X)} amostras acumuladas)")
    print(f"CSV salvo: {csv_path}")


def deduplicate_samples(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep: list[int] = []
    seen: set[tuple[str, bytes]] = set()
    rounded = np.round(X.astype(np.float32), decimals=6)
    for index, label in enumerate(y):
        key = (str(label), rounded[index].tobytes())
        if key in seen:
            continue
        seen.add(key)
        keep.append(index)
    return X[keep], y[keep]


def filter_outlier_samples(
    X: np.ndarray,
    y: np.ndarray,
    min_samples: int = 8,
    mad_multiplier: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    keep = np.ones(len(X), dtype=bool)
    removed: dict[str, int] = {}
    for label in sorted(set(str(v) for v in y)):
        idx = np.where(y == label)[0]
        if len(idx) < min_samples:
            continue
        values = X[idx].astype(np.float32)
        center = np.median(values, axis=0)
        distances = np.linalg.norm(values - center, axis=1)
        median_distance = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median_distance)))
        if mad <= 1e-8:
            threshold = float(np.percentile(distances, 99))
        else:
            threshold = median_distance + mad_multiplier * mad
        drop = idx[distances > threshold]
        if len(drop):
            keep[drop] = False
            removed[label] = int(len(drop))
    return X[keep], y[keep], removed


def load_npz_dataset(path: Path, features: int, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise SystemExit(f"Dataset nao encontrado: {path}")
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = np.array([str(v).upper() for v in data["y"]])
    if X.ndim != 2 or X.shape[1] != features:
        raise SystemExit(f"{path} tem shape {X.shape}; esperado [N, {features}]")
    keep = np.isin(y, labels)
    return X[keep], y[keep]


def cleanup_partial_models(name: str) -> None:
    if not PARTIAL_MODELS_DIR.exists():
        return
    runs = sorted(
        [path for path in PARTIAL_MODELS_DIR.glob(f"{name}_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for old_run in runs[PARTIAL_MODELS_TO_KEEP:]:
        shutil.rmtree(old_run, ignore_errors=True)


def save_partial_model(
    name: str,
    model,
    labels_for_model: list[str],
    features: int,
    report_text: str,
    missing_labels: list[str],
) -> None:
    run_dir = PARTIAL_MODELS_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_pickle(run_dir / f"{name}_model.pkl", model)
    save_pickle(run_dir / f"{name}_classes.pkl", labels_for_model)
    save_labels(run_dir / f"{name}_labels.txt", labels_for_model)
    export_onnx_model(model, run_dir / f"{name}_model.onnx", features)
    if name == "dynamic":
        partial_dir = DYNAMIC_ASSETS / "parcial"
        partial_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / "dynamic_model.onnx", partial_dir / "model.onnx")
        shutil.copy2(run_dir / "dynamic_labels.txt", partial_dir / "labels.txt")
        print("Modelo parcial dinamico tambem foi aplicado como fallback prioritario no app.")
    (run_dir / "RELATORIO.txt").write_text(
        "MODELO PARCIAL - NAO APLICADO NO APP\n\n"
        f"Labels treinados: {labels_for_model}\n"
        f"Labels faltando: {missing_labels}\n\n"
        f"{report_text}\n",
        encoding="utf-8",
    )
    cleanup_partial_models(name)
    print(f"Modelo parcial salvo em: {run_dir}")
    print("O app nao foi sobrescrito porque ainda faltam labels obrigatorios.")


def save_individual_assets(name: str, trained_labels: list[str]) -> None:
    base_assets = STATIC_ASSETS if name == "static" else DYNAMIC_ASSETS
    label_order = STATIC_LABELS if name == "static" else DYNAMIC_LABELS
    base_assets.mkdir(parents=True, exist_ok=True)
    existing_path = base_assets / "individual_labels.txt"
    existing_labels = []
    if existing_path.exists():
        existing_labels = [
            line.strip().upper()
            for line in existing_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    merged = [label for label in label_order if label in set(existing_labels) | set(trained_labels)]
    for label in trained_labels:
        safe = safe_label_name(label)
        source = INDIVIDUAL_MODELS_DIR / name / f"{safe}.onnx"
        if source.exists():
            target_dir = base_assets / safe
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target_dir / "model.onnx")
    save_labels(existing_path, merged)


def train_individual_mlp(name: str, features: int, labels: list[str], max_per_class: int) -> None:
    require_sklearn_stack()
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    dataset = DATA_DIR / f"dataset_{name}.npz"
    if not dataset.exists():
        print(f"Pulando modelos individuais {name}: dataset_{name}.npz nao existe.")
        return

    X_all, y_all = load_npz_dataset(dataset, features, labels)
    if len(X_all) == 0:
        print(f"Pulando modelos individuais {name}: sem amostras validas.")
        return

    output_dir = INDIVIDUAL_MODELS_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    trained_labels: list[str] = []
    rng = np.random.default_rng(42)

    # Exemplos de "nao e sinal nenhum" (dados/raw_negativos/). Sem eles, um
    # modelo individual so aprende a separar SUA letra das OUTRAS LETRAS --
    # nunca ve mao a toa, entao no app ele fica confiante demais em qualquer
    # movimento e o reconhecimento dispara sem a pessoa estar sinalizando.
    # Ver CONFIANCA_INDIVIDUAL em LibrasAnalyzer.kt, que e o remendo do lado
    # do app pro mesmo problema.
    negativos = load_negative_pool(name, features)
    if negativos is None:
        print(
            f"  AVISO: sem clipes de 'Nada' pra {name}. Os modelos individuais vao "
            "ficar mais propensos a reconhecer letra onde nao tem. Grave alguns "
            "clipes na categoria 'Nada (nao e sinal nenhum)' do Capturar."
        )
    else:
        print(f"  usando {len(negativos)} exemplos de 'nao e sinal nenhum'.")

    labels_com_amostras = [label for label in labels if np.any(y_all == label)]
    print(f"Treinando modelos individuais {name}: {labels_com_amostras}")
    for label in labels_com_amostras:
        positivos = X_all[y_all == label]
        outras_letras = X_all[y_all != label]
        if len(outras_letras) == 0 and negativos is None:
            print(f"  {label}: precisa de exemplos de outras classes para negativo.")
            continue

        if len(positivos) > max_per_class:
            positivos = positivos[rng.choice(len(positivos), max_per_class, replace=False)]

        # Teto por fonte de negativo: sem isso, um punhado de positivos contra
        # milhares de negativos faz o modelo aprender so a dizer "nao".
        teto_negativo = min(max(len(positivos), 1) * 2, max_per_class * 2)
        partes_negativas = []
        for fonte in (outras_letras, negativos):
            if fonte is None or len(fonte) == 0:
                continue
            limite = min(len(fonte), teto_negativo)
            partes_negativas.append(fonte[rng.choice(len(fonte), limite, replace=False)])

        if not partes_negativas:
            print(f"  {label}: sem nenhum exemplo negativo, nao treinado.")
            continue

        negativos_label = np.concatenate(partes_negativas)
        X = np.concatenate([positivos, negativos_label])
        y_binary = np.concatenate([
            np.ones(len(positivos), dtype=np.int64),
            np.zeros(len(negativos_label), dtype=np.int64),
        ])
        ordem = rng.permutation(len(X))
        X, y_binary = X[ordem], y_binary[ordem]

        stratify = y_binary if np.all(np.bincount(y_binary, minlength=2) >= 2) else None
        if len(X) >= 5 and len(np.unique(y_binary)) == 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_binary,
                test_size=0.2,
                random_state=42,
                stratify=stratify,
            )
        else:
            X_train, y_train = X, y_binary
            X_test, y_test = None, None

        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=400,
            random_state=42,
            early_stopping=len(X_train) >= 30,
            validation_fraction=0.1,
            verbose=False,
        )
        model.fit(X_train, y_train)

        report_text = "Sem conjunto de teste separado."
        if X_test is not None:
            y_pred = model.predict(X_test)
            report_text = classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=[f"NAO_{label}", label],
                zero_division=0,
            )

        safe = safe_label_name(label)
        save_pickle(output_dir / f"{safe}.pkl", model)
        export_onnx_model(model, output_dir / f"{safe}.onnx", features)
        # Individuais sao binarios (2 classes: e a letra / nao e).
        verificar_onnx_exportado(output_dir / f"{safe}.onnx", features, n_classes=2)
        (output_dir / f"{safe}_RELATORIO.txt").write_text(
            f"MODELO INDIVIDUAL {name}:{label}\n\n"
            f"Positivos: {int((y_binary == 1).sum())}\n"
            f"Negativos: {int((y_binary == 0).sum())}\n"
            f"Exemplos de 'nao e sinal nenhum' no pool: "
            f"{0 if negativos is None else len(negativos)}\n\n"
            f"{report_text}\n",
            encoding="utf-8",
        )
        trained_labels.append(label)
        print(f"  {label}: treinado individualmente.")

    if trained_labels:
        save_labels(output_dir / f"{name}_individual_labels.txt", trained_labels)
        save_individual_assets(name, trained_labels)
        print(f"Modelos individuais {name} aplicados como prioridade no app: {trained_labels}")
    else:
        print(f"Nenhum modelo individual {name} foi treinado.")


def train_mlp(name: str, features: int, labels: list[str], max_per_class: int) -> None:
    require_sklearn_stack()
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    dataset = DATA_DIR / f"dataset_{name}.npz"
    X, y = load_npz_dataset(dataset, features, labels)
    if len(X) == 0:
        raise SystemExit(f"Sem amostras validas para {name}.")

    present_labels = [label for label in labels if np.any(y == label)]
    missing_labels = [label for label in labels if label not in present_labels]
    if missing_labels:
        print(f"Treino {name} parcial: faltam amostras para {missing_labels}.")
        print("Vou treinar o que existe e salvar como parcial, sem sobrescrever o app.")
        if len(present_labels) < 2:
            raise SystemExit(
                f"Treino {name} cancelado: precisa de pelo menos 2 labels com amostras. "
                f"Encontrado: {present_labels}"
            )
        labels_for_model = present_labels
    else:
        labels_for_model = labels

    X, y = balance_by_class(X, y, max_per_class=max_per_class)
    y_enc = encode_labels_in_mobile_order(y, labels_for_model)
    counts = {label: int((y == label).sum()) for label in labels_for_model}
    print(f"Treinando {name}: {len(X)} amostras | {counts}")

    class_counts = np.bincount(y_enc, minlength=len(labels_for_model))
    stratify = y_enc if np.all(class_counts[class_counts > 0] >= 2) else None
    if len(np.unique(y_enc)) > 1 and len(X) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_enc,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
    else:
        X_train, y_train = X, y_enc
        X_test, y_test = None, None

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=len(X_train) >= 30,
        validation_fraction=0.1,
        verbose=True,
    )
    model.fit(X_train, y_train)

    report_text = "Sem conjunto de teste separado."
    if X_test is not None:
        y_pred = model.predict(X_test)
        present = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
        report_text = classification_report(
            y_test,
            y_pred,
            labels=present,
            target_names=[labels_for_model[i] for i in present],
            zero_division=0,
        )
        print(report_text)

    ensure_dirs()
    if missing_labels:
        save_partial_model(name, model, labels_for_model, features, report_text, missing_labels)
        return

    save_pickle(MODELS_DIR / f"{name}_model.pkl", model)
    save_pickle(MODELS_DIR / f"{name}_classes.pkl", labels_for_model)
    general_dir = (STATIC_ASSETS if name == "static" else DYNAMIC_ASSETS) / "geral"
    general_dir.mkdir(parents=True, exist_ok=True)
    save_labels(general_dir / "labels.txt", labels_for_model)
    export_onnx_model(model, general_dir / "model.onnx", features)
    verificar_onnx_exportado(general_dir / "model.onnx", features, len(labels_for_model))
    print(f"Modelo {name} exportado e verificado para backend e Android.")


def extract_body_points(pose_result, hand_result) -> tuple[np.ndarray, bool, bool]:
    pose = np.zeros((33, 3), dtype=np.float32)
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    has_pose = bool(pose_result.pose_landmarks)
    has_hand = bool(hand_result.hand_landmarks)

    if pose_result.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]], dtype=np.float32)

    for hand_index, landmarks in enumerate(hand_result.hand_landmarks or []):
        handedness = ""
        if hand_index < len(hand_result.handedness):
            categories = hand_result.handedness[hand_index]
            if categories:
                handedness = categories[0].category_name
        hand_points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        if handedness.lower() == "left":
            left = hand_points
        elif handedness.lower() == "right":
            right = hand_points
        else:
            avg_x = float(hand_points[:, 0].mean())
            if avg_x < 0.5:
                left = hand_points
            else:
                right = hand_points
    return np.concatenate([pose, left, right], axis=0), has_pose, has_hand


# Escala minima aceita entre os ombros -- tem que ser IGUAL ao
# LibrasMath.ESCALA_MINIMA_OMBROS do Kotlin. Antes aqui era `or 1.0`, que so
# trata o zero exato: com os ombros quase colados (pose degenerada, pessoa de
# lado, ombro fora do quadro) a escala virava ~1e-5 e as features explodiam --
# um ponto a 1cm do centro virava 1000 no dataset de treino, enquanto o app
# (que ja tinha o teto) gerava 0.01. Quadros assim envenenavam o treino com
# valores que o app nunca produz.
ESCALA_MINIMA_OMBROS = 0.0001


def normalize_body_frame(frame: np.ndarray) -> np.ndarray:
    normalized = frame.copy()
    center = (frame[11] + frame[12]) / 2.0
    scale = float(np.linalg.norm(frame[11] - frame[12]))
    if scale <= ESCALA_MINIMA_OMBROS:
        scale = 1.0
    normalized[:, :2] = (normalized[:, :2] - center[:2]) / scale
    return normalized


def resample_sequence(frames: list[np.ndarray], count: int) -> np.ndarray:
    if len(frames) == count:
        return np.array(frames, dtype=np.float32)
    indexes = np.linspace(0, len(frames) - 1, count).astype(int)
    return np.array([frames[i] for i in indexes], dtype=np.float32)


def extract_body_dataset(window: int = 30) -> None:
    require_cv_stack()
    import cv2
    from mediapipe.tasks.python.vision import RunningMode

    print("Extraindo gestos corporais...")
    X, y = [], []
    pose_landmarker = create_pose_landmarker(RunningMode.VIDEO)
    hand_landmarker = create_hand_landmarker(RunningMode.VIDEO, num_hands=2)
    timestamp_ms = 0

    for label in BODY_LABELS:
        video_dir = DATA_DIR / "raw_body_videos" / label
        video_files = sorted(video_dir.glob("*")) if video_dir.exists() else []
        if video_files:
            print(f"  {label}: {len(video_files)} videos")
        for video_index, video_path in enumerate(video_files, start=1):
            if video_path.suffix.lower() not in VIDEO_EXTS:
                continue
            if video_index == 1 or video_index % 5 == 0 or video_index == len(video_files):
                print(f"    {label}: video {video_index}/{len(video_files)}")
            cap = cv2.VideoCapture(str(video_path))
            frames: list[np.ndarray] = []
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                mp_image = mp_image_from_bgr(frame)
                pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                timestamp_ms += 1
                points, has_pose, has_hand = extract_body_points(pose_result, hand_result)
                if has_pose and has_hand:
                    frames.append(normalize_body_frame(points).reshape(-1))
                frame_index += 1
            cap.release()

            if len(frames) >= 10:
                X.append(resample_sequence(frames, window))
                y.append(label)
            else:
                print(f"Ignorado corpo {video_path}: poucos frames validos ({len(frames)}).")

    pose_landmarker.close()
    hand_landmarker.close()

    if not X:
        print("Nenhuma amostra corporal gerada.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_DIR / "dataset_body.npz", X=np.array(X, dtype=np.float32), y=np.array(y))
    print(f"Dataset corporal salvo: {DATA_DIR / 'dataset_body.npz'} ({len(X)} amostras)")


def train_body_model() -> None:
    require_tensorflow()
    import tensorflow as tf

    dataset = DATA_DIR / "dataset_body.npz"
    if not dataset.exists():
        raise SystemExit(f"Dataset corporal nao encontrado: {dataset}")
    data = np.load(dataset, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = np.array([str(v).upper() for v in data["y"]])
    keep = np.isin(y, BODY_LABELS)
    X, y = X[keep], y[keep]
    if len(X) == 0:
        raise SystemExit("Sem amostras corporais validas.")

    y_enc = encode_labels_in_mobile_order(y, BODY_LABELS)
    print(f"Treinando corpo: {len(X)} amostras | { {label: int((y == label).sum()) for label in BODY_LABELS} }")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(30, 225)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(BODY_LABELS), activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(X, y_enc, epochs=80, batch_size=8, validation_split=0.2 if len(X) >= 10 else 0.0)

    gesture_general_dir = GESTURE_ASSETS / "geral"
    gesture_general_dir.mkdir(parents=True, exist_ok=True)
    save_labels(gesture_general_dir / "labels.txt", BODY_LABELS)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    (gesture_general_dir / "model.tflite").write_bytes(tflite_model)
    verificar_tflite_exportado(
        gesture_general_dir / "model.tflite",
        janela=30,
        features=225,
        n_classes=len(BODY_LABELS),
    )
    print(f"Modelo corporal Android salvo e verificado em: "
          f"{gesture_general_dir / 'model.tflite'}")


def print_status() -> None:
    print("Labels estaticos:", " ".join(STATIC_LABELS))
    print("Labels dinamicos:", " ".join(DYNAMIC_LABELS))
    print("Labels corporais:", " ".join(BODY_LABELS))
    print()
    for path in [
        DATA_DIR / "raw_images",
        DATA_DIR / "raw_static_videos",
        DATA_DIR / "raw_videos",
        DATA_DIR / "raw_body_videos",
        NEGATIVE_DIR,
    ]:
        count = len([p for p in path.rglob("*") if p.is_file()]) if path.exists() else 0
        print(f"{path}: {count} arquivos")
    print()
    for path in [
        DATA_DIR / "dataset_static.npz",
        DATA_DIR / "dataset_dynamic.npz",
        DATA_DIR / "dataset_body.npz",
        DATA_DIR / "dataset_static_negativos.npz",
        DATA_DIR / "dataset_dynamic_negativos.npz",
        STATIC_ASSETS / "geral" / "model.onnx",
        DYNAMIC_ASSETS / "geral" / "model.onnx",
        GESTURE_ASSETS / "geral" / "model.tflite",
    ]:
        print(f"{path}: {'OK' if path.exists() else 'faltando'}")


def run_extract(kinds: set[str]) -> None:
    if "static" in kinds:
        extract_static_dataset()
    if "dynamic" in kinds:
        extract_dynamic_dataset()
    if "body" in kinds:
        extract_body_dataset()
    # Os negativos alimentam os modelos individuais de letra (estatica e
    # dinamica), entao basta uma das duas categorias estar no pedido.
    if kinds & {"static", "dynamic"}:
        extract_negative_dataset()


def run_train(kinds: set[str]) -> None:
    print("Iniciando etapa de treino...")
    if "static" in kinds:
        print("Verificando treino static...")
        if (DATA_DIR / "dataset_static.npz").exists():
            train_mlp("static", 42, STATIC_LABELS, max_per_class=500)
            train_individual_mlp("static", 42, STATIC_LABELS, max_per_class=500)
        else:
            print("Pulando static: dataset_static.npz nao existe.")
    if "dynamic" in kinds:
        print("Verificando treino dynamic...")
        if (DATA_DIR / "dataset_dynamic.npz").exists():
            print("Treino dynamic geral/parcial preservado; atualizando apenas modelos individuais.")
            train_individual_mlp("dynamic", 420, DYNAMIC_LABELS, max_per_class=500)
        else:
            print("Pulando dynamic: dataset_dynamic.npz nao existe.")
    if "body" in kinds:
        print("Verificando treino body...")
        if (DATA_DIR / "dataset_body.npz").exists():
            train_body_model()
        else:
            print("Pulando body: dataset_body.npz nao existe.")
    print("Etapa de treino finalizada.")


def parse_kinds(raw: str) -> set[str]:
    if raw == "todos":
        return {"static", "dynamic", "body"}
    values = {part.strip() for part in raw.split(",") if part.strip()}
    invalid = values - {"static", "dynamic", "body"}
    if invalid:
        raise SystemExit(f"Tipos invalidos: {sorted(invalid)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Importa midias e treina modelos do VisuAll.")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Mostra labels, midias e datasets existentes.")
    status.set_defaults(func=lambda args: print_status())

    importer = sub.add_parser("importar", help="Copia fotos/videos para as pastas de treino.")
    importer.add_argument("entrada", type=Path)
    importer.add_argument("--label", help="Label quando a entrada for arquivo isolado ou pasta sem subpastas por label.")

    extractor = sub.add_parser("extrair", help="Extrai landmarks para datasets.")
    extractor.add_argument("--tipos", default="todos", help="todos, static, dynamic, body ou lista: static,dynamic")

    trainer = sub.add_parser("treinar", help="Treina modelos e exporta para o Android.")
    trainer.add_argument("--tipos", default="todos", help="todos, static, dynamic, body ou lista: static,dynamic")

    all_in_one = sub.add_parser("tudo", help="Importa, extrai e treina em uma etapa.")
    all_in_one.add_argument("entrada", type=Path)
    all_in_one.add_argument("--label", help="Label quando a entrada for arquivo isolado ou pasta sem subpastas por label.")
    all_in_one.add_argument("--tipos", default="todos", help="todos, static, dynamic, body ou lista: static,dynamic")

    args = parser.parse_args()
    start = time.time()

    if args.command == "status":
        print_status()
    elif args.command == "importar":
        stats = import_media(args.entrada, args.label)
        print(f"Importacao concluida: {stats.copied} copiados, {stats.skipped} ignorados.")
    elif args.command == "extrair":
        run_extract(parse_kinds(args.tipos))
    elif args.command == "treinar":
        run_train(parse_kinds(args.tipos))
    elif args.command == "tudo":
        kinds = parse_kinds(args.tipos)
        stats = import_media(args.entrada, args.label)
        print(f"Importacao concluida: {stats.copied} copiados, {stats.skipped} ignorados.")
        run_extract(kinds)
        run_train(kinds)

    print(f"Tempo total: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

