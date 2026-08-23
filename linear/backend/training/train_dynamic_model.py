"""
Treina MLP no dataset extraído e salva modelo + classes.

Entrada:  dataset_mlp.npz
Saída:    modelo_mlp.pkl  +  letras_mlp.pkl
"""
import numpy as np
import pickle
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Windows + console cp1252 quebra nos prints com "→"/"✅" abaixo
# (UnicodeEncodeError). Força UTF-8 na saída, sem alterar as mensagens.
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============ CONFIGURAÇÃO ============
# Caminhos ancorados no próprio arquivo, não no diretório de onde se chama o
# script: training -> backend -> linear -> raiz do repo.
RAIZ          = Path(__file__).resolve().parents[3]
DATASET       = str(RAIZ / "data" / "dataset_dynamic.npz")
SAIDA_MODELO  = str(RAIZ / "models" / "dynamic_model.pkl")
SAIDA_CLASSES = str(RAIZ / "models" / "dynamic_classes.pkl")
(RAIZ / "models").mkdir(exist_ok=True)
# ======================================

data = np.load(DATASET, allow_pickle=True)
X    = data["X"].astype(np.float32)
y    = data["y"]

# Balanceia — limita cada letra a 400 amostras
MAX_POR_CLASSE = 400
idx_balanceado = []
for letra in np.unique(y):
    idx = np.where(y == letra)[0]
    if len(idx) > MAX_POR_CLASSE:
        idx = np.random.choice(idx, MAX_POR_CLASSE, replace=False)
    idx_balanceado.extend(idx)

idx_balanceado = np.array(idx_balanceado)
X = X[idx_balanceado]
y = y[idx_balanceado]
print(f"Após balanceamento: { {l: int((y==l).sum()) for l in np.unique(y)} }")

le      = LabelEncoder()
y_enc   = le.fit_transform(y)
classes = le.classes_
print(f"Classes: {classes}")
print(f"Amostras: {len(X)}  |  Features: {X.shape[1]}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print("Treinando MLP...")
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=True
)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
print("\n" + classification_report(y_test, y_pred, target_names=classes))

with open(SAIDA_MODELO, "wb") as f:
    pickle.dump(mlp, f)
with open(SAIDA_CLASSES, "wb") as f:
    pickle.dump(classes, f)

print(f"✅ Modelo salvo em '{SAIDA_MODELO}'")
print(f"✅ Classes salvas em '{SAIDA_CLASSES}'")
