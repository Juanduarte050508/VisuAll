import pickle # Pickle serializa em binario, ou seja, transforma o que foi treinado no modelo em
#-------------# "Linguagem de computador" o tal do binário (0 e 1) em um arquivo ".pkl"
from pathlib import Path

import numpy as np # numpy, importado como (as) np, manipula matrizes (imagem é matriz de pixel)
from tensorflow.keras.models import load_model #tensorflow simplifica treino de IA
#----------------------------------------------# "load_model" carrega o modelo salvo

from m01_visuall_config import BASE_DIR, P #Importa a função P do arquivo 1 (configurações e variáveis)

MODELS_DIR = Path(BASE_DIR).parent / "models"

def carrega_pickle(caminho): #Cria a função para carregar o treino do modelo (caminho = arquivo)
    with open(caminho, "rb") as f: # with (fecha o arquivo depois de executar)
        #--------------------------# open() é uma função que apenas abre o arquivo no "caminho"
        #--------------------------# "rb" significa read binary, e o arquivo pickle vem em binário
        return pickle.load(f) 


print("🔄 Carregando modelos...")

try:
    modelo_estatico = carrega_pickle(MODELS_DIR / "static_model.pkl")
    classes_estatico = carrega_pickle(MODELS_DIR / "static_classes.pkl")
    print(f"✅ MLP estático ({len(classes_estatico)} letras)")
except Exception as e:
    modelo_estatico, classes_estatico = None, []
    print(f"⚠ MLP estático indisponível: {e}")

try:
    modelo_mlp = carrega_pickle(MODELS_DIR / "dynamic_model.pkl")
    classes_mlp = carrega_pickle(MODELS_DIR / "dynamic_classes.pkl")
    print(f"✅ MLP dinâmico ({len(classes_mlp)} letras)")
except Exception as e:
    modelo_mlp, classes_mlp = None, []
    print(f"⚠ MLP dinâmico indisponível: {e}")

try:
    modelo_corpo = load_model(P("modelo_sinais.h5"))
    classes_corpo = np.load(P("classes_sinais.npy"), allow_pickle=True)
    print(f"✅ Modelo de corpo ({len(classes_corpo)} sinais: {list(classes_corpo)})")
except Exception as e:
    modelo_corpo, classes_corpo = None, []
    print(f"⚠ Modelo de corpo indisponível: {e}")
