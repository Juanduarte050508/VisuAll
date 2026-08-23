"""
Treina e exporta os modelos NO FORMATO QUE O APP ANDROID LE (.onnx).

Este e o passo que fecha o ciclo: videos/fotos no PC -> .onnx -> app do
celular. Nao passa por .pkl e nao tem nada a ver com o backend Python.

Entrada (gerada pelos extractors deste repo):
    data/dataset_static.npz    X [N, 42]   y [N]
    data/dataset_dynamic.npz   X [N, 420]  y [N]

Saida (direto na pasta de assets do app):
    mobile/app/src/main/assets/letras_estaticas/geral/model.onnx
    mobile/app/src/main/assets/letras_estaticas/geral/labels.txt
    mobile/app/src/main/assets/letras_dinamicas/geral/model.onnx
    mobile/app/src/main/assets/letras_dinamicas/geral/labels.txt

CONTRATO com o app -- estes 4 pontos nao podem mudar (LetraEngine.kt:234-242
e mobile/TREINAMENTO_MODELOS.md):

  1. nome da entrada .... "landmarks_input"
  2. shape da entrada ... [1, 42] (estatico) / [1, 420] (dinamico), float32
  3. saida .............. zipmap=False, entao out[0]=label e
                          out[1]=probabilidades. O Kotlin le out[1].
  4. labels.txt ......... uma letra por linha, NA MESMA ORDEM das colunas de
                          probabilidade. E o que traduz "saida numero 3" na
                          letra certa.

A normalizacao dos landmarks ja bate: normalize_landmarks() do Python e
LibrasMath.normalizeLandmarks() do Kotlin fazem a mesma conta (translada pelo
ponto 0, divide pelo maior valor absoluto).

Uso:
    python treino/exportar_onnx.py                  # os dois modelos
    python treino/exportar_onnx.py --tipo estatico  # so um
    python treino/exportar_onnx.py --limpar-individuais
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
ASSETS = RAIZ / "mobile" / "app" / "src" / "main" / "assets"

NOME_ENTRADA = "landmarks_input"

# Serve pra saber a que categoria pertence cada letra pedida em --reforcar.
LETRAS_DO_TIPO = {
    "estatico": list("ABCDEFGILMNOPQRSTUVWY"),
    "dinamico": ["H", "J", "K", "X", "Z"],
}

# (dataset, features, pasta de assets, maximo de amostras por classe)
CONFIG = {
    "estatico": (DATA / "dataset_static.npz", 42, ASSETS / "letras_estaticas", 500),
    "dinamico": (DATA / "dataset_dynamic.npz", 420, ASSETS / "letras_dinamicas", 400),
}


def exige(modulo, pacote=None):
    try:
        return __import__(modulo)
    except ImportError:
        print("ERRO: falta o pacote '%s'." % (pacote or modulo))
        print("Instale com:  python -m pip install %s" % (pacote or modulo))
        raise SystemExit(1)


def balanceia(X, y, maximo):
    """Limita cada classe a `maximo` amostras -- mesma logica dos scripts de treino."""
    indices = []
    for classe in np.unique(y):
        idx = np.where(y == classe)[0]
        if len(idx) > maximo:
            idx = np.random.RandomState(42).choice(idx, maximo, replace=False)
        indices.extend(idx)
    indices = np.array(indices)
    return X[indices], y[indices]


def treina(X, y):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(le.classes_)

    print("  classes: %s" % classes)
    print("  amostras: %d  |  features: %d" % (len(X), X.shape[1]))
    for c in classes:
        print("     %-4s %d" % (c, int((y == c).sum())))

    if len(classes) < 2:
        print("\n  ERRO: so ha 1 classe com dados. O modelo geral precisa de pelo")
        print("  menos 2 letras pra ter o que distinguir. Grave outra letra.")
        raise SystemExit(1)

    # stratify exige >= 2 amostras por classe; com pouca amostra, desliga.
    minimo = min(int((y_enc == i).sum()) for i in range(len(classes)))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, random_state=42,
        stratify=y_enc if minimo >= 5 else None,
    )

    print("\n  treinando MLP (256, 128)...")
    modelo = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=minimo >= 10,   # precisa de amostra pra separar validacao
        validation_fraction=0.1,
    )
    modelo.fit(X_tr, y_tr)

    y_pred = modelo.predict(X_te)
    print("\n" + classification_report(
        y_te, y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
    ))
    return modelo, classes


def exporta_onnx(modelo, n_features, destino):
    """Converte pro .onnx no contrato que o LetraEngine espera."""
    exige("skl2onnx")
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    tipos = [(NOME_ENTRADA, FloatTensorType([None, n_features]))]
    # zipmap=False e o que faz a saida virar [label, probabilidades] em vez de
    # uma lista de dicionarios. O Kotlin le out[1] como Array<FloatArray>;
    # com zipmap ligado, aquele indice vem como mapa e o app quebra.
    onx = convert_sklearn(
        modelo, initial_types=tipos, target_opset=12,
        options={id(modelo): {"zipmap": False}},
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(onx.SerializeToString())
    return destino


def salva_labels(classes, destino):
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(classes) + "\n", encoding="utf-8")
    return destino


def valida(caminho_onnx, n_features, n_classes):
    """Roda o .onnx recem-gerado do jeito que o app roda. Falha aqui e melhor
    que descobrir com o app instalado e sem reconhecer nada."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  AVISO: onnxruntime nao instalado -- pulando a validacao.")
        print("         Instale com: python -m pip install onnxruntime")
        return False

    sessao = ort.InferenceSession(str(caminho_onnx), providers=["CPUExecutionProvider"])

    entradas = sessao.get_inputs()
    if len(entradas) != 1 or entradas[0].name != NOME_ENTRADA:
        print("  FALHOU: entrada devia se chamar '%s', veio %s"
              % (NOME_ENTRADA, [e.name for e in entradas]))
        return False

    forma = entradas[0].shape
    if len(forma) != 2 or forma[1] != n_features:
        print("  FALHOU: shape da entrada devia ser [1, %d], veio %s" % (n_features, forma))
        return False

    saidas = sessao.get_outputs()
    if len(saidas) < 2:
        print("  FALHOU: o app le a saida de indice [1] (probabilidades), mas o")
        print("          modelo tem so %d saida(s). Faltou zipmap=False?" % len(saidas))
        return False

    amostra = np.zeros((1, n_features), dtype=np.float32)
    resultado = sessao.run(None, {NOME_ENTRADA: amostra})
    probs = np.asarray(resultado[1])

    if probs.shape != (1, n_classes):
        print("  FALHOU: probabilidades deviam ter shape (1, %d), veio %s"
              % (n_classes, probs.shape))
        return False
    soma = float(probs.sum())
    if not (0.98 <= soma <= 1.02):
        print("  FALHOU: as probabilidades somam %.3f (deviam somar 1)." % soma)
        return False

    print("  validado: entrada '%s' %s, saida[1] %s, soma=%.3f"
          % (NOME_ENTRADA, forma, probs.shape, soma))
    return True


def limpa_individuais(pasta_assets, classes):
    """Remove os modelos por letra. O LetraEngine tenta os individuais ANTES do
    geral (LetraEngine.kt:209-219), entao um individual velho continua mandando
    mesmo depois de voce treinar o geral."""
    removidos = []
    for sub in sorted(pasta_assets.iterdir()):
        if sub.is_dir() and sub.name != "geral":
            shutil.rmtree(sub)
            removidos.append(sub.name)
    return removidos


def carrega_negativos(tipo, n_features):
    """Amostras de 'Nada' (mao a mostra sem sinalizar), se houver."""
    caminho = DATA / ("dataset_negativos_static.npz" if tipo == "estatico"
                      else "dataset_negativos_dynamic.npz")
    if not caminho.exists():
        return None
    X = np.load(caminho)["X"].astype(np.float32)
    return X if X.ndim == 2 and X.shape[1] == n_features else None


def treina_individuais(tipo, X, y, n_features, pasta_assets, alvos):
    """Treina um modelo SIM/NAO por letra e salva em <LETRA>/model.onnx.

    O app testa estes ANTES do modelo geral (LetraEngine.kt:209-219), entao
    reforcar uma letra aqui NAO mexe em nenhuma outra -- e por isso este e o
    caminho pra aprimorar letras sem regravar o alfabeto inteiro.

    Cada modelo responde "isto e a letra L?". Os exemplos de "nao" sao as
    outras letras gravadas MAIS os clipes de 'Nada'. Sem os de 'Nada', o
    modelo so aprende a separar L de outras letras e passa a dizer "sim" pra
    qualquer mao solta no ar.
    """
    from sklearn.neural_network import MLPClassifier

    negativos_nada = carrega_negativos(tipo, n_features)
    disponiveis = sorted(set(y))
    print("\n  --- modelos individuais (%s) ---" % tipo)
    if negativos_nada is None:
        print("  AVISO: sem clipes de 'Nada'. Os modelos individuais vao ficar")
        print("  propensos a falso positivo. Grave alguns no modo 'nada' do")
        print("  Gravar.bat e rode de novo -- e a melhoria mais barata que tem.")
    else:
        print("  usando %d amostras de 'Nada' como exemplo negativo"
              % len(negativos_nada))

    feitos = []
    for letra in alvos:
        if letra not in disponiveis:
            print("  %-3s pulado: voce nao gravou amostras dessa letra." % letra)
            continue

        positivos = X[y == letra]
        outras = X[y != letra]
        partes = [p for p in (outras, negativos_nada) if p is not None and len(p)]
        if not partes:
            print("  %-3s pulado: nao ha nenhum exemplo de 'nao' pra comparar." % letra)
            continue
        negativos = np.vstack(partes)

        if len(positivos) < 10:
            print("  %-3s pulado: so %d amostras (minimo 10)." % (letra, len(positivos)))
            continue

        # Equilibra os dois lados: um modelo treinado com 20 "sim" e 2000 "nao"
        # aprende a responder sempre "nao".
        rs = np.random.RandomState(42)
        limite = min(len(negativos), max(len(positivos) * 3, 60))
        if len(negativos) > limite:
            negativos = negativos[rs.choice(len(negativos), limite, replace=False)]

        X_bin = np.vstack([positivos, negativos]).astype(np.float32)
        y_bin = np.concatenate([np.ones(len(positivos), dtype=int),
                                np.zeros(len(negativos), dtype=int)])

        modelo = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", max_iter=500,
            random_state=42, early_stopping=len(X_bin) >= 50, validation_fraction=0.15,
        )
        modelo.fit(X_bin, y_bin)
        acerto = modelo.score(X_bin, y_bin)

        destino = pasta_assets / letra / "model.onnx"
        exporta_onnx(modelo, n_features, destino)
        ok = valida(destino, n_features, 2)
        print("  %-3s %d sim / %d nao  |  acerto %.0f%%  |  %s"
              % (letra, len(positivos), len(negativos), acerto * 100,
                 "ok" if ok else "NAO VALIDOU"))
        feitos.append(letra)

    return feitos


def checa_regressao(classes, pasta_assets, forcar):
    """O modelo geral so conhece as letras com que foi treinado. Treinar com um
    punhado de letras SUBSTITUI o modelo do app por um que esqueceu todas as
    outras -- e o labels.txt vai junto, entao nem da erro: o app simplesmente
    para de reconhecer o que reconhecia. Aqui a gente barra isso."""
    labels_path = pasta_assets / "geral" / "labels.txt"
    if not labels_path.exists():
        return True

    atuais = [l.strip() for l in labels_path.read_text(encoding="utf-8").splitlines()
              if l.strip()]
    faltando = sorted(set(atuais) - set(classes))
    if not faltando:
        return True

    print("\n  " + "!" * 58)
    print("  PAREI: isto faria o app ESQUECER letras que ele ja sabe.")
    print("  " + "!" * 58)
    print("  o app conhece hoje (%d): %s" % (len(atuais), " ".join(atuais)))
    print("  voce gravou so     (%d): %s" % (len(classes), " ".join(classes)))
    print("  seriam PERDIDAS    (%d): %s" % (len(faltando), " ".join(faltando)))
    print()
    print("  O modelo geral so sabe as letras com que foi treinado. Se ele for")
    print("  substituido por um treinado so com %s, o app para de" % " ".join(classes))
    print("  reconhecer as outras %d -- sem dar erro nenhum." % len(faltando))
    print()
    print("  O que fazer:")
    print("    -> grave tambem as letras que faltam e rode de novo (recomendado)")
    print("    -> ou, se voce REALMENTE quer um app que so conhece essas,")
    print("       rode com --forcar")

    if forcar:
        print("\n  --forcar usado: seguindo mesmo assim.")
        return True
    return False


def processa(tipo, limpar, forcar=False, reforcar=None):
    dataset, n_features, pasta_assets, maximo = CONFIG[tipo]

    print("\n" + "=" * 62)
    print("  %s  (%d features)" % (tipo.upper(), n_features))
    print("=" * 62)

    if not dataset.exists():
        print("  pulando: %s nao existe." % dataset.relative_to(RAIZ))
        print("  rode os extractors antes (treino\\Treinar.bat faz isso).")
        return None

    dados = np.load(dataset, allow_pickle=True)
    X = dados["X"].astype(np.float32)
    y = dados["y"]

    if X.ndim != 2 or X.shape[1] != n_features:
        print("  ERRO: o dataset tem shape %s, mas o app espera [N, %d]."
              % (X.shape, n_features))
        return None

    X, y = balanceia(X, y, maximo)

    # Modo reforco: mexe SO nos modelos individuais das letras pedidas.
    # O modelo geral (e o labels.txt) fica exatamente como estava, entao
    # nenhuma outra letra e afetada.
    if reforcar is not None:
        alvos = [l for l in reforcar if l in set(LETRAS_DO_TIPO[tipo])]
        if not alvos:
            print("  nenhuma das letras pedidas e desta categoria -- pulando.")
            return None
        feitos = treina_individuais(tipo, X, y, n_features, pasta_assets, alvos)
        if not feitos:
            print("\n  Nenhum modelo individual foi gerado.")
            return False
        print("\n  reforcadas: %s" % " ".join(feitos))
        print("  o modelo geral NAO foi tocado -- as outras letras continuam iguais.")
        return True

    modelo, classes = treina(X, y)

    if not checa_regressao(classes, pasta_assets, forcar):
        return "bloqueado"

    destino_modelo = pasta_assets / "geral" / "model.onnx"
    destino_labels = pasta_assets / "geral" / "labels.txt"

    exporta_onnx(modelo, n_features, destino_modelo)
    salva_labels(classes, destino_labels)

    print("  gerado: %s (%.0f KB)"
          % (destino_modelo.relative_to(RAIZ), destino_modelo.stat().st_size / 1024))
    print("  gerado: %s -> %s"
          % (destino_labels.relative_to(RAIZ), " ".join(classes)))

    ok = valida(destino_modelo, n_features, len(classes))

    if limpar:
        removidos = limpa_individuais(pasta_assets, classes)
        if removidos:
            print("  removidos modelos individuais: %s" % ", ".join(removidos))
    else:
        individuais = [p.name for p in pasta_assets.iterdir()
                       if p.is_dir() and p.name != "geral"]
        if individuais:
            print("\n  ATENCAO: existem modelos individuais aqui: %s"
                  % ", ".join(individuais))
            print("  O app testa os individuais ANTES do geral, entao essas letras")
            print("  vao continuar usando o modelo antigo. Use --limpar-individuais")
            print("  se quiser que o modelo que voce acabou de treinar valha.")

    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Treina e exporta os modelos .onnx do app Android.")
    ap.add_argument("--tipo", choices=["estatico", "dinamico", "todos"],
                    default="todos", help="qual modelo gerar (padrao: todos)")
    ap.add_argument("--limpar-individuais", action="store_true",
                    help="apaga os modelos por letra, que tem precedencia sobre o geral")
    ap.add_argument("--forcar", action="store_true",
                    help="exporta mesmo que o app perca letras que ja conhecia")
    ap.add_argument("--reforcar", metavar="LETRAS",
                    help="aprimora SO estas letras (ex: --reforcar E,F,G), gerando "
                         "modelos individuais e sem tocar no modelo geral")
    args = ap.parse_args()

    if not ASSETS.exists():
        print("ERRO: nao achei a pasta de assets do app em %s" % ASSETS)
        return 1

    reforcar = None
    if args.reforcar:
        # Aceita as formas que alguem digita na pratica: "E,F,G", "E F G",
        # "e;f;g" e com aspas sobrando do .bat.
        bruto = args.reforcar.strip().strip('"').strip("'")
        for separador in (";", " "):
            bruto = bruto.replace(separador, ",")
        reforcar = [x.strip().upper() for x in bruto.split(",") if x.strip()]
        conhecidas = set(LETRAS_DO_TIPO["estatico"]) | set(LETRAS_DO_TIPO["dinamico"])
        desconhecidas = [l for l in reforcar if l not in conhecidas]
        if desconhecidas:
            print("ERRO: nao conheco a(s) letra(s): %s" % " ".join(desconhecidas))
            print("Validas: %s" % " ".join(sorted(conhecidas)))
            return 1
        print("Modo REFORCO: %s (o modelo geral nao sera alterado)" % " ".join(reforcar))

    tipos = ["estatico", "dinamico"] if args.tipo == "todos" else [args.tipo]
    resultados = {t: processa(t, args.limpar_individuais, args.forcar, reforcar)
                  for t in tipos}

    print("\n" + "=" * 62)
    for t, r in resultados.items():
        if r is None:
            rotulo = "sem dados (nao gravou ainda)"
        elif r == "bloqueado":
            rotulo = "NAO EXPORTADO - perderia letras (veja acima)"
        elif r:
            rotulo = "OK - modelo novo no app"
        else:
            rotulo = "gerado, mas nao passou na validacao"
        print("  %-9s %s" % (t, rotulo))
    print("=" * 62)

    exportou = [t for t, r in resultados.items() if r is True]
    if not exportou:
        print("\nNenhum modelo novo foi gerado -- o app continua como estava.")
        return 1

    print("\nProximo passo -- instalar o app com os modelos novos:")
    print("  pelo Android Studio: abra a pasta 'mobile' e clique em Run")
    print("  ou pelo terminal:    cd mobile  &&  .\\gradlew.bat assembleDebug")
    return 0


if __name__ == "__main__":
    sys.exit(main())
