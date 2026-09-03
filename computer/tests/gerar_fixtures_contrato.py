"""Gera computer/tests/fixtures/landmark_contract.json.

Esse arquivo e o "contrato": um conjunto congelado de entradas e saidas que
as DUAS implementacoes da mesma matematica -- a do treino (Python,
computer/treino/) e a do app (Kotlin, LibrasMath.kt) -- tem que reproduzir
identicamente.

So rode isto de novo se voce mudou a matematica DE PROPOSITO nos dois lados.
Regerar pra "fazer o teste passar" derruba justamente a protecao que ele da:
o teste existe pra avisar que os dois lados sairam de sincronia, e nessa
situacao o certo e corrigir o lado errado, nao reescrever o gabarito.

Uso, dentro de computer/:  python tests/gerar_fixtures_contrato.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "treino"))

import treinar_corpo as corpo  # noqa: E402
from extrair_negativos import normalize_landmarks  # noqa: E402

SAIDA = ROOT / "tests" / "fixtures" / "landmark_contract.json"
N_POSE, N_HAND = 33, 21
N_PONTOS = N_POSE + N_HAND * 2  # 75


def arredondar(valores):
    # 6 casas: bem acima da precisao que importa (as features vivem em ~1e-2)
    # e bem abaixo do ruido de float32, entao o mesmo numero sai igual nos
    # dois lados sem depender de detalhe de arredondamento.
    return [round(float(v), 6) for v in valores]


def casos_mao():
    casos = []

    # 1) Mao "normal": pontos espalhados, valores tipicos do MediaPipe (0..1).
    rng = np.random.default_rng(20240728)
    pontos = [(float(x), float(y)) for x, y in rng.uniform(0.2, 0.8, size=(21, 2))]
    casos.append(("mao tipica", pontos))

    # 2) Todos os pontos iguais: a escala daria 0 e dividiria por zero.
    casos.append(("todos os pontos no mesmo lugar", [(0.5, 0.5)] * 21))

    # 3) Coordenadas negativas e assimetricas (mao parcialmente fora do quadro
    #    -- o MediaPipe extrapola e devolve valores fora de 0..1).
    pontos = [(0.5 - i * 0.05, -0.2 + i * 0.03) for i in range(21)]
    casos.append(("mao saindo do quadro (valores negativos)", pontos))

    return [
        {
            "nome": nome,
            "entrada": [[round(x, 6), round(y, 6)] for x, y in pontos],
            "esperado": arredondar(normalize_landmarks(pontos)),
        }
        for nome, pontos in casos
    ]


def casos_corpo():
    casos = []

    # 1) Pose normal: ombros separados, pontos espalhados.
    rng = np.random.default_rng(7)
    frame = rng.uniform(-0.5, 0.5, size=(N_PONTOS, 3)).astype(np.float32)
    frame[11] = (0.40, 0.50, 0.01)   # ombro esquerdo
    frame[12] = (0.60, 0.52, -0.01)  # ombro direito
    casos.append(("pose tipica", frame))

    # 2) Ombros quase colados: era AQUI que os dois lados divergiam. O Python
    #    dividia por ~1e-5 e gerava valores na casa dos milhares; o Kotlin ja
    #    protegia e dividia por 1. Agora os dois protegem.
    frame = np.zeros((N_PONTOS, 3), dtype=np.float32)
    frame[11] = (0.500000, 0.5, 0.0)
    frame[12] = (0.500005, 0.5, 0.0)
    frame[0] = (0.51, 0.49, 0.0)
    casos.append(("ombros quase colados (pose degenerada)", frame))

    # 3) Ombros exatamente no mesmo ponto: escala zero.
    frame = np.zeros((N_PONTOS, 3), dtype=np.float32)
    frame[0] = (0.3, 0.7, 0.2)
    casos.append(("ombros identicos (escala zero)", frame))

    # 4) Ombros a 0.002 de distancia: LOGO ACIMA do teto (0.0001), entao aqui
    #    a normalizacao acontece de verdade (divide por 0.002, nao por 1).
    #    Esse caso e o que torna o VALOR do teto observavel: sem ele, mudar o
    #    teto de 0.0001 pra 0.01 num lado so nao mudaria resultado nenhum e o
    #    teste passaria feliz com os dois lados divergindo. Sempre que um
    #    limiar entra no contrato, precisa de um caso de cada lado dele.
    frame = np.zeros((N_PONTOS, 3), dtype=np.float32)
    frame[11] = (0.499, 0.5, 0.0)
    frame[12] = (0.501, 0.5, 0.0)
    frame[0] = (0.5021, 0.4979, 0.0)
    casos.append(("ombros logo acima do teto de escala", frame))

    return [
        {
            "nome": nome,
            "entrada": arredondar(frame.reshape(-1)),
            "esperado": arredondar(corpo.normaliza_corpo(frame.reshape(-1))),
        }
        for nome, frame in casos
    ]


def casos_resample():
    # So os indices importam: o resample escolhe quadros existentes, nao
    # interpola valores. Cobre encolher, esticar e o tamanho exato.
    casos = []
    for tamanho in (10, 12, 30, 45, 60, 137):
        indices = np.linspace(0, tamanho - 1, 30).astype(int).tolist()
        casos.append({
            "tamanho_entrada": tamanho,
            "quantidade": 30,
            "indices_esperados": [int(i) for i in indices],
        })
    return casos


def main():
    fixtures = {
        "_leia_me": (
            "Contrato entre a matematica do treino (Python) e a do app (Kotlin). "
            "Gerado por computer/tests/gerar_fixtures_contrato.py. Nao edite "
            "na mao, e nao regere so pra fazer teste passar -- ver docstring "
            "do gerador."
        ),
        "normalize_hand_landmarks": casos_mao(),
        "normalize_body_frame": casos_corpo(),
        "resample": casos_resample(),
    }

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(fixtures, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Fixtures escritas em {SAIDA}")
    print(f"  mao:      {len(fixtures['normalize_hand_landmarks'])} casos")
    print(f"  corpo:    {len(fixtures['normalize_body_frame'])} casos")
    print(f"  resample: {len(fixtures['resample'])} casos")


if __name__ == "__main__":
    main()
