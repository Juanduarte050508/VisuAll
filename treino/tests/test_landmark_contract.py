"""Trava o lado Python no contrato de tests/fixtures/landmark_contract.json.

O gemeo deste arquivo e LandmarkContractTest.kt, que roda os MESMOS casos
contra a implementacao Kotlin do app. Enquanto os dois passarem, treino e
inferencia estao preparando os dados do mesmo jeito.

Se este teste falhar depois de voce mexer na matematica, NAO regere as
fixtures: ou a mudanca foi sem querer (corrija o codigo), ou foi de proposito
e o lado Kotlin precisa acompanhar antes de o gabarito ser refeito.

As tres funcoes cobertas vivem espalhadas pelo treino/ porque cada uma nasceu
onde era usada; o que as une e serem gemeas de LibrasMath.kt:

    normalize_landmarks  (extrair_negativos.py) <-> LibrasMath.normalizeLandmarks
    normaliza_corpo      (treinar_corpo.py)     <-> LibrasMath.normalizeBodyFrame
    reamostra            (treinar_corpo.py)     <-> LibrasMath.resample
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "treino"))

import treinar_corpo as corpo  # noqa: E402
from extrair_negativos import normalize_landmarks  # noqa: E402

FIXTURES = json.loads(
    (ROOT / "tests" / "fixtures" / "landmark_contract.json").read_text(encoding="utf-8")
)
# Mesma tolerancia usada no lado Kotlin. Folgada o bastante pra absorver a
# diferenca entre float64 (numpy) e float32 (Kotlin), apertada o bastante pra
# pegar qualquer mudanca real de formula.
TOLERANCIA = 1e-5


class TestContratoLandmarks(unittest.TestCase):

    def _comparar(self, esperado, obtido, contexto):
        self.assertEqual(len(esperado), len(obtido), f"{contexto}: tamanho diferente")
        for i, (e, o) in enumerate(zip(esperado, obtido)):
            self.assertAlmostEqual(
                e, float(o), delta=TOLERANCIA,
                msg=f"{contexto}: posicao {i} esperava {e}, obteve {o}",
            )

    def test_normalize_hand_landmarks(self):
        casos = FIXTURES["normalize_hand_landmarks"]
        self.assertTrue(casos, "fixtures de mao vazias")
        for caso in casos:
            with self.subTest(caso["nome"]):
                pontos = [(x, y) for x, y in caso["entrada"]]
                self._comparar(
                    caso["esperado"], normalize_landmarks(pontos), caso["nome"]
                )

    def test_normalize_body_frame(self):
        casos = FIXTURES["normalize_body_frame"]
        self.assertTrue(casos, "fixtures de corpo vazias")
        for caso in casos:
            with self.subTest(caso["nome"]):
                # normaliza_corpo trabalha com o quadro achatado (225 floats),
                # nao com a matriz (75, 3) -- e o mesmo layout que o Kotlin usa.
                frame = np.array(caso["entrada"], dtype=np.float32)
                self._comparar(caso["esperado"], corpo.normaliza_corpo(frame), caso["nome"])

    def test_resample(self):
        casos = FIXTURES["resample"]
        self.assertTrue(casos, "fixtures de resample vazias")
        for caso in casos:
            with self.subTest(n=caso["tamanho_entrada"]):
                # Quadros identificaveis: cada um carrega o proprio indice,
                # entao a saida revela quais quadros foram escolhidos.
                frames = [
                    np.full(3, i, dtype=np.float32) for i in range(caso["tamanho_entrada"])
                ]
                saida = corpo.reamostra(frames, caso["quantidade"])
                indices = [int(linha[0]) for linha in saida]
                self.assertEqual(caso["indices_esperados"], indices, caso["tamanho_entrada"])

    def test_guarda_de_escala_bate_com_o_kotlin(self):
        # O valor tem que ser identico ao LibrasMath.ESCALA_MINIMA_OMBROS.
        # Le direto do fonte Kotlin em vez de repetir o numero aqui: assim,
        # mudar la sem mudar aqui quebra o teste.
        fonte = (
            ROOT / "mobile" / "app" / "src" / "main" / "java" / "com" / "visuall"
            / "app" / "libras" / "LibrasMath.kt"
        ).read_text(encoding="utf-8")
        linha = next(
            l for l in fonte.splitlines() if "ESCALA_MINIMA_OMBROS" in l and "const" in l
        )
        valor_kotlin = float(linha.split("=")[1].strip().rstrip("f"))
        self.assertAlmostEqual(corpo.ESCALA_MINIMA_OMBROS, valor_kotlin, places=9)


if __name__ == "__main__":
    unittest.main()
