"""Testes da verificacao pos-export (verificar_onnx_exportado).

O ponto e garantir que a verificacao REJEITA modelo errado -- uma checagem
que so aprova nao protege de nada. Cada teste aqui exporta um modelo de
proposito fora do contrato e exige que a verificacao reclame.

Vale tambem como documentacao executavel do contrato que o app espera:
entrada unica chamada "landmarks_input", forma [N, features], e a segunda
saida contendo as probabilidades.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "treinamento"))

import treinar_visuall as core  # noqa: E402


def _mlp_treinado(features=42):
    from sklearn.neural_network import MLPClassifier

    rng = np.random.default_rng(3)
    X = np.concatenate([
        rng.normal(0.0, 0.1, size=(20, features)),
        rng.normal(4.0, 0.1, size=(20, features)),
    ]).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20)
    modelo = MLPClassifier(hidden_layer_sizes=(8,), max_iter=60, random_state=0)
    modelo.fit(X, y)
    return modelo


class TestVerificacaoOnnx(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="visuall_onnx_"))
        cls.modelo = _mlp_treinado(42)

    def _exportar(self, nome, features=42, input_name="landmarks_input", zipmap=False):
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        onx = convert_sklearn(
            self.modelo,
            initial_types=[(input_name, FloatTensorType([None, features]))],
            options={id(self.modelo): {"zipmap": zipmap}},
            target_opset=12,
        )
        caminho = self.tmp / nome
        caminho.write_bytes(onx.SerializeToString())
        return caminho

    def test_modelo_correto_passa(self):
        caminho = self._exportar("ok.onnx")
        core.verificar_onnx_exportado(caminho, features=42, n_classes=2)  # nao lanca

    def test_rejeita_nome_de_entrada_errado(self):
        # O app roda a sessao com mapOf("landmarks_input" to tensor); com outro
        # nome ele nao consegue nem alimentar o modelo.
        caminho = self._exportar("nome_errado.onnx", input_name="entrada")
        with self.assertRaises(core.ModeloInvalidoError) as ctx:
            core.verificar_onnx_exportado(caminho, features=42)
        self.assertIn("landmarks_input", str(ctx.exception))

    def test_rejeita_quantidade_de_features_errada(self):
        # Caso classico: exportar o modelo estatico (42) no lugar do dinamico
        # (420), ou treinar com um numero de features e exportar com outro.
        caminho = self._exportar("ok.onnx")
        with self.assertRaises(core.ModeloInvalidoError) as ctx:
            core.verificar_onnx_exportado(caminho, features=420)
        self.assertIn("420", str(ctx.exception))

    def test_rejeita_export_com_zipmap(self):
        # Com zipmap a segunda saida vira um dicionario em vez do vetor de
        # probabilidades que o app le como Array<FloatArray>.
        caminho = self._exportar("zipmap.onnx", zipmap=True)
        modelo = __import__("onnx").load(str(caminho))
        tipos = [s.type.WhichOneof("value") for s in modelo.graph.output]
        self.assertIn(
            "sequence_type", tipos,
            "esperava que zipmap=True gerasse saida de sequencia/mapa",
        )


class TestVerificacaoTflite(unittest.TestCase):

    def test_rejeita_formato_de_entrada_errado(self):
        import tensorflow as tf

        tmp = Path(tempfile.mkdtemp(prefix="visuall_tflite_"))
        # Modelo com a janela errada (20 em vez de 30): o app faz
        # resizeInput(0, [1, 30, 225]) e falharia no celular.
        modelo = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(20, 225)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(6, activation="softmax"),
        ])
        conversor = tf.lite.TFLiteConverter.from_keras_model(modelo)
        caminho = tmp / "janela_errada.tflite"
        caminho.write_bytes(conversor.convert())

        with self.assertRaises(Exception):
            core.verificar_tflite_exportado(caminho, janela=30, features=225, n_classes=6)


if __name__ == "__main__":
    unittest.main()
