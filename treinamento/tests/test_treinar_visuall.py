"""Testes do motor de treino (treinar_visuall.py).

Rodar da raiz do repo:  python -m unittest discover -s treinamento/tests

Nao precisam de webcam nem de video de verdade: tudo que depende de camera
ou de MediaPipe fica de fora, e o que e testado aqui e a parte que da errado
em silencio -- roteamento de pasta por rotulo, a matematica de normalizacao
(que precisa bater com a do app Kotlin) e o uso dos exemplos negativos no
treino dos modelos individuais.

Todo teste que escreve arquivo redireciona as pastas do modulo pra um
diretorio temporario, inclusive as de assets do Android: sem isso, rodar os
testes sobrescreveria os modelos de verdade do app.
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "treinamento"))

import treinar_visuall as core  # noqa: E402


class TestRoteamentoDeMidia(unittest.TestCase):

    def test_media_kind_reconhece_imagem_e_video(self):
        self.assertEqual("image", core.media_kind(Path("foto.JPG")))
        self.assertEqual("video", core.media_kind(Path("clipe.mp4")))
        self.assertIsNone(core.media_kind(Path("anotacoes.txt")))

    def test_cada_rotulo_vai_pra_pasta_certa(self):
        casos = [
            ("A", "image", "raw_images"),
            ("A", "video", "raw_static_videos"),
            ("H", "video", "raw_videos"),
            ("AJUDAR", "video", "raw_body_videos"),
        ]
        for label, kind, esperado in casos:
            with self.subTest(label=label, kind=kind):
                destino = core.target_dir_for(label, kind)
                self.assertIsNotNone(destino)
                self.assertEqual(esperado, destino.parent.name)
                self.assertEqual(label, destino.name)

    def test_combinacao_invalida_nao_tem_destino(self):
        # Letra dinamica so faz sentido como video, gesto corporal idem.
        self.assertIsNone(core.target_dir_for("H", "image"))
        self.assertIsNone(core.target_dir_for("AJUDAR", "image"))
        self.assertIsNone(core.target_dir_for("NAO_EXISTE", "video"))

    def test_safe_label_name_tira_caractere_de_caminho(self):
        # Vira nome de pasta em assets/, entao nao pode ter barra nem ponto.
        self.assertEqual("AJUDAR", core.safe_label_name("ajudar"))
        self.assertEqual("AB", core.safe_label_name("a/b"))
        self.assertEqual("H", core.safe_label_name("../h"))

    def test_parse_kinds(self):
        self.assertEqual({"static", "dynamic", "body"}, core.parse_kinds("todos"))
        self.assertEqual({"static", "dynamic"}, core.parse_kinds("static,dynamic"))
        with self.assertRaises(SystemExit):
            core.parse_kinds("invalido")


class TestMatematica(unittest.TestCase):

    def test_normalize_hand_landmarks_translada_e_escala(self):
        # Porte do LibrasMath.normalizeLandmarks do app: subtrai o pulso
        # (ponto 0) e divide pelo maior valor absoluto.
        pontos = [(10.0, 20.0), (15.0, 20.0), (10.0, 30.0)]
        self.assertEqual(
            [0.0, 0.0, 0.5, 0.0, 0.0, 1.0],
            core.normalize_hand_landmarks(pontos),
        )

    def test_normalize_hand_landmarks_nao_divide_por_zero(self):
        self.assertEqual([0.0, 0.0], core.normalize_hand_landmarks([(5.0, 5.0)]))

    def test_normalize_body_frame_centraliza_nos_ombros(self):
        frame = np.zeros((75, 3), dtype=np.float32)
        frame[11] = (0.4, 0.5, 0.0)   # ombro esquerdo
        frame[12] = (0.6, 0.5, 0.0)   # ombro direito
        frame[0] = (0.5, 0.5, 0.0)    # ponto exatamente no centro

        normalizado = core.normalize_body_frame(frame)

        # O ponto no centro dos ombros vira a origem.
        self.assertAlmostEqual(0.0, float(normalizado[0][0]), places=5)
        self.assertAlmostEqual(0.0, float(normalizado[0][1]), places=5)
        # A escala e a distancia entre os ombros (0.2 aqui).
        self.assertAlmostEqual(-0.5, float(normalizado[11][0]), places=5)
        # z fica cru, sem normalizar (igual ao BodyGestureEngine.kt).
        self.assertAlmostEqual(0.0, float(normalizado[11][2]), places=5)

    def test_normalize_body_frame_sobrevive_a_ombros_colados(self):
        # Escala 0 dividiria tudo por zero -- o codigo cai pra 1.0.
        frame = np.zeros((75, 3), dtype=np.float32)
        normalizado = core.normalize_body_frame(frame)
        self.assertTrue(np.isfinite(normalizado).all())

    def test_resample_sequence_sempre_devolve_o_tamanho_pedido(self):
        curto = [np.full(225, i, dtype=np.float32) for i in range(12)]
        longo = [np.full(225, i, dtype=np.float32) for i in range(90)]
        exato = [np.full(225, i, dtype=np.float32) for i in range(30)]

        for frames in (curto, longo, exato):
            with self.subTest(n=len(frames)):
                saida = core.resample_sequence(frames, 30)
                self.assertEqual((30, 225), saida.shape)
                # Sempre comeca no primeiro e termina no ultimo quadro.
                self.assertAlmostEqual(float(frames[0][0]), float(saida[0][0]))
                self.assertAlmostEqual(float(frames[-1][0]), float(saida[-1][0]))


class TestPastasTemporarias(unittest.TestCase):
    """Base pros testes que escrevem arquivo: redireciona TUDO pro tmp."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="visuall_test_"))
        self._originais = {
            nome: getattr(core, nome)
            for nome in (
                "DATA_DIR", "NEGATIVE_DIR", "INDIVIDUAL_MODELS_DIR",
                "PARTIAL_MODELS_DIR", "STATIC_ASSETS", "DYNAMIC_ASSETS",
                "GESTURE_ASSETS", "MODELS_DIR",
            )
        }
        core.DATA_DIR = self.tmp / "dados"
        core.NEGATIVE_DIR = core.DATA_DIR / "raw_negativos"
        core.INDIVIDUAL_MODELS_DIR = self.tmp / "modelos_individuais"
        core.PARTIAL_MODELS_DIR = self.tmp / "modelos_parciais"
        core.STATIC_ASSETS = self.tmp / "assets" / "letras_estaticas"
        core.DYNAMIC_ASSETS = self.tmp / "assets" / "letras_dinamicas"
        core.GESTURE_ASSETS = self.tmp / "assets" / "gestos"
        core.MODELS_DIR = self.tmp / "models"
        core.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        for nome, valor in self._originais.items():
            setattr(core, nome, valor)
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestImportacao(TestPastasTemporarias):

    def test_importa_pasta_com_subpastas_por_rotulo(self):
        origem = self.tmp / "entrada"
        (origem / "A").mkdir(parents=True)
        (origem / "H").mkdir(parents=True)
        (origem / "LIXO").mkdir(parents=True)
        (origem / "A" / "foto.jpg").write_bytes(b"x")
        (origem / "H" / "clipe.mp4").write_bytes(b"x")
        (origem / "LIXO" / "coisa.mp4").write_bytes(b"x")

        stats = core.import_media(origem)

        self.assertEqual(2, stats.copied)
        self.assertEqual(1, stats.skipped)  # pasta com nome que nao e rotulo
        self.assertTrue((core.DATA_DIR / "raw_images" / "A" / "foto.jpg").exists())
        self.assertTrue((core.DATA_DIR / "raw_videos" / "H" / "clipe.mp4").exists())

    def test_importar_duas_vezes_nao_sobrescreve(self):
        origem = self.tmp / "entrada"
        (origem / "A").mkdir(parents=True)
        (origem / "A" / "foto.jpg").write_bytes(b"x")

        core.import_media(origem)
        core.import_media(origem)

        salvos = sorted(p.name for p in (core.DATA_DIR / "raw_images" / "A").iterdir())
        self.assertEqual(["foto.jpg", "foto_2.jpg"], salvos)


class TestExemplosNegativos(TestPastasTemporarias):

    def _gravar_pool(self, name, features, quantidade=40):
        X = np.random.default_rng(1).normal(size=(quantidade, features)).astype(np.float32)
        np.savez(
            core.DATA_DIR / f"dataset_{name}_negativos.npz",
            X=X,
            y=np.array([core.NEGATIVE_LABEL] * quantidade),
        )
        return X

    def test_pool_ausente_devolve_none(self):
        self.assertIsNone(core.load_negative_pool("static", 42))

    def test_pool_carrega_quando_existe(self):
        self._gravar_pool("static", 42)
        pool = core.load_negative_pool("static", 42)
        self.assertIsNotNone(pool)
        self.assertEqual((40, 42), pool.shape)

    def test_pool_com_shape_errado_e_ignorado(self):
        # Melhor ignorar que estourar no meio do treino do usuario.
        self._gravar_pool("static", 99)
        self.assertIsNone(core.load_negative_pool("static", 42))

    def test_negativos_entram_no_treino_individual(self):
        rng = np.random.default_rng(7)
        labels = ["A", "B"]
        # Duas classes bem separadas, pra o treino convergir rapido.
        X = np.concatenate([
            rng.normal(loc=0.0, scale=0.1, size=(30, 42)),
            rng.normal(loc=5.0, scale=0.1, size=(30, 42)),
        ]).astype(np.float32)
        y = np.array(["A"] * 30 + ["B"] * 30)
        np.savez(core.DATA_DIR / "dataset_static.npz", X=X, y=y)
        self._gravar_pool("static", 42, quantidade=25)

        core.train_individual_mlp("static", 42, labels, max_per_class=100)

        relatorio = (
            core.INDIVIDUAL_MODELS_DIR / "static" / "A_RELATORIO.txt"
        ).read_text(encoding="utf-8")
        # O pool tem que aparecer no relatorio E ter aumentado os negativos:
        # 30 exemplos de "B" + 25 de "nao e sinal" = 55.
        self.assertIn("no pool: 25", relatorio)
        self.assertIn("Negativos: 55", relatorio)

    def test_treino_individual_funciona_sem_pool_negativo(self):
        # Compatibilidade: quem ainda nao gravou clipes de "Nada" continua
        # conseguindo treinar, so que com modelos mais permissivos.
        rng = np.random.default_rng(7)
        X = np.concatenate([
            rng.normal(loc=0.0, scale=0.1, size=(30, 42)),
            rng.normal(loc=5.0, scale=0.1, size=(30, 42)),
        ]).astype(np.float32)
        y = np.array(["A"] * 30 + ["B"] * 30)
        np.savez(core.DATA_DIR / "dataset_static.npz", X=X, y=y)

        core.train_individual_mlp("static", 42, ["A", "B"], max_per_class=100)

        relatorio = (
            core.INDIVIDUAL_MODELS_DIR / "static" / "A_RELATORIO.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("no pool: 0", relatorio)
        self.assertIn("Negativos: 30", relatorio)

    def test_modelo_individual_e_copiado_pros_assets(self):
        rng = np.random.default_rng(7)
        X = np.concatenate([
            rng.normal(loc=0.0, scale=0.1, size=(30, 42)),
            rng.normal(loc=5.0, scale=0.1, size=(30, 42)),
        ]).astype(np.float32)
        y = np.array(["A"] * 30 + ["B"] * 30)
        np.savez(core.DATA_DIR / "dataset_static.npz", X=X, y=y)

        core.train_individual_mlp("static", 42, ["A", "B"], max_per_class=100)

        # O app carrega exatamente estes caminhos (ver LetraEngine.kt).
        self.assertTrue((core.STATIC_ASSETS / "A" / "model.onnx").exists())
        self.assertTrue((core.STATIC_ASSETS / "individual_labels.txt").exists())


if __name__ == "__main__":
    unittest.main()
