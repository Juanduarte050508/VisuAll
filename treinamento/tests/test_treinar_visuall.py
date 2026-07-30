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


class TestFiltroDeOutliers(unittest.TestCase):
    """Fixa o comportamento de filter_outlier_samples.

    Este filtro decide o que ENTRA no treino, e o erro dele e invisivel: uma
    amostra descartada nao aparece em lugar nenhum, so no modelo pior no fim.
    A primeira versao (MAD das distancias) apagava grupos legitimos inteiros --
    a partir de 60/40 entre dois enquadramentos, o grupo menor sumia 100%. Os
    testes abaixo existem pra isso nao voltar sem alguem notar.

    Ancora numerica: 42 features e a contagem de uma mao (21 pontos x, y), o
    mesmo shape que o app usa.
    """

    FEATURES = 42

    def _grupo(self, centro: np.ndarray, n: int, tremor: float, semente: int) -> np.ndarray:
        """n amostras em volta de um centro -- imita quadros de um mesmo clipe."""
        rng = np.random.default_rng(semente)
        return np.clip(
            centro + rng.normal(0.0, tremor, (n, self.FEATURES)), -1.0, 1.0
        ).astype(np.float32)

    def _centro(self, semente: int) -> np.ndarray:
        return np.random.default_rng(semente).uniform(
            -1.0, 1.0, self.FEATURES
        ).astype(np.float32)

    def _descartadas(self, X: np.ndarray) -> int:
        _, _, removidos = core.filter_outlier_samples(X, np.array(["H"] * len(X)))
        return removidos.get("H", 0)

    def test_amostra_degenerada_e_descartada(self):
        # A razao de existir do filtro: quadros em que a deteccao saiu errada
        # (mao saindo do enquadramento) viram pontos soltos longe do miolo.
        boas = self._grupo(self._centro(1), 100, tremor=0.01, semente=2)
        lixo = np.random.default_rng(3).uniform(
            -1.0, 1.0, (3, self.FEATURES)
        ).astype(np.float32)
        self.assertEqual(3, self._descartadas(np.concatenate([boas, lixo])))

    def test_dados_limpos_nao_perdem_nada(self):
        # A versao antiga caia num percentil 99 quando a dispersao era pequena,
        # o que descartava ~1% SEMPRE -- mesmo sem nenhuma amostra ruim.
        firme = self._grupo(self._centro(4), 90, tremor=0.003, semente=5)
        self.assertEqual(0, self._descartadas(firme))

    def test_grupo_minoritario_legitimo_nao_e_apagado(self):
        # O caso real: gravar a maioria dos clipes a uma distancia e o resto a
        # outra. As duas gravacoes valem, e o README pede essa variedade.
        # Antes, de 60/40 pra cima o grupo menor era apagado inteiro.
        centro = self._centro(6)
        for maioria, minoria in ((60, 40), (70, 30), (80, 20)):
            with self.subTest(proporcao=f"{maioria}/{minoria}"):
                X = np.concatenate([
                    self._grupo(centro, maioria, 0.01, semente=7),
                    self._grupo(centro * 0.6, minoria, 0.01, semente=8),
                ])
                self.assertEqual(0, self._descartadas(X))

    def test_trava_recusa_descarte_grande_em_vez_de_apagar_calado(self):
        # Com a minoria bem pequena os percentis nao a alcancam, e o filtro
        # ainda quer apaga-la. A trava de fracao e a ultima defesa: ela desiste
        # da classe inteira em vez de tirar um pedaco grande em silencio.
        centro = self._centro(9)
        X = np.concatenate([
            self._grupo(centro, 90, 0.01, semente=10),
            self._grupo(centro * 0.6, 10, 0.01, semente=11),
        ])
        self.assertEqual(0, self._descartadas(X))

    def test_ainda_limpa_lixo_quando_ha_dois_enquadramentos(self):
        """O caso que separa a dispersao por percentis da dispersao por MAD.

        A trava de fracao sozinha ja evita a PERDA de dados: com o MAD antigo
        ela recusava o descarte inteiro e o grupo minoritario sobrevivia. Mas
        recusar tem um preco -- o lixo de verdade tambem fica, e o usuario leva
        um aviso assustador a cada treino, so por ter gravado de duas
        distancias.

        Com percentis o filtro continua FUNCIONANDO nessa situacao: tira as
        amostras degeneradas e mantem os dois grupos. Sem este teste, os
        outros passam igual com o algoritmo antigo -- foi verificado
        reinjetando o MAD de proposito.
        """
        centro = self._centro(19)
        X = np.concatenate([
            self._grupo(centro, 80, 0.01, semente=20),        # enquadramento 1
            self._grupo(centro * 0.6, 20, 0.01, semente=21),  # enquadramento 2
            np.random.default_rng(22).uniform(               # deteccao ruim
                -1.0, 1.0, (3, self.FEATURES)
            ).astype(np.float32),
        ])
        _, _, removidos = core.filter_outlier_samples(X, np.array(["H"] * len(X)))
        self.assertEqual(
            3, removidos.get("H", 0),
            "esperado tirar so as 3 degeneradas, mantendo os dois "
            "enquadramentos legitimos",
        )

    def test_calculo_em_blocos_da_o_mesmo_resultado(self):
        # A distancia ao vizinho e calculada em blocos de linhas pra limitar
        # memoria, e cada bloco precisa marcar a diagonal na coluna GLOBAL
        # certa. Um erro de indice ali faria a amostra virar vizinha de si
        # mesma (distancia 0) sem quebrar nada visivelmente.
        X = self._grupo(self._centro(23), 50, 0.05, semente=24)
        inteiro = core._distancia_ao_kesimo_vizinho(X, 3, bloco=len(X))
        picado = core._distancia_ao_kesimo_vizinho(X, 3, bloco=7)
        np.testing.assert_allclose(inteiro, picado, rtol=1e-6)
        # Distancia 0 significaria que a amostra se contou como vizinha.
        self.assertTrue(np.all(picado > 0.0))

    def test_teto_de_descarte_e_o_documentado(self):
        # Se este valor mudar, o equilibrio dos testes acima muda junto --
        # entao ele e lido daqui, nao repetido na mao.
        self.assertEqual(0.05, core.LIMITE_FRACAO_DESCARTE)

    def test_classe_com_poucas_amostras_nao_e_filtrada(self):
        # Abaixo de min_samples nao ha estatistica pra confiar, entao o filtro
        # nao roda -- e o lixo passa. Documentado de proposito: e melhor que
        # inventar um limite com 7 pontos.
        boas = self._grupo(self._centro(12), 6, tremor=0.01, semente=13)
        lixo = np.random.default_rng(14).uniform(
            -1.0, 1.0, (1, self.FEATURES)
        ).astype(np.float32)
        self.assertEqual(0, self._descartadas(np.concatenate([boas, lixo])))

    def test_filtro_nao_mistura_classes(self):
        # Cada letra e avaliada contra o proprio miolo. Se as classes fossem
        # medidas juntas, a letra menos frequente viraria "outlier" em bloco.
        a = self._grupo(self._centro(15), 40, 0.01, semente=16)
        b = self._grupo(self._centro(17), 40, 0.01, semente=18)
        X = np.concatenate([a, b])
        y = np.array(["A"] * 40 + ["B"] * 40)
        Xf, yf, removidos = core.filter_outlier_samples(X, y)
        self.assertEqual({}, removidos)
        self.assertEqual(80, len(Xf))


class TestDeduplicacao(unittest.TestCase):

    def test_extrair_duas_vezes_nao_duplica_o_dataset(self):
        # A extracao reprocessa TODOS os videos da pasta e o resultado e
        # somado ao dataset que ja existia. Sem deduplicar, rodar "extrair"
        # duas vezes dobraria o peso de cada amostra no treino.
        rng = np.random.default_rng(21)
        X = rng.normal(0.0, 0.1, (50, 42)).astype(np.float32)
        y = np.array(["H"] * 50)
        Xd, yd = core.deduplicate_samples(
            np.concatenate([X, X]), np.concatenate([y, y])
        )
        self.assertEqual(50, len(Xd))
        self.assertEqual(50, len(yd))

    def test_amostras_de_classes_diferentes_nao_se_anulam(self):
        # Mesmos valores em rotulos diferentes sao amostras diferentes: a
        # chave da deduplicacao precisa incluir o rotulo.
        X = np.zeros((4, 42), dtype=np.float32)
        y = np.array(["H", "H", "J", "J"])
        Xd, yd = core.deduplicate_samples(X, y)
        self.assertEqual(2, len(Xd))
        self.assertEqual({"H", "J"}, set(yd))


if __name__ == "__main__":
    unittest.main()
