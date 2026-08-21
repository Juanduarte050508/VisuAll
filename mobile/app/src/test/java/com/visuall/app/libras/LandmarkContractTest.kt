package com.visuall.app.libras

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Trava o lado Kotlin no contrato de tests/fixtures/landmark_contract.json.
 *
 * O gêmeo deste arquivo é treinamento/tests/test_landmark_contract.py, que
 * roda os MESMOS casos contra a implementação Python usada no treino.
 * Enquanto os dois passarem, o modelo é treinado e usado com os dados
 * preparados exatamente do mesmo jeito.
 *
 * Se este teste falhar depois de você mexer em LibrasMath, NÃO regere as
 * fixtures: ou a mudança foi sem querer (corrija o código), ou foi de
 * propósito e o Python precisa acompanhar antes do gabarito ser refeito.
 */
class LandmarkContractTest {

    // Folgada o bastante pra absorver float32 (Kotlin) vs float64 (numpy),
    // apertada o bastante pra pegar qualquer mudança real de fórmula.
    private val tolerancia = 1e-5f

    private val fixtures: JSONObject by lazy {
        JSONObject(acharFixtures().readText())
    }

    // O diretório de trabalho dos testes JVM depende de como o Gradle é
    // invocado, então sobe a árvore até achar o arquivo em vez de assumir um
    // caminho relativo fixo.
    private fun acharFixtures(): File {
        val raiz = System.getProperty("user.dir") ?: "."
        var dir: File? = File(raiz)
        while (dir != null) {
            val candidato = File(dir, "tests/fixtures/landmark_contract.json")
            if (candidato.isFile) return candidato
            dir = dir.parentFile
        }
        throw IllegalStateException(
            "tests/fixtures/landmark_contract.json não encontrado a partir de $raiz" +
                " — rode 'python tests/gerar_fixtures_contrato.py' na raiz do repo."
        )
    }

    private fun JSONArray.toFloatList(): List<Float> =
        (0 until length()).map { getDouble(it).toFloat() }

    private fun comparar(esperado: List<Float>, obtido: FloatArray, contexto: String) {
        assertEquals("$contexto: tamanho diferente", esperado.size, obtido.size)
        esperado.indices.forEach { i ->
            assertEquals("$contexto: posição $i", esperado[i], obtido[i], tolerancia)
        }
    }

    @Test
    fun `normalizeLandmarks bate com o contrato do treino`() {
        val casos = fixtures.getJSONArray("normalize_hand_landmarks")
        assertTrue("fixtures de mão vazias", casos.length() > 0)

        for (i in 0 until casos.length()) {
            val caso = casos.getJSONObject(i)
            val entrada = caso.getJSONArray("entrada")
            val pontos = (0 until entrada.length()).map { p ->
                val par = entrada.getJSONArray(p)
                par.getDouble(0).toFloat() to par.getDouble(1).toFloat()
            }
            comparar(
                caso.getJSONArray("esperado").toFloatList(),
                LibrasMath.normalizeLandmarks(pontos),
                caso.getString("nome"),
            )
        }
    }

    @Test
    fun `normalizeBodyFrame bate com o contrato do treino`() {
        val casos = fixtures.getJSONArray("normalize_body_frame")
        assertTrue("fixtures de corpo vazias", casos.length() > 0)

        for (i in 0 until casos.length()) {
            val caso = casos.getJSONObject(i)
            val entrada = caso.getJSONArray("entrada").toFloatList().toFloatArray()
            comparar(
                caso.getJSONArray("esperado").toFloatList(),
                LibrasMath.normalizeBodyFrame(entrada),
                caso.getString("nome"),
            )
        }
    }

    @Test
    fun `resample escolhe os mesmos quadros que o treino`() {
        val casos = fixtures.getJSONArray("resample")
        assertTrue("fixtures de resample vazias", casos.length() > 0)

        for (i in 0 until casos.length()) {
            val caso = casos.getJSONObject(i)
            val tamanho = caso.getInt("tamanho_entrada")
            val quantidade = caso.getInt("quantidade")
            val esperado = caso.getJSONArray("indices_esperados")
                .let { arr -> (0 until arr.length()).map { arr.getInt(it) } }

            // Quadros identificáveis: cada um carrega o próprio índice, então
            // a saída revela quais quadros foram escolhidos.
            val frames = (0 until tamanho).toList()
            assertEquals(
                "resample de $tamanho quadros",
                esperado,
                LibrasMath.resample(frames, quantidade),
            )
        }
    }
}
