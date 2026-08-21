package com.visuall.app.libras

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class TrainingProgressCalculatorTest {

    private val letras = listOf("A", "B", "C", "D")
    private val alvo = 100

    private fun calcular(contagens: Map<String, Int>) =
        TrainingProgressCalculator.calcular(letras, alvo) { contagens[it] ?: 0 }

    @Test
    fun `sem nenhuma amostra fica em zero por cento`() {
        val progress = calcular(emptyMap())
        assertEquals(0, progress.percent)
        assertEquals(0, progress.trainedLetters)
        assertEquals(0, progress.totalSamples)
        assertEquals(letras, progress.missingLetters)
    }

    @Test
    fun `todas no alvo fecham cem por cento`() {
        val progress = calcular(letras.associateWith { alvo })
        assertEquals(100, progress.percent)
        assertEquals(4, progress.trainedLetters)
        assertEquals(emptyList<String>(), progress.missingLetters)
    }

    @Test
    fun `excesso numa letra nao compensa outra vazia`() {
        // 400 amostras de "A" e nada no resto: sem o teto por letra isso
        // marcaria 100% com 3 das 4 letras sem nenhum dado.
        val progress = calcular(mapOf("A" to 400))
        assertEquals(25, progress.percent)
        assertEquals(1, progress.trainedLetters)
        assertEquals(400, progress.totalSamples)
    }

    @Test
    fun `letras fracas vem da mais fraca pra menos fraca`() {
        val progress = calcular(mapOf("A" to 90, "B" to 10, "C" to 50))
        assertEquals(listOf("D", "B", "C", "A"), progress.missingLetters)
    }

    @Test
    fun `empate em amostras desempata alfabeticamente`() {
        // Ordem estável entre atualizações: sem o desempate a lista dança.
        val progress = calcular(mapOf("D" to 5, "B" to 5, "C" to 5, "A" to 5))
        assertEquals(listOf("A", "B", "C", "D"), progress.missingLetters)
    }

    @Test
    fun `lista de letras vazia nao divide por zero`() {
        val progress = TrainingProgressCalculator.calcular(emptyList(), alvo) { 0 }
        assertEquals(0, progress.percent)
    }

    @Test
    fun `proxima letra fraca varre circularmente a partir da atual`() {
        val contagens = mapOf("A" to alvo, "B" to alvo, "C" to 0, "D" to alvo)
        // Começando em "D" (índice 3), a próxima fraca é "C" (índice 2),
        // dando a volta pelo fim da lista.
        val index = TrainingProgressCalculator.indiceProximaLetraFraca(
            letras, indiceAtual = 3, includeCurrent = false, alvoForte = alvo
        ) { contagens[it] ?: 0 }
        assertEquals(2, index)
    }

    @Test
    fun `includeCurrent decide se a letra atual conta`() {
        val contagens = mapOf("A" to 0, "B" to alvo, "C" to alvo, "D" to alvo)
        val comAtual = TrainingProgressCalculator.indiceProximaLetraFraca(
            letras, indiceAtual = 0, includeCurrent = true, alvoForte = alvo
        ) { contagens[it] ?: 0 }
        assertEquals(0, comAtual)

        // Sem incluir a atual e com todo o resto forte, dá a volta e para nela.
        val semAtual = TrainingProgressCalculator.indiceProximaLetraFraca(
            letras, indiceAtual = 0, includeCurrent = false, alvoForte = alvo
        ) { contagens[it] ?: 0 }
        assertEquals(0, semAtual)
    }

    @Test
    fun `todas fortes devolve nulo`() {
        val index = TrainingProgressCalculator.indiceProximaLetraFraca(
            letras, indiceAtual = 0, includeCurrent = true, alvoForte = alvo
        ) { alvo }
        assertNull(index)
    }

    @Test
    fun `nivel usa as faixas de amostras`() {
        assertEquals("SEM DADOS", TrainingProgressCalculator.nivel(0, 100, 24))
        assertEquals("INICIO", TrainingProgressCalculator.nivel(1, 100, 24))
        assertEquals("BASICO", TrainingProgressCalculator.nivel(24, 100, 24))
        assertEquals("FORTE", TrainingProgressCalculator.nivel(100, 100, 24))
    }
}
