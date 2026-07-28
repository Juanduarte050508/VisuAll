package com.visuall.app.libras

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LibrasMathTest {

    private fun assertFloatArrayEquals(expected: FloatArray, actual: FloatArray, delta: Float = 1e-4f) {
        assertEquals("tamanhos diferentes", expected.size, actual.size)
        expected.indices.forEach { i ->
            assertEquals("index $i", expected[i], actual[i], delta)
        }
    }

    @Test
    fun `normalizeLandmarks translada pro pulso e escala pelo maior valor absoluto`() {
        val pontos = listOf(10f to 20f, 15f to 20f, 10f to 30f)
        val result = LibrasMath.normalizeLandmarks(pontos)
        // Translação: (0,0), (5,0), (0,10) -- maior abs = 10 -- escala tudo por 10.
        assertFloatArrayEquals(floatArrayOf(0f, 0f, 0.5f, 0f, 0f, 1f), result)
    }

    @Test
    fun `normalizeLandmarks nao divide por zero quando todos os pontos sao iguais`() {
        val pontos = listOf(5f to 5f)
        val result = LibrasMath.normalizeLandmarks(pontos)
        assertFloatArrayEquals(floatArrayOf(0f, 0f), result)
    }

    @Test
    fun `mirrorLandmarks nega só as coordenadas x`() {
        val dados = floatArrayOf(1f, 2f, -3f, 4f)
        val mirrored = LibrasMath.mirrorLandmarks(dados)
        assertFloatArrayEquals(floatArrayOf(-1f, 2f, 3f, 4f), mirrored)
        // não deve mutar o array original
        assertFloatArrayEquals(floatArrayOf(1f, 2f, -3f, 4f), dados)
    }

    private fun maoComDedos(esticados: Boolean): List<Pair<Float, Float>> {
        val lms = MutableList(21) { 0f to 0f }
        lms[0] = 0.5f to 0.9f  // pulso
        lms[4] = 0.7f to 0.85f // ponta do polegar, afastada do pulso em x
        if (esticados) {
            lms[5] = 0.5f to 0.6f;  lms[8]  = 0.5f to 0.4f  // indicador
            lms[9] = 0.5f to 0.6f;  lms[12] = 0.5f to 0.4f  // médio
            lms[13] = 0.5f to 0.6f; lms[16] = 0.5f to 0.4f  // anelar
            lms[17] = 0.5f to 0.6f; lms[20] = 0.5f to 0.4f  // mindinho
        } else {
            lms[5] = 0.5f to 0.6f;  lms[8]  = 0.5f to 0.65f
            lms[9] = 0.5f to 0.6f;  lms[12] = 0.5f to 0.65f
            lms[13] = 0.5f to 0.6f; lms[16] = 0.5f to 0.65f
            lms[17] = 0.5f to 0.6f; lms[20] = 0.5f to 0.65f
        }
        return lms
    }

    @Test
    fun `detectarDedosEsticados reconhece mao aberta`() {
        assertTrue(LibrasMath.detectarDedosEsticados(maoComDedos(esticados = true)))
    }

    @Test
    fun `detectarDedosEsticados rejeita mao fechada`() {
        assertFalse(LibrasMath.detectarDedosEsticados(maoComDedos(esticados = false)))
    }

    @Test
    fun `std de valores constantes e zero`() {
        assertEquals(0f, LibrasMath.std(listOf(2f, 2f, 2f)), 1e-6f)
    }

    @Test
    fun `std calcula o desvio padrao populacional`() {
        // média 2, variância populacional (1+0+1)/3 = 0.6667, raiz ~0.8165
        assertEquals(0.8165f, LibrasMath.std(listOf(1f, 2f, 3f)), 1e-3f)
    }
}
