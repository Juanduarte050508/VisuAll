package com.visuall.app.libras

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DynamicLetterMotionTest {

    @Test
    fun `j com trajeto pequeno e rejeitado`() {
        val janela = List(LibrasAnalyzer.JANELA_MLP) { i ->
            frame(pinkyX = i * 0.005f, pinkyY = i * 0.005f)
        }

        assertFalse(DynamicLetterMotion.jTemTrajetoSuficiente(janela))
        assertEquals(
            "-",
            DynamicLetterMotion.filtrar(Prediction("J", 0.99f, "dinamico"), janela).letra
        )
    }

    @Test
    fun `j com movimento em um eixo so e rejeitado`() {
        val janela = List(LibrasAnalyzer.JANELA_MLP) { i ->
            frame(pinkyX = 0f, pinkyY = i * 0.05f)
        }

        assertFalse(DynamicLetterMotion.jTemTrajetoSuficiente(janela))
    }

    @Test
    fun `j com deslocamento em dois eixos e aceito`() {
        val janela = List(LibrasAnalyzer.JANELA_MLP) { i ->
            val t = i / (LibrasAnalyzer.JANELA_MLP - 1f)
            val hook = if (t < 0.65f) 0f else (t - 0.65f) / 0.35f
            frame(pinkyX = hook * 0.25f, pinkyY = t * 0.42f)
        }

        assertTrue(DynamicLetterMotion.jTemTrajetoSuficiente(janela))
        assertEquals(
            "J",
            DynamicLetterMotion.filtrar(Prediction("J", 0.99f, "dinamico"), janela).letra
        )
    }

    @Test
    fun `filtro nao altera outras letras dinamicas`() {
        val janela = List(LibrasAnalyzer.JANELA_MLP) { frame(0f, 0f) }

        assertEquals(
            "Z",
            DynamicLetterMotion.filtrar(Prediction("Z", 0.99f, "dinamico"), janela).letra
        )
    }

    private fun frame(pinkyX: Float, pinkyY: Float): FloatArray {
        val dados = FloatArray(LibrasAnalyzer.FEATURES_ESTATICO)
        dados[0] = 0f
        dados[1] = 0f
        dados[18] = 0f
        dados[19] = 1f
        dados[40] = pinkyX
        dados[41] = pinkyY
        return dados
    }
}
