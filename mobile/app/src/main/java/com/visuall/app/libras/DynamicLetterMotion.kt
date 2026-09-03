package com.visuall.app.libras

import kotlin.math.hypot
import kotlin.math.sqrt

// Filtros geometricos leves para letras dinamicas. O modelo ONNX continua
// recebendo a janela normalizada de sempre; isto olha a janela crua apenas
// para rejeitar casos em que o movimento fisico nao combina com a letra.
internal object DynamicLetterMotion {

    fun filtrar(prediction: Prediction, janelaCrua: List<FloatArray>): Prediction {
        if (!prediction.letra.equals("J", ignoreCase = true)) return prediction
        return if (jTemTrajetoSuficiente(janelaCrua)) {
            prediction
        } else {
            prediction.copy(letra = "-")
        }
    }

    fun jTemTrajetoSuficiente(janelaCrua: List<FloatArray>): Boolean {
        if (janelaCrua.size < LibrasAnalyzer.JANELA_MLP) return false

        val escala = janelaCrua
            .mapNotNull { escalaDaMao(it).takeIf { valor -> valor > 0f } }
            .average()
            .takeIf { !it.isNaN() && it > 0.0 }
            ?.toFloat()
            ?: return false

        val pontos = janelaCrua.mapNotNull { ponto(it, 20) }
        if (pontos.size < LibrasAnalyzer.JANELA_MLP) return false

        val spanX = (pontos.maxOf { it.x } - pontos.minOf { it.x }) / escala
        val spanY = (pontos.maxOf { it.y } - pontos.minOf { it.y }) / escala
        val trajeto = caminhoTotal(pontos) / escala

        return spanX >= LibrasAnalyzer.J_TRAJETO_X_MIN &&
            spanY >= LibrasAnalyzer.J_TRAJETO_Y_MIN &&
            trajeto >= LibrasAnalyzer.J_TRAJETO_TOTAL_MIN
    }

    private data class Point(val x: Float, val y: Float)

    private fun ponto(frame: FloatArray, index: Int): Point? {
        val base = index * 2
        if (frame.size <= base + 1) return null
        return Point(frame[base], frame[base + 1])
    }

    private fun escalaDaMao(frame: FloatArray): Float {
        val pulso = ponto(frame, 0) ?: return 0f
        val baseMedio = ponto(frame, 9) ?: return 0f
        val dx = baseMedio.x - pulso.x
        val dy = baseMedio.y - pulso.y
        return sqrt(dx * dx + dy * dy)
            .takeIf { it > LibrasMath.ESCALA_MINIMA_MAO } ?: 0f
    }

    private fun caminhoTotal(pontos: List<Point>): Float {
        var total = 0f
        for (i in 1 until pontos.size) {
            total += hypot(pontos[i].x - pontos[i - 1].x, pontos[i].y - pontos[i - 1].y)
        }
        return total
    }
}
