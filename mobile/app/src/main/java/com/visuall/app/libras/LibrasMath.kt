package com.visuall.app.libras

import kotlin.math.abs
import kotlin.math.sqrt

// Funções puras de geometria/estatística usadas pelos módulos de
// reconhecimento (LetraEngine, BodyGestureEngine). Sem dependência de
// Android/Context, então dá pra testar direto em JVM (ver LibrasMathTest).
internal object LibrasMath {

    // Traduz o ponto 0 (pulso) pra origem e escala pelo maior valor absoluto
    // — invariante à posição da mão na tela e à distância da câmera.
    fun normalizeLandmarks(pontos: List<Pair<Float, Float>>): FloatArray {
        val baseX = pontos[0].first
        val baseY = pontos[0].second
        val norm  = FloatArray(pontos.size * 2)
        for (i in pontos.indices) {
            norm[i * 2 + 0] = pontos[i].first  - baseX
            norm[i * 2 + 1] = pontos[i].second - baseY
        }
        val maxV = norm.map { abs(it) }.maxOrNull()?.takeIf { it > 0f } ?: 1f
        return FloatArray(norm.size) { norm[it] / maxV }
    }

    // Espelha os landmarks normalizados no eixo X — usado pra comparar uma
    // amostra calibrada com a mão oposta (mesma letra, lado espelhado).
    fun mirrorLandmarks(dados: FloatArray): FloatArray {
        val mirrored = dados.copyOf()
        for (i in mirrored.indices step 2) {
            mirrored[i] = -mirrored[i]
        }
        return mirrored
    }

    // Detecta os 4 dedos (indicador a mindinho) esticados + polegar afastado
    // — usado tanto pelo gesto de "limpar" no alfabeto quanto pelo de "mão
    // aberta" no modo corpo.
    fun detectarDedosEsticados(lms: List<Pair<Float, Float>>): Boolean {
        val margem = 0.06f
        return lms[8].second  < lms[5].second  - margem &&
               lms[12].second < lms[9].second  - margem &&
               lms[16].second < lms[13].second - margem &&
               lms[20].second < lms[17].second - margem &&
               abs(lms[4].first - lms[0].first) > 0.12f
    }

    fun std(values: List<Float>): Float {
        val mean = values.average().toFloat()
        return sqrt(values.map { (it - mean) * (it - mean) }.average().toFloat())
    }
}
