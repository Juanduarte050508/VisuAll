package com.visuall.app.libras

import kotlin.math.abs
import kotlin.math.sqrt

// Funções puras de geometria/estatística usadas pelos módulos de
// reconhecimento (LetraEngine, BodyGestureEngine). Sem dependência de
// Android/Context, então dá pra testar direto em JVM (ver LibrasMathTest).
//
// ATENÇÃO: tudo aqui tem um gêmeo em Python, em treinamento/treinar_visuall.py
// (normalize_hand_landmarks, normalize_body_frame, resample_sequence). Os dois
// lados PRECISAM produzir exatamente os mesmos números: o Python prepara os
// dados com que o modelo aprende, e este arquivo prepara os dados que o modelo
// recebe no celular. Se um mudar sem o outro, nada quebra e nenhum erro
// aparece — o app só passa a errar mais, porque foi ensinado num formato e
// está sendo usado em outro. LandmarkContractTest (Kotlin) e
// test_landmark_contract.py (Python) travam os dois lados nos mesmos valores;
// se você mexer aqui, os dois testes falham até o Python acompanhar.
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

    // Régua da mão: distância do pulso (0) à base do dedo médio (9). É o
    // segmento mais estável da mão — não encolhe quando os dedos dobram, só
    // quando a mão se afasta da câmera — então serve pra converter uma medida
    // em unidades de imagem numa fração do tamanho da mão.
    //
    // Devolve 0 quando a mão veio degenerada (os dois pontos no mesmo lugar):
    // quem chama trata isso como "não dá pra medir", em vez de dividir por
    // quase zero e transformar qualquer ruído num gesto.
    fun escalaDaMao(lms: List<Pair<Float, Float>>): Float {
        val dx = lms[9].first  - lms[0].first
        val dy = lms[9].second - lms[0].second
        val escala = sqrt(dx * dx + dy * dy)
        return if (escala > ESCALA_MINIMA_MAO) escala else 0f
    }

    // Abaixo disso a mão veio degenerada (pulso e base do médio no mesmo
    // ponto) e não há régua confiável — mesmo papel do ESCALA_MINIMA_OMBROS
    // no corpo.
    const val ESCALA_MINIMA_MAO = 0.0001f

    // Quanto cada dedo precisa subir acima da própria base, e quanto o polegar
    // precisa se afastar do pulso — os dois EM FRAÇÃO DO TAMANHO DA MÃO.
    //
    // Eram 0.06 e 0.12 em unidades de imagem, o que só valia na distância em
    // que foram calibrados: de perto a mão ocupa muito do quadro e passava; de
    // longe a MESMA mão aberta encolhe, as diferenças caem abaixo dos limiares
    // fixos e o gesto neutro deixava de ser reconhecido como mão aberta — daí
    // ele caía no classificador de letras e virava um falso positivo ("F").
    // Os valores relativos foram escolhidos pra reproduzir os antigos na
    // distância de calibração (mão com ~0.30 de altura no quadro).
    const val FRACAO_DEDO_ESTICADO = 0.20f
    const val FRACAO_POLEGAR_AFASTADO = 0.40f

    // Detecta os 4 dedos (indicador a mindinho) esticados + polegar afastado
    // — usado tanto pelo gesto de "limpar" no alfabeto quanto pelo de "mão
    // aberta" no modo corpo. Invariante à distância da câmera: todos os
    // limiares são medidos em fração da própria mão (ver escalaDaMao).
    fun detectarDedosEsticados(lms: List<Pair<Float, Float>>): Boolean {
        val escala = escalaDaMao(lms)
        if (escala <= 0f) return false
        val margem = FRACAO_DEDO_ESTICADO * escala
        return lms[8].second  < lms[5].second  - margem &&
               lms[12].second < lms[9].second  - margem &&
               lms[16].second < lms[13].second - margem &&
               lms[20].second < lms[17].second - margem &&
               abs(lms[4].first - lms[0].first) > FRACAO_POLEGAR_AFASTADO * escala
    }

    fun std(values: List<Float>): Float {
        val mean = values.average().toFloat()
        return sqrt(values.map { (it - mean) * (it - mean) }.average().toFloat())
    }

    // Escala mínima aceita entre os ombros. Abaixo disso a pose veio
    // degenerada (pessoa de lado, ombro fora do quadro, detecção ruim) e
    // dividir por um número quase zero explodiria as features — um ponto a
    // 1cm do centro viraria 1000. Nesse caso não normaliza (divide por 1).
    const val ESCALA_MINIMA_OMBROS = 0.0001f

    // Centraliza x,y no meio dos ombros e escala pela distância entre eles.
    // z fica cru, como sai do MediaPipe. Gêmeo do normalize_body_frame do
    // Python — ver aviso no topo do arquivo.
    fun normalizeBodyFrame(frame: FloatArray): FloatArray {
        val normalized = frame.copyOf()
        val leftShoulder = LibrasAnalyzer.BODY_POINT_LEFT_SHOULDER * 3
        val rightShoulder = LibrasAnalyzer.BODY_POINT_RIGHT_SHOULDER * 3
        val centerX = (frame[leftShoulder] + frame[rightShoulder]) / 2f
        val centerY = (frame[leftShoulder + 1] + frame[rightShoulder + 1]) / 2f
        // A escala é a distância entre os ombros em 3D — o dz ENTRA na conta.
        // Usar só dx/dy deixava todas as features numa escala diferente da do
        // treino.
        val dx = frame[leftShoulder] - frame[rightShoulder]
        val dy = frame[leftShoulder + 1] - frame[rightShoulder + 1]
        val dz = frame[leftShoulder + 2] - frame[rightShoulder + 2]
        val scale = sqrt(dx * dx + dy * dy + dz * dz)
            .takeIf { it > ESCALA_MINIMA_OMBROS } ?: 1f
        for (point in 0 until LibrasAnalyzer.BODY_TOTAL_POINTS) {
            val base = point * 3
            normalized[base] = (normalized[base] - centerX) / scale
            normalized[base + 1] = (normalized[base + 1] - centerY) / scale
        }
        return normalized
    }

    // Reduz/estica uma sequência de quadros pro tamanho fixo que o modelo
    // espera, escolhendo quadros existentes por índice (não interpola
    // valores). Gêmeo do resample_sequence do Python.
    fun <T> resample(frames: List<T>, count: Int): List<T> {
        if (frames.size == count) return frames
        return List(count) { index ->
            frames[resampleIndex(frames.size, count, index)]
        }
    }

    fun resampleIndex(size: Int, count: Int, index: Int): Int =
        ((size - 1) * index.toFloat() / (count - 1)).toInt()
}
