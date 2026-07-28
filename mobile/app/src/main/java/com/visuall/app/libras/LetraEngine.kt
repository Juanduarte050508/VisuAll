package com.visuall.app.libras

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer

// Reconhecimento de letra do alfabeto (MLP estático + dinâmico) e calibração
// pessoal — extraído do LibrasAnalyzer pra isolar esse conjunto de
// modelos/estado (sessões ONNX, janela deslizante, calibração) do resto do
// pipeline (corpo, marcador de rosto), que não têm nada a ver com isso.
internal class LetraEngine(
    context: Context,
    private val onFeedback: (mensagem: String, nivel: Int) -> Unit
) {
    private val ortEnv          = OrtEnvironment.getEnvironment()
    private val sessionEstatico = ortEnv.createSession(
        context.assets.open("letras_estaticas/geral/model.onnx").readBytes()
    )
    private val sessionDinamico = ortEnv.createSession(
        context.assets.open("letras_dinamicas/geral/model.onnx").readBytes()
    )
    private val sessionDinamicoParcial: OrtSession? =
        runCatching {
            ortEnv.createSession(context.assets.open("letras_dinamicas/parcial/model.onnx").readBytes())
        }.getOrNull()
    private val labelsEstatico  = context.assets.open("letras_estaticas/geral/labels.txt")
        .bufferedReader().readLines().filter { it.isNotBlank() }
    private val labelsDinamico  = context.assets.open("letras_dinamicas/geral/labels.txt")
        .bufferedReader().readLines().filter { it.isNotBlank() }
    private val labelsDinamicoParcial = runCatching {
        context.assets.open("letras_dinamicas/parcial/labels.txt")
            .bufferedReader().readLines().filter { it.isNotBlank() }
    }.getOrDefault(emptyList())
    private val modelosEstaticosIndividuais = carregarModelosIndividuais(
        context, labelsEstatico, "letras_estaticas"
    )
    private val modelosDinamicosIndividuais = carregarModelosIndividuais(
        context, labelsDinamico, "letras_dinamicas"
    )
    private val trainingStore = CalibrationTrainingStore(context, labelsEstatico, labelsDinamico)

    private val bufferLm          = ArrayDeque<FloatArray>()
    private val calibrationLock   = Any()
    private val calibrationBuffer = ArrayList<FloatArray>()
    @Volatile private var calibrationTarget: String? = null

    // Timestamp de quando o movimento acima de LIMIAR_MOVIMENTO começou a ser
    // sustentado (0L = não está sustentando agora). Só confiamos no modelo
    // dinâmico depois de sustentado por MOVIMENTO_SUSTENTADO_MS — ver
    // comentário do LIMIAR_MOVIMENTO em LibrasAnalyzer.
    //
    // Era contado em FRAMES (não em tempo) até esta correção: num aparelho
    // que analisa menos frames por segundo, "3 frames" passava a levar bem
    // mais tempo de parede, então a janela de gesto (que dura ~300-500ms de
    // verdade) terminava antes da histerese liberar o classificador
    // dinâmico — o app perdia H/J/K/X/Z em celulares mais lentos. Tempo em
    // vez de contagem de frames deixa o gate consistente independente da
    // taxa de quadros real.
    private var movimentoSustentadoDesde = 0L

    private data class ModeloIndividual(
        val label: String,
        val session: OrtSession
    )

    private fun carregarModelosIndividuais(
        context: Context,
        labels: List<String>,
        basePath: String
    ): List<ModeloIndividual> {
        return labels.mapNotNull { label ->
            val safe = label.uppercase().filter { it.isLetterOrDigit() || it == '_' || it == '-' }
            runCatching {
                ModeloIndividual(
                    label = label,
                    session = ortEnv.createSession(
                        context.assets.open("$basePath/$safe/model.onnx").readBytes()
                    )
                )
            }.getOrNull()
        }
    }

    fun resetMovimentoSustentado() {
        movimentoSustentadoDesde = 0L
    }

    fun limparBuffer() {
        bufferLm.clear()
    }

    // Pipeline completo de um frame com mão detectada: normaliza, alimenta a
    // calibração em andamento (se houver), atualiza a janela deslizante,
    // classifica (estático, dinâmico ou calibração pessoal) e emite o
    // feedback textual correspondente.
    fun process(pontos: List<Pair<Float, Float>>): Prediction {
        val dados = LibrasMath.normalizeLandmarks(pontos)
        captureCalibrationFrame(dados)
        bufferLm.addLast(dados)
        while (bufferLm.size > LibrasAnalyzer.JANELA_MLP + 5) bufferLm.removeFirst()

        val movimento = calcularMovimento()
        val predicao = escolherClassificacao(dados, movimento)
        emitirFeedback(predicao, movimento)
        return predicao
    }

    private fun captureCalibrationFrame(dados: FloatArray) {
        if (calibrationTarget == null) return
        synchronized(calibrationLock) {
            if (calibrationTarget == null) return
            if (calibrationBuffer.size >= LibrasAnalyzer.CALIBRATION_MAX_FRAMES) {
                calibrationBuffer.removeAt(0)
            }
            calibrationBuffer.add(dados.copyOf())
            val letra = calibrationTarget.orEmpty()
            val total = calibrationBuffer.size
            onFeedback(
                "GRAVANDO $letra  $total/${LibrasAnalyzer.CALIBRATION_TARGET_FRAMES}",
                LibrasAnalyzer.FEEDBACK_BOM
            )
        }
    }

    private fun emitirFeedback(prediction: Prediction, movimento: Float) {
        if (calibrationTarget != null) return

        val mensagem = when {
            prediction.letra != "-" && prediction.confianca >= 0.92f -> "SINAL ESTAVEL"
            prediction.letra != "-" && prediction.confianca >= 0.82f -> "SEGURE MAIS FIRME"
            movimento > LibrasAnalyzer.LIMIAR_MOVIMENTO -> "MOVIMENTO ALTO"
            prediction.confianca >= 0.68f -> "QUASE: AJUSTE ANGULO"
            else -> "APROXIME A MAO"
        }
        val nivel = when {
            prediction.letra != "-" && prediction.confianca >= 0.90f -> LibrasAnalyzer.FEEDBACK_BOM
            prediction.confianca >= 0.72f -> LibrasAnalyzer.FEEDBACK_NEUTRO
            else -> LibrasAnalyzer.FEEDBACK_ALERTA
        }
        onFeedback(mensagem, nivel)
    }

    private fun aplicarCalibracaoPessoal(dados: FloatArray, base: Prediction): Prediction {
        // Só usamos a calibração pessoal quando o modelo não reconheceu nada,
        // para nunca sobrescrever uma decisão confiante do modelo por um
        // vizinho-mais-próximo que pode estar errado.
        if (base.letra != "-") return base
        val calibrado = melhorCalibracao(dados)
        return if (calibrado.letra != "-") calibrado else base
    }

    private fun melhorCalibracao(dados: FloatArray): Prediction {
        val match = synchronized(calibrationLock) {
            trainingStore.bestStaticMatch(
                candidates = listOf(dados, LibrasMath.mirrorLandmarks(dados)),
                maxDistance = LibrasAnalyzer.CALIBRATION_MATCH_LIMIT
            )
        } ?: return Prediction("-", 0f, "calibrado")

        val score = (1f - match.distance / LibrasAnalyzer.CALIBRATION_MATCH_LIMIT).coerceIn(0f, 1f)
        val confianca = (0.86f + score * 0.13f).coerceAtMost(0.99f)
        return Prediction(match.letter, confianca, "calibrado")
    }

    private fun averageFrames(frames: List<FloatArray>): FloatArray {
        val result = FloatArray(LibrasAnalyzer.FEATURES_ESTATICO)
        if (frames.isEmpty()) return result
        frames.forEach { frame ->
            for (i in result.indices) {
                result[i] += frame[i]
            }
        }
        for (i in result.indices) {
            result[i] = result[i] / frames.size.toFloat()
        }
        return result
    }

    private fun calcularMovimento(): Float {
        if (bufferLm.size < 5) return 0f
        val recent = bufferLm.toList().takeLast(5)
        return LibrasMath.std(recent.map { it[2] }) + LibrasMath.std(recent.map { it[3] }) +
               LibrasMath.std(recent.map { it[16] }) + LibrasMath.std(recent.map { it[17] })
    }

    private fun escolherClassificacao(dados: FloatArray, movimento: Float): Prediction {
        // Histerese: LIMIAR_MOVIMENTO sozinho não distingue "tremida de 1
        // frame" de "traço real de H/J/K/X/Z" — os dois cruzam o mesmo valor
        // de magnitude. A diferença é a DURAÇÃO: um gesto de verdade sustenta
        // o movimento por um tempo mínimo; ruído da câmera não.
        val agora = System.currentTimeMillis()
        if (movimento > LibrasAnalyzer.LIMIAR_MOVIMENTO) {
            if (movimentoSustentadoDesde == 0L) movimentoSustentadoDesde = agora
        } else {
            movimentoSustentadoDesde = 0L
        }
        val movimentoConfiavel = movimentoSustentadoDesde != 0L &&
            (agora - movimentoSustentadoDesde) >= LibrasAnalyzer.MOVIMENTO_SUSTENTADO_MS

        if (movimentoConfiavel && bufferLm.size >= LibrasAnalyzer.JANELA_MLP) {
            return classificarDinamico()
        }
        // Calibração pessoal entra apenas como reforço quando o modelo não
        // reconheceu nada (ver aplicarCalibracaoPessoal).
        return aplicarCalibracaoPessoal(dados, classificarEstatico(dados))
    }

    private fun classificarEstatico(dados: FloatArray): Prediction {
        if (modelosEstaticosIndividuais.isNotEmpty()) {
            val individual = classificarIndividual(
                entrada = dados,
                modelos = modelosEstaticosIndividuais,
                features = LibrasAnalyzer.FEATURES_ESTATICO,
                confiancaMinima = LibrasAnalyzer.CONFIANCA_MINIMA,
                margemMinima = LibrasAnalyzer.MARGEM_ESTATICA_MINIMA,
                modo = "estatico_individual"
            )
            if (individual.letra != "-") return individual
        }

        val tensor = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(dados), longArrayOf(1, LibrasAnalyzer.FEATURES_ESTATICO.toLong()))
        val out    = sessionEstatico.run(mapOf("landmarks_input" to tensor))
        @Suppress("UNCHECKED_CAST")
        val probs  = (out[1].value as Array<FloatArray>)[0]
        val idx    = probs.indices.maxByOrNull { probs[it] } ?: 0
        val conf   = probs[idx]
        // Margem contra a segunda melhor opção: se o modelo está dividido
        // (ex.: C vs mão aberta), rejeitamos em vez de chutar.
        val second = probs.indices.filter { it != idx }.maxOfOrNull { probs[it] } ?: 0f
        val margem = conf - second
        val label  = labelsEstatico.getOrNull(idx)
        val letra  = if (label != null && conf >= LibrasAnalyzer.CONFIANCA_MINIMA &&
                         margem >= LibrasAnalyzer.MARGEM_ESTATICA_MINIMA) label else "-"
        tensor.close(); out.close()
        return Prediction(letra, conf, "estatico", margem)
    }

    private fun classificarDinamico(): Prediction {
        val entrada = FloatArray(LibrasAnalyzer.FEATURES_DINAMICO)
        bufferLm.toList().takeLast(LibrasAnalyzer.JANELA_MLP).forEachIndexed { i, frame ->
            frame.copyInto(entrada, i * LibrasAnalyzer.FEATURES_ESTATICO)
        }

        if (modelosDinamicosIndividuais.isNotEmpty()) {
            val individual = classificarIndividual(
                entrada = entrada,
                modelos = modelosDinamicosIndividuais,
                features = LibrasAnalyzer.FEATURES_DINAMICO,
                confiancaMinima = LibrasAnalyzer.CONFIANCA_DINAMICA,
                margemMinima = LibrasAnalyzer.MARGEM_DINAMICA_MINIMA,
                modo = "dinamico_individual"
            )
            if (individual.letra != "-") return individual
        }

        // Se existe um modelo parcial treinado pela ferramenta local, ele é
        // usado primeiro para as classes que já têm dados novos (ex.: H/J/K/Z).
        // O modelo completo continua como fallback, preservando letras ainda
        // sem re-treino parcial, como X.
        sessionDinamicoParcial?.let { partialSession ->
            val parcial = classificarDinamicoComModelo(
                entrada = entrada,
                session = partialSession,
                labels = labelsDinamicoParcial,
                modo = "dinamico_parcial"
            )
            if (parcial.letra != "-") return parcial
        }

        return classificarDinamicoComModelo(
            entrada = entrada,
            session = sessionDinamico,
            labels = labelsDinamico,
            modo = "dinamico"
        )
    }

    private fun classificarDinamicoComModelo(
        entrada: FloatArray,
        session: OrtSession,
        labels: List<String>,
        modo: String
    ): Prediction {
        val tensor = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(entrada), longArrayOf(1, LibrasAnalyzer.FEATURES_DINAMICO.toLong()))
        val out    = session.run(mapOf("landmarks_input" to tensor))
        @Suppress("UNCHECKED_CAST")
        val probs  = (out[1].value as Array<FloatArray>)[0]
        val idx    = probs.indices.maxByOrNull { probs[it] } ?: 0
        val conf   = probs[idx]
        val second = probs.indices.filter { it != idx }.maxOfOrNull { probs[it] } ?: 0f
        val margem = conf - second
        val letra  = if (conf >= LibrasAnalyzer.CONFIANCA_DINAMICA &&
                         margem >= LibrasAnalyzer.MARGEM_DINAMICA_MINIMA &&
                         idx < labels.size) labels[idx] else "-"
        tensor.close(); out.close()
        return Prediction(letra, conf, modo, margem)
    }

    private fun classificarIndividual(
        entrada: FloatArray,
        modelos: List<ModeloIndividual>,
        features: Int,
        confiancaMinima: Float,
        margemMinima: Float,
        modo: String
    ): Prediction {
        var melhorLabel = "-"
        var melhorConf = 0f
        var segundoConf = 0f

        modelos.forEach { modelo ->
            val tensor = OnnxTensor.createTensor(
                ortEnv, FloatBuffer.wrap(entrada), longArrayOf(1, features.toLong()))
            val out = modelo.session.run(mapOf("landmarks_input" to tensor))
            @Suppress("UNCHECKED_CAST")
            val probs = (out[1].value as Array<FloatArray>)[0]
            val positivo = when {
                probs.size >= 2 -> probs[1]
                probs.size == 1 -> probs[0]
                else -> 0f
            }
            if (positivo > melhorConf) {
                segundoConf = melhorConf
                melhorConf = positivo
                melhorLabel = modelo.label
            } else if (positivo > segundoConf) {
                segundoConf = positivo
            }
            tensor.close()
            out.close()
        }

        val margem = melhorConf - segundoConf
        val letra = if (melhorConf >= confiancaMinima && margem >= margemMinima) melhorLabel else "-"
        return Prediction(letra, melhorConf, modo, margem)
    }

    // ── Calibração pessoal (API pública usada pela UI) ─────────────────────
    fun startCalibration(letra: String) {
        synchronized(calibrationLock) {
            calibrationTarget = letra.uppercase()
            calibrationBuffer.clear()
        }
        onFeedback("CALIBRANDO ${letra.uppercase()}: SEGURE A MAO", LibrasAnalyzer.FEEDBACK_NEUTRO)
    }

    fun finishCalibration(): Boolean {
        synchronized(calibrationLock) {
            val letra = calibrationTarget ?: return false
            if (calibrationBuffer.size < LibrasAnalyzer.CALIBRATION_MIN_FRAMES) {
                onFeedback("POUCAS AMOSTRAS PARA $letra", LibrasAnalyzer.FEEDBACK_ALERTA)
                return false
            }

            val frames = calibrationBuffer.map { it.copyOf() }
            if (letra in labelsDinamico) {
                val sequencias = trainingStore.saveDynamicSamples(letra, frames)
                if (sequencias == 0) {
                    onFeedback("MOVIMENTE MAIS PARA $letra", LibrasAnalyzer.FEEDBACK_ALERTA)
                    return false
                }
                trainingStore.incrementSampleCount(letra, sequencias)
            } else {
                val referencia = averageFrames(frames)
                trainingStore.saveStaticCalibration(letra, referencia)
                trainingStore.saveStaticSamples(letra, frames)
                trainingStore.incrementSampleCount(letra, frames.size)
            }

            calibrationTarget = null
            calibrationBuffer.clear()
            onFeedback("$letra SALVA PARA TREINO", LibrasAnalyzer.FEEDBACK_BOM)
            return true
        }
    }

    fun cancelCalibration() {
        synchronized(calibrationLock) {
            calibrationTarget = null
            calibrationBuffer.clear()
        }
    }

    fun getCalibrationCount(): Int {
        synchronized(calibrationLock) {
            return trainingStore.calibrationCount()
        }
    }

    fun getCalibrationFrameCount(): Int {
        synchronized(calibrationLock) {
            return calibrationBuffer.size
        }
    }

    fun getTrainingSampleCount(letra: String? = null): Int {
        if (letra != null) {
            return trainingStore.sampleCount(letra)
        }
        return trainingStore.totalSampleCount()
    }

    fun getTrainingDatasetPath(): String = trainingStore.staticDatasetPath()

    fun getDynamicTrainingDatasetPath(): String = trainingStore.dynamicDatasetPath()

    fun clearTrainingData() {
        synchronized(calibrationLock) {
            calibrationTarget = null
            calibrationBuffer.clear()
            trainingStore.clear()
        }
        onFeedback("TREINO ZERADO", LibrasAnalyzer.FEEDBACK_NEUTRO)
    }

    fun close() {
        modelosEstaticosIndividuais.forEach { it.session.close() }
        modelosDinamicosIndividuais.forEach { it.session.close() }
        sessionEstatico.close()
        sessionDinamicoParcial?.close()
        sessionDinamico.close()
        // OrtEnvironment.getEnvironment() retorna um ambiente compartilhado.
        // Fechar aqui quebra a proxima criacao do LetraEngine quando a camera
        // e aberta de novo ou quando o CameraX recria o analyzer.
    }
}

