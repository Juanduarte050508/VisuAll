package com.visuall.app.libras

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
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
    // Estes labels.txt são a ÚNICA fonte da verdade sobre quais letras
    // existem: a ordem deles é o que traduz "saída número 3" na letra certa,
    // e eles são regravados junto do modelo a cada treino. Qualquer outra
    // lista de letras no app tem que sair daqui (ver labelsAlfabeto), senão
    // uma some/entra no treino e a tela continua mostrando a lista velha.
    private val labelsEstatico  = context.assets.open("letras_estaticas/geral/labels.txt")
        .bufferedReader().readLines().filter { it.isNotBlank() }
    private val labelsDinamico  = context.assets.open("letras_dinamicas/geral/labels.txt")
        .bufferedReader().readLines().filter { it.isNotBlank() }

    // Alfabeto completo (estáticas + dinâmicas), em ordem alfabética, pra UI
    // de calibração percorrer.
    val labelsAlfabeto: List<String> = (labelsEstatico + labelsDinamico).distinct().sorted()
    // Quais delas exigem movimento — usado pra instrução na tela ("segure" vs
    // "faça o movimento").
    val labelsDinamicasSet: Set<String> = labelsDinamico.toSet()
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

    // Só confiamos no modelo dinâmico depois de o movimento ser sustentado por
    // MOVIMENTO_SUSTENTADO_MS. A regra (e o histórico de por que ela é medida
    // em tempo e não em contagem de frames) está no MovementGate.
    private val movementGate = MovementGate()

    private data class ModeloIndividual(
        val label: String,
        val session: OrtSession
    )

    private fun carregarModelosIndividuais(
        context: Context,
        labels: List<String>,
        basePath: String
    ): List<ModeloIndividual> {
        val carregados = labels.mapNotNull { label ->
            val safe = label.uppercase().filter { it.isLetterOrDigit() || it == '_' || it == '-' }
            carregarOpcional(context, "$basePath/$safe/model.onnx") { bytes ->
                ModeloIndividual(label = label, session = ortEnv.createSession(bytes))
            }
        }
        Log.i("LetraEngine", "Modelos individuais em $basePath: " +
            if (carregados.isEmpty()) "nenhum" else carregados.joinToString { it.label })
        return carregados
    }

    // Carrega um asset que pode legitimamente não existir. A diferença que
    // importa: NÃO existir é normal e silencioso; existir e falhar ao abrir é
    // erro e vai pro log. Antes os dois casos eram engolidos igualmente, então
    // um modelo corrompido ou em formato errado ficava indistinguível de um
    // modelo que nunca foi treinado.
    private fun <T> carregarOpcional(
        context: Context,
        asset: String,
        build: (ByteArray) -> T
    ): T? {
        val bytes = try {
            context.assets.open(asset).use { entrada -> entrada.readBytes() }
        } catch (_: java.io.FileNotFoundException) {
            return null
        } catch (error: Throwable) {
            Log.w("LetraEngine", "Não consegui ler o asset $asset", error)
            return null
        }
        return try {
            build(bytes)
        } catch (error: Throwable) {
            Log.e("LetraEngine", "Asset $asset existe mas não pôde ser carregado " +
                "(modelo corrompido ou em formato incompatível?)", error)
            null
        }
    }

    fun resetMovimentoSustentado() {
        movementGate.reset()
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

    private fun calcularMovimento(): Float {
        if (bufferLm.size < 5) return 0f
        val recent = bufferLm.toList().takeLast(5)
        return LibrasMath.std(recent.map { it[2] }) + LibrasMath.std(recent.map { it[3] }) +
               LibrasMath.std(recent.map { it[16] }) + LibrasMath.std(recent.map { it[17] })
    }

    private fun escolherClassificacao(dados: FloatArray, movimento: Float): Prediction {
        // A histerese de movimento (por que ela existe e como se comporta) vive
        // no MovementGate, que é testável porque recebe o instante.
        val movimentoConfiavel = movementGate.avaliar(movimento, System.currentTimeMillis())

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
                confiancaMinima = LibrasAnalyzer.CONFIANCA_INDIVIDUAL,
                margemMinima = LibrasAnalyzer.MARGEM_ESTATICA_MINIMA,
                modo = "estatico_individual"
            )
            if (individual.letra != "-") return individual
        }

        return LetterDecision.deProbabilidades(
            probs = rodarModelo(sessionEstatico, dados, LibrasAnalyzer.FEATURES_ESTATICO),
            labels = labelsEstatico,
            confiancaMinima = LibrasAnalyzer.CONFIANCA_MINIMA,
            margemMinima = LibrasAnalyzer.MARGEM_ESTATICA_MINIMA,
            modo = "estatico"
        )
    }

    // Roda uma sessão ONNX e devolve o vetor de probabilidades. O índice [1] da
    // saída é a lista de probabilidades porque o treino exporta com
    // zipmap=False (ver verificar_onnx_exportado em treinar_visuall.py, que
    // falha o treino se isso mudar).
    private fun rodarModelo(session: OrtSession, entrada: FloatArray, features: Int): FloatArray {
        val tensor = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(entrada), longArrayOf(1, features.toLong()))
        val out = session.run(mapOf("landmarks_input" to tensor))
        @Suppress("UNCHECKED_CAST")
        val probs = (out[1].value as Array<FloatArray>)[0]
        tensor.close(); out.close()
        return probs
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
                confiancaMinima = LibrasAnalyzer.CONFIANCA_INDIVIDUAL,
                margemMinima = LibrasAnalyzer.MARGEM_DINAMICA_MINIMA,
                modo = "dinamico_individual"
            )
            if (individual.letra != "-") return individual
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
    ): Prediction = LetterDecision.deProbabilidades(
        probs = rodarModelo(session, entrada, LibrasAnalyzer.FEATURES_DINAMICO),
        labels = labels,
        confiancaMinima = LibrasAnalyzer.CONFIANCA_DINAMICA,
        margemMinima = LibrasAnalyzer.MARGEM_DINAMICA_MINIMA,
        modo = modo
    )

    private fun classificarIndividual(
        entrada: FloatArray,
        modelos: List<ModeloIndividual>,
        features: Int,
        confiancaMinima: Float,
        margemMinima: Float,
        modo: String
    ): Prediction {
        val pontuacoes = modelos.map { modelo ->
            val probs = rodarModelo(modelo.session, entrada, features)
            // Modelo binário: a saída [1] é "é esta letra". Alguns exports com
            // uma classe só devolvem um valor — aí ele é a própria resposta.
            val positivo = when {
                probs.size >= 2 -> probs[1]
                probs.size == 1 -> probs[0]
                else -> 0f
            }
            modelo.label to positivo
        }
        return LetterDecision.deModelosIndividuais(
            pontuacoes = pontuacoes,
            confiancaMinima = confiancaMinima,
            margemMinima = margemMinima,
            confiancaSemRival = LibrasAnalyzer.CONFIANCA_INDIVIDUAL_SEM_RIVAL,
            modo = modo
        )
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
                val referencia = LetterDecision.media(
                    frames, LibrasAnalyzer.FEATURES_ESTATICO
                )
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
        // Antes das sessões: espera terminar de gravar as amostras de
        // calibração, que agora vão pra disco em segundo plano.
        trainingStore.close()
        modelosEstaticosIndividuais.forEach { modelo ->
            runCatching { modelo.session.close() }
                .onFailure { Log.w("LetraEngine", "Falha ao fechar modelo individual ${modelo.label}", it) }
        }
        modelosDinamicosIndividuais.forEach { modelo ->
            runCatching { modelo.session.close() }
                .onFailure { Log.w("LetraEngine", "Falha ao fechar modelo individual ${modelo.label}", it) }
        }
        runCatching { sessionEstatico.close() }
            .onFailure { Log.w("LetraEngine", "Falha ao fechar modelo estatico", it) }
        runCatching { sessionDinamico.close() }
            .onFailure { Log.w("LetraEngine", "Falha ao fechar modelo dinamico", it) }
        // OrtEnvironment.getEnvironment() retorna um ambiente compartilhado.
        // Fechar aqui quebra a proxima criacao do LetraEngine quando a camera
        // e aberta de novo ou quando o CameraX recria o analyzer.
    }
}

