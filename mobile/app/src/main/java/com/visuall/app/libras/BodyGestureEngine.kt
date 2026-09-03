package com.visuall.app.libras

import android.content.Context
import android.util.Log
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker.PoseLandmarkerOptions
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder

// Reconhecimento de sinais de corpo (pose + mãos + modelo TFLite) — extraído
// do LibrasAnalyzer pra isolar a máquina de estado de captura/classificação
// e a tradução de frase do resto do pipeline (letra, marcador de rosto). O
// gesto de "mão aberta limpa a frase" e a frase compartilhada continuam no
// LibrasAnalyzer, porque são idênticos ao do modo alfabeto.
internal class BodyGestureEngine(private val context: Context) {

    private enum class BodyState { OCIOSO, CAPTURANDO }

    data class Frame(
        val points: FloatArray,
        val hasHand: Boolean,
        val hasPose: Boolean,
        val hasLeftHand: Boolean,
        val hasRightHand: Boolean
    )

    private var poseLandmarker: PoseLandmarker? = null
    private var flexDelegate: FlexDelegate? = null
    private var bodyInterpreter: Interpreter? = null
    private var labelsCorpo: List<String> = emptyList()
    // Null enquanto nada falhou. Quando o modo corpo não consegue carregar,
    // guarda o motivo pra ser mostrado na tela em vez de o app ficar mudo.
    var motivoFalha: String? = null
        private set

    private val bodyMovementBuffer = ArrayDeque<FloatArray>()
    private val bodyGestureBuffer  = ArrayList<FloatArray>()
    private var bodyState          = BodyState.OCIOSO

    // Verdadeiro enquanto um sinal esta sendo gravado. O gesto de limpar a
    // frase (LibrasAnalyzer) consulta isto pra nao contar durante um sinal --
    // ver ClearGestureGate. Vale a decisao do quadro anterior, porque quem
    // pergunta decide antes de chamar processarFrame; um quadro de atraso nao
    // muda nada diante dos 3s que a limpeza exige.
    val gestoEmAndamento: Boolean
        get() = bodyState == BodyState.CAPTURANDO

    private var ultimoNormalizado: FloatArray? = null
    // Ultima posicao conhecida de cada slot de mao, pra cobrir os quadros em
    // que o MediaPipe perde a mao por um instante. Ver preencheMaoPerdida.
    private val ultimaMao = arrayOfNulls<FloatArray>(2)
    private val quadrosSemMao = intArrayOf(0, 0)
    private var ultimoMovimento    = 0f
    private var bodyStartCount     = 0
    private var bodyEndCount       = 0
    private var capturaComecouEm   = 0L
    private var ultimoTempoCorpo   = 0L
    private var ultimaClasseCorpo  = ""
    private var bodyNoHandSince    = 0L
    // Sinais reconhecidos em sequência (rótulos: PESSOA, SURDO...). A frase é
    // montada re-traduzindo ESTA lista inteira (concordância de artigo/
    // gênero/verbo), igual ao traduzir_frase do Python — não dá pra montar
    // só concatenando palavras.
    private val bodyTokens = ArrayList<String>()

    private fun poseOptions(delegate: Delegate) = PoseLandmarkerOptions.builder()
        .setBaseOptions(BaseOptions.builder()
            .setModelAssetPath("pose_landmarker_lite.task")
            .setDelegate(delegate)
            .build())
        .setRunningMode(RunningMode.VIDEO)
        .setMinPoseDetectionConfidence(0.35f)
        .setMinPosePresenceConfidence(0.35f)
        .setMinTrackingConfidence(0.35f)
        .build()

    // GPU primeiro (mesma ideia do HandLandmarker em LibrasAnalyzer); se o
    // aparelho não aceitar, cai pra CPU antes de propagar o erro pro catch
    // de fora, que já desativa o modo corpo de forma graciosa.
    private fun createPoseLandmarker(): PoseLandmarker {
        return try {
            PoseLandmarker.createFromOptions(context, poseOptions(Delegate.GPU))
        } catch (e: Throwable) {
            Log.w("BodyGestureEngine", "GPU indisponivel pro PoseLandmarker, usando CPU", e)
            PoseLandmarker.createFromOptions(context, poseOptions(Delegate.CPU))
        }
    }

    fun ensureLoaded(): PoseLandmarker? {
        poseLandmarker?.let { return it }

        return try {
            val loadedPose = createPoseLandmarker()
            val loadedDelegate = FlexDelegate()
            val loadedInterpreter = Interpreter(
                loadAssetBuffer("gestos/geral/model.tflite"),
                Interpreter.Options().addDelegate(loadedDelegate)
            )
            loadedInterpreter.resizeInput(
                0, intArrayOf(1, LibrasAnalyzer.BODY_WINDOW, LibrasAnalyzer.BODY_FEATURES))
            loadedInterpreter.allocateTensors()

            labelsCorpo = context.assets.open("gestos/geral/labels.txt")
                .bufferedReader().readLines().filter { it.isNotBlank() }
            flexDelegate = loadedDelegate
            bodyInterpreter = loadedInterpreter
            poseLandmarker = loadedPose
            loadedPose
        } catch (error: Throwable) {
            poseLandmarker = null
            bodyInterpreter?.close()
            bodyInterpreter = null
            flexDelegate?.close()
            flexDelegate = null
            labelsCorpo = emptyList()
            // Antes isso falhava em silêncio total: o modo corpo simplesmente
            // não reconhecia nada e, de fora, parecia que a pessoa estava
            // fazendo o sinal errado. Guardar o motivo (e mostrar na tela, ver
            // motivoFalha) é o que transforma "não funciona" em algo
            // diagnosticável — importante agora que os modelos passam a ser
            // regerados com frequência pela ferramenta de treino.
            motivoFalha = descreverFalha(error)
            Log.e("BodyGestureEngine", "Falha ao carregar o modo corpo: $motivoFalha", error)
            null
        }
    }

    // Traduz a exceção pra algo que aponte o que fazer. Os dois casos comuns
    // depois de um retreino são o arquivo não ter sido gerado e o modelo ter
    // saído num formato que o app não consegue usar.
    private fun descreverFalha(error: Throwable): String = when {
        error is java.io.FileNotFoundException ->
            "arquivo do modelo de gestos não encontrado em assets/gestos/geral/"
        error.message?.contains("resize", ignoreCase = true) == true ||
            error.message?.contains("shape", ignoreCase = true) == true ->
            "modelo de gestos com formato inesperado (esperado [1, " +
                "${LibrasAnalyzer.BODY_WINDOW}, ${LibrasAnalyzer.BODY_FEATURES}])"
        else -> error.message ?: error::class.java.simpleName
    }

    fun extractFrame(
        handResult: HandLandmarkerResult,
        poseResult: PoseLandmarkerResult,
        aspectX: Float
    ): Frame {
        val frame = FloatArray(LibrasAnalyzer.BODY_FEATURES)
        val poses = poseResult.landmarks()
        val hasPose = poses.isNotEmpty()
        if (poses.isNotEmpty()) {
            poses[0].take(LibrasAnalyzer.BODY_POSE_POINTS).forEachIndexed { index, lm ->
                writeBodyPoint(frame, index, lm.x() * aspectX, lm.y(), lm.z())
            }
        }

        var hasHand = false
        var hasLeftHand = false
        var hasRightHand = false
        handResult.landmarks().forEachIndexed { handIndex, landmarks ->
            val handedness = handResult.handedness().getOrNull(handIndex)
                ?.firstOrNull()?.categoryName().orEmpty()
            // ATENCAO: o rotulo vem INVERTIDO em relacao ao pipeline de treino.
            //
            // O treino extrai os landmarks com o MediaPipe Solutions (Python) e
            // o app usa o MediaPipe Tasks. Medido na mesma cena, com a mao
            // DIREITA sozinha erguida:
            //    video (Solutions):  Left@esq   Right@dir
            //    app   (Tasks):      Right@esq  Left@dir
            // Coordenadas iguais, rotulos trocados. Sem esta inversao cada mao
            // cai no slot da outra e o modelo recebe o gesto espelhado nas
            // maos.
            //
            // So aparece num sinal de duas maos assimetrico: rodando os 200
            // clipes com os slots trocados de proposito, COMPUTADOR vai de
            // 32/32 pra 0/32 (29 viram CONVERSAR) enquanto PESSOA fica em
            // 29/30 e SURDO em 30/37 -- exatamente o relato do aparelho, "so o
            // COMPUTADOR nao sai, e sai CONVERSAR".
            // Ver treino/diagnostico/testa_troca_maos.py.
            //
            // O desempate por avgX abaixo NAO se inverte: e coordenada, e as
            // coordenadas ja concordam entre os dois lados.
            val offset = if (handedness.equals("Right", ignoreCase = true)) {
                LibrasAnalyzer.BODY_POSE_POINTS
            } else if (handedness.equals("Left", ignoreCase = true)) {
                LibrasAnalyzer.BODY_POSE_POINTS + LibrasAnalyzer.BODY_HAND_POINTS
            } else {
                val avgX = landmarks.map { it.x() }.average()
                if (avgX < 0.5) LibrasAnalyzer.BODY_POSE_POINTS
                else LibrasAnalyzer.BODY_POSE_POINTS + LibrasAnalyzer.BODY_HAND_POINTS
            }
            landmarks.take(LibrasAnalyzer.BODY_HAND_POINTS).forEachIndexed { index, lm ->
                writeBodyPoint(frame, offset + index, lm.x() * aspectX, lm.y(), lm.z())
            }
            hasHand = true
            if (offset == LibrasAnalyzer.BODY_POSE_POINTS) {
                hasLeftHand = true
            } else {
                hasRightHand = true
            }
        }

        val slotA = preencheMaoPerdida(frame, 0, LibrasAnalyzer.BODY_POSE_POINTS, hasLeftHand)
        val slotB = preencheMaoPerdida(
            frame, 1, LibrasAnalyzer.BODY_POSE_POINTS + LibrasAnalyzer.BODY_HAND_POINTS,
            hasRightHand)
        return Frame(frame, hasHand || slotA || slotB, hasPose, slotA, slotB)
    }

    // Quando o MediaPipe perde uma mao por alguns quadros, os 63 valores dela
    // ficam em ZERO -- a mao "salta" pra origem e volta. Como bodyMotion() e o
    // desvio padrao sobre uma janela de 5 quadros, esse salto vira um pico
    // enorme de movimento que nao existe de verdade.
    //
    // Medido no aparelho com as maos PARADAS (treino/diagnostico, diag3):
    // 81% dos quadros com movimento acima de 0.20 tinham mudanca no numero de
    // maos detectadas na janela, contra 3% dos demais. O efeito pratico era a
    // captura nunca encerrar (o movimento nao ficava 5 quadros abaixo do
    // limiar de parada) e, por tabela, a barra de limpar nunca completar.
    //
    // Nos 16623 quadros dos videos de treino a contagem de maos NUNCA muda no
    // meio de um clipe -- entao repetir a ultima posicao aproxima o app do que
    // o modelo aprendeu, em vez de afastar.
    //
    // O limite existe pra nao deixar uma mao fantasma: se ela sumiu de verdade,
    // depois de MAX_QUADROS_MAO_PERDIDA os zeros voltam.
    private fun preencheMaoPerdida(
        frame: FloatArray,
        slot: Int,
        pontoInicial: Int,
        detectada: Boolean
    ): Boolean {
        val base = pontoInicial * 3
        val tamanho = LibrasAnalyzer.BODY_HAND_POINTS * 3
        if (detectada) {
            ultimaMao[slot] = frame.copyOfRange(base, base + tamanho)
            quadrosSemMao[slot] = 0
            return true
        }
        val ultima = ultimaMao[slot] ?: return false
        if (quadrosSemMao[slot] >= LibrasAnalyzer.MAX_QUADROS_MAO_PERDIDA) {
            ultimaMao[slot] = null
            return false
        }
        quadrosSemMao[slot]++
        ultima.copyInto(frame, base)
        return true
    }

    private fun writeBodyPoint(frame: FloatArray, pointIndex: Int, x: Float, y: Float, z: Float) {
        val base = pointIndex * 3
        frame[base] = x
        frame[base + 1] = y
        frame[base + 2] = z
    }

    // Atualiza a janela de movimento com este quadro e devolve a magnitude
    // atual. Precisa ser chamado UMA vez por quadro, antes de processarFrame.
    //
    // Existe porque o gesto de "limpar a frase" (decidido no LibrasAnalyzer)
    // precisa saber se a mao esta parada, e essa decisao acontece antes de
    // processarFrame. Sem isto, o limpar so conseguia olhar se a mao estava
    // ABERTA -- e qualquer sinal feito de palma aberta, como AJUDAR, caia no
    // contador de limpar e nunca chegava a ser classificado.
    fun registrarMovimento(bodyFrame: Frame): Float {
        val normalized = LibrasMath.normalizeBodyFrame(bodyFrame.points)
        ultimoNormalizado = normalized
        bodyMovementBuffer.addLast(normalized)
        while (bodyMovementBuffer.size > LibrasAnalyzer.BODY_MOV_WINDOW) {
            bodyMovementBuffer.removeFirst()
        }
        ultimoMovimento = bodyMotion()
        return ultimoMovimento
    }

    private fun bodyMotion(): Float {
        if (bodyMovementBuffer.size < 3) return 0f
        var total = 0f
        var count = 0
        for (point in LibrasAnalyzer.BODY_POSE_POINTS until LibrasAnalyzer.BODY_TOTAL_POINTS) {
            for (coord in 0..1) {
                val values = bodyMovementBuffer.map { it[point * 3 + coord] }
                total += LibrasMath.std(values)
                count++
            }
        }
        return if (count == 0) 0f else total / count
    }

    // Chamado só quando já se sabe que há pose+mão no frame (o LibrasAnalyzer
    // trata o "sem corpo" e o gesto de "mão aberta limpa" antes de chamar
    // isto). Roda a máquina de estado de captura/classificação e, se um novo
    // sinal foi confirmado, devolve a frase re-traduzida (senão, null) — quem
    // chama decide se atualiza o campo compartilhado `frase` e dispara
    // onFraseUpdate.
    fun processarFrame(
        bodyFrame: Frame,
        agora: Long,
        onLetra: (letra: String, confianca: Float, modo: String) -> Unit,
        onFeedback: (mensagem: String, nivel: Int) -> Unit
    ): String? {
        bodyNoHandSince = 0L
        // A janela de movimento e o valor ja foram atualizados por
        // registrarMovimento(), que o LibrasAnalyzer chama antes de decidir o
        // gesto de limpar. Recalcular aqui contaria o mesmo quadro duas vezes.
        val normalized = ultimoNormalizado ?: LibrasMath.normalizeBodyFrame(bodyFrame.points)
        val movimento = ultimoMovimento
        var novaFrase: String? = null

        when (bodyState) {
            BodyState.OCIOSO -> {
                // Igual ao Python: basta ter mão no quadro e movimento acima
                // do limiar de início. As "trancas" extras de trajeto/
                // amplitude da mão foram removidas porque bloqueavam gestos
                // de menor amplitude.
                val gestoComecou = bodyFrame.hasHand &&
                    bodyMovementBuffer.size >= LibrasAnalyzer.BODY_MOV_WINDOW &&
                    movimento > LibrasAnalyzer.BODY_START_MOTION

                if (gestoComecou) {
                    bodyStartCount++
                    if (bodyStartCount >= LibrasAnalyzer.BODY_START_FRAMES) {
                        bodyState = BodyState.CAPTURANDO
                        capturaComecouEm = agora
                        bodyGestureBuffer.clear()
                        bodyGestureBuffer.addAll(bodyMovementBuffer)
                        bodyEndCount = 0
                        onFeedback("CAPTURANDO GESTO", LibrasAnalyzer.FEEDBACK_NEUTRO)
                    }
                } else {
                    bodyStartCount = 0
                    onFeedback("PRONTO - FAÇA O SINAL", LibrasAnalyzer.FEEDBACK_BOM)
                }
            }
            BodyState.CAPTURANDO -> {
                bodyGestureBuffer.add(normalized)
                bodyEndCount = if (movimento < LibrasAnalyzer.BODY_END_MOTION) bodyEndCount + 1 else 0

                if (bodyEndCount >= LibrasAnalyzer.BODY_END_FRAMES ||
                    bodyGestureBuffer.size >= LibrasAnalyzer.BODY_MAX_FRAMES ||
                    (agora - capturaComecouEm) >= LibrasAnalyzer.BODY_MAX_DURACAO_MS) {
                    val prediction = classify(bodyGestureBuffer)
                    bodyState = BodyState.OCIOSO
                    bodyStartCount = 0
                    bodyEndCount = 0
                    bodyGestureBuffer.clear()

                    if (prediction != null && isReliable(prediction)) {
                        onLetra(prediction.letra, prediction.confianca, "corpo")
                        onFeedback(
                            "CORPO: ${SentenceTranslator.traduzirCorpo(prediction.letra).uppercase()}",
                            LibrasAnalyzer.FEEDBACK_BOM
                        )
                        if (prediction.letra != "NEUTRO" &&
                            prediction.letra != ultimaClasseCorpo &&
                            agora - ultimoTempoCorpo > LibrasAnalyzer.BODY_COOLDOWN) {
                            bodyTokens.add(prediction.letra.uppercase())
                            novaFrase = SentenceTranslator.traduzirFrase(bodyTokens)
                            ultimoTempoCorpo = agora
                            ultimaClasseCorpo = prediction.letra
                        }
                    } else {
                        onLetra("-", 0f, "corpo")
                        onFeedback("REPITA MAIS DEVAGAR", LibrasAnalyzer.FEEDBACK_ALERTA)
                    }
                }
            }
        }

        if (bodyState == BodyState.OCIOSO) {
            onLetra("-", 0f, "corpo")
        }
        return novaFrase
    }

    private fun classify(frames: List<FloatArray>): Prediction? {
        val interpreter = bodyInterpreter ?: return null
        val labels = labelsCorpo
        if (frames.size < LibrasAnalyzer.BODY_MIN_FRAMES || labels.isEmpty()) return null

        val sampled = LibrasMath.resample(frames, LibrasAnalyzer.BODY_WINDOW)
        val input = Array(1) { Array(LibrasAnalyzer.BODY_WINDOW) { FloatArray(LibrasAnalyzer.BODY_FEATURES) } }
        sampled.forEachIndexed { index, frame ->
            frame.copyInto(input[0][index])
        }
        val output = Array(1) { FloatArray(labels.size) }
        interpreter.run(input, output)
        val probs = output[0]
        val idx = probs.indices.maxByOrNull { probs[it] } ?: return null
        val second = probs.indices
            .filter { it != idx }
            .maxOfOrNull { probs[it] } ?: 0f
        return Prediction(labels[idx], probs[idx], "corpo", probs[idx] - second)
    }

    private fun isReliable(prediction: Prediction): Boolean {
        // Python aceita a palavra apenas com confiança >= 0.85 (e != NEUTRO,
        // tratado em processarFrame). Sem exigência de margem.
        return prediction.letra != "-" && prediction.confianca >= LibrasAnalyzer.BODY_CONFIDENCE
    }

    // Só zera a máquina de estado de captura (buffers/contadores) — chamado
    // toda vez que a mão está aberta ou quando não há corpo no quadro.
    fun resetCapture() {
        ultimaMao[0] = null
        ultimaMao[1] = null
        quadrosSemMao[0] = 0
        quadrosSemMao[1] = 0
        bodyState = BodyState.OCIOSO
        bodyStartCount = 0
        bodyEndCount = 0
        bodyMovementBuffer.clear()
        bodyGestureBuffer.clear()
    }

    // Chamado quando não há pose ou mão no quadro (o LibrasAnalyzer já
    // decidiu isso). Zera a classe anterior depois de 1.5s sem corpo, igual
    // ao original.
    fun onCorpoAusente(agora: Long) {
        resetCapture()
        bodyNoHandSince = if (bodyNoHandSince == 0L) agora else bodyNoHandSince
        if (agora - bodyNoHandSince > 1_500L) {
            ultimaClasseCorpo = ""
        }
    }

    // Limpa os sinais reconhecidos (frase de corpo) sem mexer na máquina de
    // captura — usado pelo gesto de "mão aberta limpa" e por limparFrase().
    fun limparTokens() {
        bodyTokens.clear()
        ultimaClasseCorpo = ""
    }

    // Remove o último sinal e re-traduz; retorna a nova frase, ou null se
    // não havia nada pra remover (nesse caso o chamador não deve nem
    // disparar onFraseUpdate, igual ao original).
    fun apagarUltimoToken(): String? {
        if (bodyTokens.isEmpty()) return null
        bodyTokens.removeAt(bodyTokens.size - 1)
        ultimaClasseCorpo = ""
        return SentenceTranslator.traduzirFrase(bodyTokens)
    }

    // Reset completo ao trocar de modo — igual ao setModo() original.
    fun resetTudo() {
        resetCapture()
        ultimaClasseCorpo = ""
        bodyNoHandSince = 0L
        bodyTokens.clear()
    }

    fun close() {
        runCatching { poseLandmarker?.close() }
            .onFailure { Log.w("BodyGestureEngine", "Falha ao fechar PoseLandmarker", it) }
        runCatching { bodyInterpreter?.close() }
            .onFailure { Log.w("BodyGestureEngine", "Falha ao fechar Interpreter", it) }
        runCatching { flexDelegate?.close() }
            .onFailure { Log.w("BodyGestureEngine", "Falha ao fechar FlexDelegate", it) }
        poseLandmarker = null
        bodyInterpreter = null
        flexDelegate = null
    }

    private fun loadAssetBuffer(assetName: String): ByteBuffer {
        val bytes = context.assets.open(assetName).use { it.readBytes() }
        return ByteBuffer.allocateDirect(bytes.size)
            .order(ByteOrder.nativeOrder())
            .put(bytes)
            .also { it.rewind() }
    }
}

