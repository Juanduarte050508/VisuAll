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
import kotlin.math.sqrt

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

    private val bodyMovementBuffer = ArrayDeque<FloatArray>()
    private val bodyGestureBuffer  = ArrayList<FloatArray>()
    private var bodyState          = BodyState.OCIOSO
    private var bodyStartCount     = 0
    private var bodyEndCount       = 0
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
            null
        }
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
            val offset = if (handedness.equals("Left", ignoreCase = true)) {
                LibrasAnalyzer.BODY_POSE_POINTS
            } else if (handedness.equals("Right", ignoreCase = true)) {
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

        return Frame(frame, hasHand, hasPose, hasLeftHand, hasRightHand)
    }

    private fun writeBodyPoint(frame: FloatArray, pointIndex: Int, x: Float, y: Float, z: Float) {
        val base = pointIndex * 3
        frame[base] = x
        frame[base + 1] = y
        frame[base + 2] = z
    }

    private fun normalize(frame: FloatArray): FloatArray {
        val normalized = frame.copyOf()
        val leftShoulder = LibrasAnalyzer.BODY_POINT_LEFT_SHOULDER * 3
        val rightShoulder = LibrasAnalyzer.BODY_POINT_RIGHT_SHOULDER * 3
        val centerX = (frame[leftShoulder] + frame[rightShoulder]) / 2f
        val centerY = (frame[leftShoulder + 1] + frame[rightShoulder + 1]) / 2f
        // A escala é a distância entre os ombros em 3D — o Python faz
        // np.linalg.norm(frame[11] - frame[12]) sobre vetores (x,y,z), então o
        // dz ENTRA na conta. Usar só dx/dy deixava todas as features do corpo
        // numa escala diferente da do treino.
        val dx = frame[leftShoulder] - frame[rightShoulder]
        val dy = frame[leftShoulder + 1] - frame[rightShoulder + 1]
        val dz = frame[leftShoulder + 2] - frame[rightShoulder + 2]
        val scale = sqrt(dx * dx + dy * dy + dz * dz).takeIf { it > 0.0001f } ?: 1f
        for (point in 0 until LibrasAnalyzer.BODY_TOTAL_POINTS) {
            val base = point * 3
            normalized[base] = (normalized[base] - centerX) / scale
            normalized[base + 1] = (normalized[base + 1] - centerY) / scale
        }
        return normalized
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
        val normalized = normalize(bodyFrame.points)
        bodyMovementBuffer.addLast(normalized)
        while (bodyMovementBuffer.size > LibrasAnalyzer.BODY_MOV_WINDOW) bodyMovementBuffer.removeFirst()
        val movimento = bodyMotion()
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
                    bodyGestureBuffer.size >= LibrasAnalyzer.BODY_MAX_FRAMES) {
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

        val sampled = resample(frames, LibrasAnalyzer.BODY_WINDOW)
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

    private fun resample(frames: List<FloatArray>, count: Int): List<FloatArray> {
        if (frames.size == count) return frames
        return List(count) { index ->
            val sourceIndex = ((frames.size - 1) * index.toFloat() / (count - 1)).toInt()
            frames[sourceIndex]
        }
    }

    // Só zera a máquina de estado de captura (buffers/contadores) — chamado
    // toda vez que a mão está aberta ou quando não há corpo no quadro.
    fun resetCapture() {
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
        poseLandmarker?.close()
        bodyInterpreter?.close()
        flexDelegate?.close()
    }

    private fun loadAssetBuffer(assetName: String): ByteBuffer {
        val bytes = context.assets.open(assetName).use { it.readBytes() }
        return ByteBuffer.allocateDirect(bytes.size)
            .order(ByteOrder.nativeOrder())
            .put(bytes)
            .also { it.rewind() }
    }
}

