package com.visuall.app.libras

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker.HandLandmarkerOptions
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker.PoseLandmarkerOptions
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.ByteOrder
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.sqrt

class LibrasAnalyzer(
    private val context: Context,
    private val onLetra: (letra: String, confianca: Float, modo: String) -> Unit,
    private val onFraseUpdate: (frase: String) -> Unit,
    private val onNoHand: () -> Unit,
    private val onGestoLimpar: (progresso: Float) -> Unit,
    private val onRepeticaoPendente: (letra: String?) -> Unit,
    private val onFeedback: (mensagem: String, nivel: Int) -> Unit,
    // Landmarks crus (normalizados 0..1 no espaço do preview) para desenhar as
    // linhas de reconhecimento. hands = lista de mãos (21 pontos x,y cada);
    // pose = 33 pontos x,y (só no modo corpo) ou null; frameAspect = proporção
    // (largura/altura) da imagem analisada, para o overlay alinhar com o preview.
    private val onLandmarks: (
        hands: List<FloatArray>, pose: FloatArray?, frameAspect: Float
    ) -> Unit = { _, _, _ -> }
) : ImageAnalysis.Analyzer {

    companion object {
        const val JANELA_MLP             = 10
        // Lado menor da imagem enviada ao MediaPipe. O preview continua na
        // resolução da câmera; este valor só reduz o bitmap analisado. 255 tem
        // ~50% da área de 360, o que dobra aproximadamente a taxa efetiva de
        // frames processados quando o gargalo é MediaPipe/ONNX.
        const val INPUT_SHORT_SIDE       = 255
        // O MLP é "superconfiante": cospe ~0.99 quase sempre, então só a
        // confiança filtra muito pouco. A MARGEM (1ª menos 2ª opção) é o
        // critério que realmente separa um sinal claro de um chute.
        const val CONFIANCA_MINIMA       = 0.90f
        const val MARGEM_ESTATICA_MINIMA = 0.25f
        // O modelo dinâmico só conhece 5 classes (H,J,K,X,Z) e não tem classe
        // "nenhuma". Mantemos margem, mas sem bloquear o gesto: letras com
        // movimento aparecem por poucos frames e precisam ser aceitas rápido.
        const val CONFIANCA_DINAMICA     = 0.90f
        const val MARGEM_DINAMICA_MINIMA = 0.20f
        // Mesmo limiar do backend Python de referência. Com 0.55, movimentos
        // reais de H/J/K/X/Z ficavam frequentemente presos no modelo estático.
        const val LIMIAR_MOVIMENTO       = 0.30f
        const val TEMPO_PRA_LIMPAR       = 3_000L
        // Dinâmicas são transitórias; exigir muitos frames consecutivos faz a
        // janela passar do gesto antes da letra ser adicionada.
        const val ESTAB_MIN_DINAMICO     = 3
        const val ESTAB_MIN_ESTATICO     = 8
        const val COOLDOWN_DINAMICO      = 250L
        const val COOLDOWN_ESTATICO      = 450L
        const val NO_HAND_TOLERANCE      = 3
        const val FEATURES_ESTATICO      = 42
        const val FEATURES_DINAMICO      = 420
        const val BODY_POSE_POINTS       = 33
        const val BODY_HAND_POINTS       = 21
        const val BODY_TOTAL_POINTS      = BODY_POSE_POINTS + BODY_HAND_POINTS * 2
        const val BODY_FEATURES          = BODY_TOTAL_POINTS * 3
        const val BODY_WINDOW            = 30
        const val BODY_MOV_WINDOW        = 5
        const val BODY_POINT_LEFT_SHOULDER = 11
        const val BODY_POINT_RIGHT_SHOULDER = 12
        // Valores de referência do pipeline Python (modo corpo):
        // LIMIAR_INICIO=0.050, LIMIAR_FIM=0.030, CONFIANCA_CORPO=0.85, cooldown 2s.
        const val BODY_START_MOTION      = 0.050f
        const val BODY_END_MOTION        = 0.030f
        const val BODY_START_FRAMES      = 3
        const val BODY_END_FRAMES        = 5
        const val BODY_MIN_FRAMES        = 10
        const val BODY_MAX_FRAMES        = 60
        const val BODY_CONFIDENCE        = 0.85f
        const val BODY_COOLDOWN          = 2_000L
        const val CALIBRATION_MIN_FRAMES = 8
        const val CALIBRATION_MAX_FRAMES = 45
        const val CALIBRATION_TARGET_FRAMES = 24
        const val TRAINING_BASIC_TARGET_SAMPLES = CALIBRATION_TARGET_FRAMES
        const val TRAINING_STRONG_TARGET_SAMPLES = 96
        const val CALIBRATION_MATCH_LIMIT = 0.18f
        const val FEEDBACK_NEUTRO = 0
        const val FEEDBACK_BOM = 1
        const val FEEDBACK_ALERTA = 2
        const val TRAINING_FILE_NAME = "visuall_libras_phone_dataset.csv"
        const val DYNAMIC_TRAINING_FILE_NAME = "visuall_libras_dynamic_phone_dataset.csv"
        val LETRAS_REPETICAO_AUTO        = setOf("S", "R")
    }

    enum class Modo {
        ALFABETO,
        CORPO
    }

    private enum class BodyState {
        OCIOSO,
        CAPTURANDO
    }

    private data class Prediction(
        val letra: String,
        val confianca: Float,
        val modo: String,
        val margem: Float = 1f
    )

    private val handLandmarker: HandLandmarker
    private var poseLandmarker: PoseLandmarker? = null
    private var flexDelegate: FlexDelegate? = null
    private var bodyInterpreter: Interpreter? = null
    private var labelsCorpo: List<String> = emptyList()
    @Volatile private var modoAtual = Modo.ALFABETO
    // Espelha a imagem na horizontal quando estamos na câmera frontal. O
    // dataset foi gravado com a webcam espelhada (cv2.flip no Python), então
    // a câmera frontal precisa do mesmo espelhamento para o "lado" da mão
    // bater com o treino. Câmera traseira não é espelhada por natureza.
    @Volatile private var espelharImagem = true

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .build()

        val options = HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setNumHands(2)
            // Reduzido de 0.5 → 0.4: facilita a detecção inicial da mão,
            // principalmente em ângulos ou iluminação menos ideais.
            .setMinHandDetectionConfidence(0.4f)
            .setMinHandPresenceConfidence(0.4f)
            .setMinTrackingConfidence(0.4f)
            // VIDEO em vez de IMAGE: mantém rastreamento temporal entre frames.
            // Isso reduz o "tremor" dos landmarks (jitter) e a latência, porque
            // o detector reaproveita a região da mão do frame anterior em vez de
            // refazer a detecção completa toda vez — igual ao Holistic do Python.
            .setRunningMode(RunningMode.VIDEO)
            .build()

        handLandmarker = HandLandmarker.createFromOptions(context, options)
    }

    private val ortEnv          = OrtEnvironment.getEnvironment()
    private val sessionEstatico = ortEnv.createSession(
        context.assets.open("static_model.onnx").readBytes()
    )
    private val sessionDinamico = ortEnv.createSession(
        context.assets.open("dynamic_model.onnx").readBytes()
    )
    private val labelsEstatico  = context.assets.open("static_labels.txt")
        .bufferedReader().readLines().filter { it.isNotBlank() }
    private val labelsDinamico  = context.assets.open("dynamic_labels.txt")
        .bufferedReader().readLines().filter { it.isNotBlank() }
    private val trainingStore = CalibrationTrainingStore(context, labelsEstatico, labelsDinamico)

    private val bufferLm              = ArrayDeque<FloatArray>()
    private val bodyMovementBuffer    = ArrayDeque<FloatArray>()
    private val bodyGestureBuffer     = ArrayList<FloatArray>()
    private val calibrationLock       = Any()
    private val calibrationBuffer     = ArrayList<FloatArray>()
    @Volatile private var calibrationTarget: String? = null
    private var ultimaPredicao        = ""
    private var contadorEstabilidade  = 0
    private var ultimaLetraAdicionada = ""
    private var ultimoTempoAdicao     = 0L
    private var tempoInicioEsticado   = 0L
    private var ultimoTempoLimpar     = 0L
    private var frase                 = ""
    private var framesSemMao          = 0
    private var letraRepetidaPendente = ""
    private var bodyState             = BodyState.OCIOSO
    private var bodyStartCount        = 0
    private var bodyEndCount          = 0
    private var ultimoTempoCorpo      = 0L
    private var ultimaClasseCorpo     = ""
    private var bodyNoHandSince       = 0L
    // Sinais de corpo reconhecidos, em sequência (rótulos: PESSOA, SURDO...).
    // A frase é montada traduzindo ESTA lista (com artigo/gênero/verbo), igual
    // ao traduzir_frase do Python — não dá pra fazer só concatenando palavras.
    private val bodyTokens            = ArrayList<String>()
    // Timestamp monotônico exigido pelo modo VIDEO do MediaPipe. Precisa ser
    // estritamente crescente a cada frame para o rastreamento funcionar.
    private var videoTimestamp        = 0L
    // Fator que corrige a proporção do quadro retrato do celular para a 4:3
    // usada no treino (multiplica o x). Substitui o antigo letterbox: em vez
    // de distorcer a imagem, corrigimos só as features. Calculado por frame.
    private var aspectX               = 0.5625f
    // Proporção real (largura/altura) do frame analisado, repassada ao overlay.
    private var frameAspect           = 0.75f

    private fun nextVideoTimestamp(): Long {
        val now = SystemClock.uptimeMillis()
        videoTimestamp = if (now <= videoTimestamp) videoTimestamp + 1 else now
        return videoTimestamp
    }

    @ExperimentalGetImage
    override fun analyze(imageProxy: ImageProxy) {
        // ── Converter YUV → RGBA_8888 (exigido pelo MediaPipe) ────────────
        val bitmap = imageProxy.toBitmap()
        val preparedBitmap = prepararBitmap(
            bitmap, imageProxy.imageInfo.rotationDegrees.toFloat(), espelharImagem)
        val mpImage = BitmapImageBuilder(preparedBitmap).build()
        // Corrige o x para a proporção 4:3 do treino (quadro retrato -> 4:3).
        frameAspect = preparedBitmap.width.toFloat() / preparedBitmap.height
        aspectX = 0.75f * frameAspect

        val timestamp = nextVideoTimestamp()
        val result = handLandmarker.detectForVideo(mpImage, timestamp)

        if (modoAtual == Modo.CORPO) {
            val poseDetector = ensureBodyModelsLoaded()
            if (poseDetector == null) {
                onLetra("-", 0f, "corpo")
                onFeedback("MODELO DE CORPO INDISPONIVEL", FEEDBACK_ALERTA)
                imageProxy.close()
                return
            }
            val poseResult = poseDetector.detectForVideo(mpImage, timestamp)
            analisarCorpo(result, poseResult)
            imageProxy.close()
            return
        }

        if (result.landmarks().isEmpty()) {
            onLandmarks(emptyList(), null, frameAspect)
            framesSemMao++
            if (framesSemMao >= NO_HAND_TOLERANCE) {
                ultimaPredicao       = ""
                contadorEstabilidade = 0
                tempoInicioEsticado  = 0L
                bufferLm.clear()
                letraRepetidaPendente = ""
                // Libera a mesma letra para ser digitada de novo: tirar a mão
                // do quadro e refazer o sinal é a forma natural de repetir.
                ultimaLetraAdicionada = ""
                onRepeticaoPendente(null)
                onFeedback("MAO FORA DO QUADRO", FEEDBACK_ALERTA)
                onNoHand()
            }
            imageProxy.close()
            return
        }

        framesSemMao = 0
        onLandmarks(handsToArrays(result), null, frameAspect)
        val lms    = result.landmarks()[0]
        // x corrigido para 4:3 (features + geometria); o desenho usa o cru.
        val pontos = lms.map { Pair(it.x() * aspectX, it.y()) }
        val dedicosEsticados = detectarDedosEsticados(pontos)

        if (dedicosEsticados) {
            val agora = System.currentTimeMillis()
            if (tempoInicioEsticado == 0L) tempoInicioEsticado = agora
            val progresso = ((agora - tempoInicioEsticado).toFloat() / TEMPO_PRA_LIMPAR)
                .coerceIn(0f, 1f)
            onGestoLimpar(progresso)

            if ((agora - tempoInicioEsticado) >= TEMPO_PRA_LIMPAR
                && (agora - ultimoTempoLimpar) > 2_000L) {
                frase = ""; ultimaLetraAdicionada = ""; letraRepetidaPendente = ""
                tempoInicioEsticado = 0L; ultimoTempoLimpar = agora
                onRepeticaoPendente(null)
                onFraseUpdate("")
            }
        } else {
            tempoInicioEsticado = 0L
            onGestoLimpar(0f)

            val dados    = normalizeLandmarks(pontos)
            captureCalibrationFrame(dados)
            bufferLm.addLast(dados)
            while (bufferLm.size > JANELA_MLP + 5) bufferLm.removeFirst()

            val movimento = calcularMovimento()
            val predicao = escolherClassificacao(dados, movimento)
            val letra = predicao.letra
            val confianca = predicao.confianca
            val modo = predicao.modo
            emitirFeedbackAlfabeto(predicao, movimento)

            onLetra(letra, confianca, modo)

            if (letra != "-") {
                contadorEstabilidade = if (letra == ultimaPredicao) contadorEstabilidade + 1 else 1
                ultimaPredicao = letra
            } else {
                contadorEstabilidade = 0
                ultimaPredicao = ""
            }

            val agora    = System.currentTimeMillis()
            val estabMin = if (modo == "dinamico") ESTAB_MIN_DINAMICO else ESTAB_MIN_ESTATICO
            val cooldown = if (modo == "dinamico") COOLDOWN_DINAMICO  else COOLDOWN_ESTATICO

            if (contadorEstabilidade >= estabMin
                && letra != "-"
                && letra != ultimaLetraAdicionada
                && (agora - ultimoTempoAdicao) > cooldown) {
                if (frase.lastOrNull()?.toString() == letra) {
                    if (podeRepetirAutomaticamente(letra)) {
                        frase += letra
                        letraRepetidaPendente = ""
                        onRepeticaoPendente(null)
                        onFraseUpdate(frase)
                    } else {
                        letraRepetidaPendente = letra
                        onRepeticaoPendente(letra)
                    }
                } else {
                    frase += letra
                    letraRepetidaPendente = ""
                    onRepeticaoPendente(null)
                    onFraseUpdate(frase)
                }
                ultimaLetraAdicionada = letra
                ultimoTempoAdicao     = agora
                contadorEstabilidade  = 0
            }

            // NÃO limpamos ultimaLetraAdicionada por tempo: fazer isso digitava
            // a mesma letra repetidamente enquanto a mão ficava parada no sinal.
            // Ela só é liberada quando a mão sai do quadro (ver bloco sem mão)
            // ou pelo botão REPETIR, deixando a repetição sempre intencional.
        }

        imageProxy.close()
    }

    // ── Preparar o bitmap para o MediaPipe ────────────────────────────────
    // 1) reduz a resolução (lado menor -> INPUT_SHORT_SIDE, como o Python fazia
    //    com 480x360): a inferência do MediaPipe/ONNX é o gargalo, e o custo
    //    cresce com o tamanho da imagem. Reduzir acelera MUITO sem desalinhar o
    //    overlay (landmarks vêm normalizados 0..1) nem mudar as features;
    // 2) rotaciona para deixar a pessoa em pé;
    // 3) espelha na horizontal na câmera frontal (para casar com o dataset).
    // A proporção 4:3 do treino é corrigida depois, no nível das features
    // (ver aspectX).
    private fun prepararBitmap(src: Bitmap, degrees: Float, espelhar: Boolean): Bitmap {
        val shortSide = minOf(src.width, src.height)
        val escala = if (shortSide > INPUT_SHORT_SIDE) {
            INPUT_SHORT_SIDE.toFloat() / shortSide
        } else 1f
        val matrix = Matrix()
        if (escala != 1f) matrix.postScale(escala, escala)
        if (degrees != 0f) matrix.postRotate(degrees)
        if (espelhar) matrix.postScale(-1f, 1f)
        if (matrix.isIdentity) return src
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
    }

    fun setEspelhamento(cameraFrontal: Boolean) {
        espelharImagem = cameraFrontal
    }

    // Converte as mãos detectadas em arrays crus [x0,y0,x1,y1,...] normalizados
    // no espaço do preview, para o overlay desenhar.
    private fun handsToArrays(
        result: com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
    ): List<FloatArray> = result.landmarks().map { hand ->
        FloatArray(hand.size * 2).also { arr ->
            hand.forEachIndexed { i, lm -> arr[i * 2] = lm.x(); arr[i * 2 + 1] = lm.y() }
        }
    }

    private fun poseToArray(
        poseResult: com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
    ): FloatArray? {
        val poses = poseResult.landmarks()
        if (poses.isEmpty()) return null
        val p = poses[0]
        return FloatArray(p.size * 2).also { arr ->
            p.forEachIndexed { i, lm -> arr[i * 2] = lm.x(); arr[i * 2 + 1] = lm.y() }
        }
    }
    private fun ensureBodyModelsLoaded(): PoseLandmarker? {
        poseLandmarker?.let { return it }

        return try {
            val poseBaseOptions = BaseOptions.builder()
                .setModelAssetPath("pose_landmarker_lite.task")
                .build()
            val poseOptions = PoseLandmarkerOptions.builder()
                .setBaseOptions(poseBaseOptions)
                .setRunningMode(RunningMode.VIDEO)
                .setMinPoseDetectionConfidence(0.35f)
                .setMinPosePresenceConfidence(0.35f)
                .setMinTrackingConfidence(0.35f)
                .build()
            val loadedPose = PoseLandmarker.createFromOptions(context, poseOptions)
            val loadedDelegate = FlexDelegate()
            val loadedInterpreter = Interpreter(
                loadAssetBuffer("body_model.tflite"),
                Interpreter.Options().addDelegate(loadedDelegate)
            )
            loadedInterpreter.resizeInput(0, intArrayOf(1, BODY_WINDOW, BODY_FEATURES))
            loadedInterpreter.allocateTensors()

            labelsCorpo = context.assets.open("body_labels.txt")
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

    private fun normalizeLandmarks(pontos: List<Pair<Float, Float>>): FloatArray {
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

    private fun mirrorLandmarks(dados: FloatArray): FloatArray {
        val mirrored = dados.copyOf()
        for (i in mirrored.indices step 2) {
            mirrored[i] = -mirrored[i]
        }
        return mirrored
    }

    private fun captureCalibrationFrame(dados: FloatArray) {
        if (calibrationTarget == null) return
        synchronized(calibrationLock) {
            if (calibrationTarget == null) return
            if (calibrationBuffer.size >= CALIBRATION_MAX_FRAMES) {
                calibrationBuffer.removeAt(0)
            }
            calibrationBuffer.add(dados.copyOf())
            val letra = calibrationTarget.orEmpty()
            val total = calibrationBuffer.size
            onFeedback("GRAVANDO $letra  $total/$CALIBRATION_TARGET_FRAMES", FEEDBACK_BOM)
        }
    }

    private fun emitirFeedbackAlfabeto(prediction: Prediction, movimento: Float) {
        if (calibrationTarget != null) return

        val mensagem = when {
            prediction.letra != "-" && prediction.confianca >= 0.92f -> "SINAL ESTAVEL"
            prediction.letra != "-" && prediction.confianca >= 0.82f -> "SEGURE MAIS FIRME"
            movimento > LIMIAR_MOVIMENTO -> "MOVIMENTO ALTO"
            prediction.confianca >= 0.68f -> "QUASE: AJUSTE ANGULO"
            else -> "APROXIME A MAO"
        }
        val nivel = when {
            prediction.letra != "-" && prediction.confianca >= 0.90f -> FEEDBACK_BOM
            prediction.confianca >= 0.72f -> FEEDBACK_NEUTRO
            else -> FEEDBACK_ALERTA
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
                candidates = listOf(dados, mirrorLandmarks(dados)),
                maxDistance = CALIBRATION_MATCH_LIMIT
            )
        } ?: return Prediction("-", 0f, "calibrado")

        val score = (1f - match.distance / CALIBRATION_MATCH_LIMIT).coerceIn(0f, 1f)
        val confianca = (0.86f + score * 0.13f).coerceAtMost(0.99f)
        return Prediction(match.letter, confianca, "calibrado")
    }

    private fun averageFrames(frames: List<FloatArray>): FloatArray {
        val result = FloatArray(FEATURES_ESTATICO)
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

    private fun detectarDedosEsticados(lms: List<Pair<Float, Float>>): Boolean {
        val M = 0.06f
        return lms[8].second  < lms[5].second  - M &&
               lms[12].second < lms[9].second  - M &&
               lms[16].second < lms[13].second - M &&
               lms[20].second < lms[17].second - M &&
               abs(lms[4].first - lms[0].first) > 0.12f
    }

    private fun calcularMovimento(): Float {
        if (bufferLm.size < 5) return 0f
        val recent = bufferLm.toList().takeLast(5)
        return std(recent.map { it[2] }) + std(recent.map { it[3] }) +
               std(recent.map { it[16] }) + std(recent.map { it[17] })
    }

    private fun std(values: List<Float>): Float {
        val mean = values.average().toFloat()
        return sqrt(values.map { (it - mean) * (it - mean) }.average().toFloat())
    }

    private fun podeRepetirAutomaticamente(letra: String): Boolean {
        if (letra !in LETRAS_REPETICAO_AUTO) return false
        return frase.length < 2 || frase[frase.length - 2].toString() != letra
    }

    private fun escolherClassificacao(dados: FloatArray, movimento: Float): Prediction {
        // Mesma decisão do pipeline Python de referência: com movimento acima
        // do limiar e o buffer cheio, usamos o modelo dinâmico (H,J,K,X,Z);
        // caso contrário, o estático. Os dois nunca disputam no mesmo frame,
        // o que elimina a principal fonte de "letra parada virando outra".
        if (movimento > LIMIAR_MOVIMENTO && bufferLm.size >= JANELA_MLP) {
            return classificarDinamico()
        }
        // Calibração pessoal entra apenas como reforço quando o modelo não
        // reconheceu nada (ver aplicarCalibracaoPessoal).
        return aplicarCalibracaoPessoal(dados, classificarEstatico(dados))
    }

    private fun classificarEstatico(dados: FloatArray): Prediction {
        val tensor = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(dados), longArrayOf(1, FEATURES_ESTATICO.toLong()))
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
        val letra  = if (label != null && conf >= CONFIANCA_MINIMA &&
                         margem >= MARGEM_ESTATICA_MINIMA) label else "-"
        tensor.close(); out.close()
        return Prediction(letra, conf, "estatico", margem)
    }

    private fun classificarDinamico(): Prediction {
        val entrada = FloatArray(FEATURES_DINAMICO)
        bufferLm.toList().takeLast(JANELA_MLP).forEachIndexed { i, frame ->
            frame.copyInto(entrada, i * FEATURES_ESTATICO)
        }
        val tensor = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(entrada), longArrayOf(1, FEATURES_DINAMICO.toLong()))
        val out    = sessionDinamico.run(mapOf("landmarks_input" to tensor))
        @Suppress("UNCHECKED_CAST")
        val probs  = (out[1].value as Array<FloatArray>)[0]
        val idx    = probs.indices.maxByOrNull { probs[it] } ?: 0
        val conf   = probs[idx]
        val second = probs.indices.filter { it != idx }.maxOfOrNull { probs[it] } ?: 0f
        val margem = conf - second
        val letra  = if (conf >= CONFIANCA_DINAMICA && margem >= MARGEM_DINAMICA_MINIMA &&
                         idx < labelsDinamico.size) labelsDinamico[idx] else "-"
        tensor.close(); out.close()
        return Prediction(letra, conf, "dinamico", margem)
    }

    private fun analisarCorpo(
        handResult: com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult,
        poseResult: com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
    ) {
        onLandmarks(handsToArrays(handResult), poseToArray(poseResult), frameAspect)
        val bodyFrame = extractBodyFrame(handResult, poseResult)
        if (!bodyFrame.hasPose || !bodyFrame.hasHand) {
            resetBodyCapture()
            bodyNoHandSince = if (bodyNoHandSince == 0L) System.currentTimeMillis() else bodyNoHandSince
            if (System.currentTimeMillis() - bodyNoHandSince > 1_500L) {
                ultimaClasseCorpo = ""
            }
            onLetra("-", 0f, "corpo")
            onFeedback("ENQUADRE CORPO E MAO", FEEDBACK_ALERTA)
            onNoHand()
            return
        }

        // NÃO re-zeramos as mãos ausentes: o Python normaliza o frame inteiro
        // (incluindo os zeros das mãos que faltam), então o modelo foi treinado
        // com esses pontos deslocados pela normalização, não com zeros.
        val normalized = normalizeBodyFrame(bodyFrame.points)
        bodyMovementBuffer.addLast(normalized)
        while (bodyMovementBuffer.size > BODY_MOV_WINDOW) bodyMovementBuffer.removeFirst()

        val movimento = bodyMotion()
        val agora = System.currentTimeMillis()

        // Gesto de limpar: mão toda aberta e parada por TEMPO_PRA_LIMPAR limpa
        // a frase — o mesmo gesto do modo alfabeto (a barra vermelha mostra o
        // progresso). Responde à dúvida "manter a mão aberta reseta as palavras".
        val maoAberta = handResult.landmarks().firstOrNull()?.let { hand ->
            detectarDedosEsticados(hand.map { Pair(it.x() * aspectX, it.y()) })
        } ?: false
        if (maoAberta) {
            if (tempoInicioEsticado == 0L) tempoInicioEsticado = agora
            val progresso = ((agora - tempoInicioEsticado).toFloat() / TEMPO_PRA_LIMPAR)
                .coerceIn(0f, 1f)
            onGestoLimpar(progresso)
            if ((agora - tempoInicioEsticado) >= TEMPO_PRA_LIMPAR &&
                (agora - ultimoTempoLimpar) > 2_000L) {
                frase = ""; tempoInicioEsticado = 0L; ultimoTempoLimpar = agora
                ultimaClasseCorpo = ""
                bodyTokens.clear()
                onFraseUpdate("")
            }
            resetBodyCapture()
            onLetra("-", 0f, "corpo")
            return
        } else {
            tempoInicioEsticado = 0L
            onGestoLimpar(0f)
        }

        bodyNoHandSince = 0L

        when (bodyState) {
            BodyState.OCIOSO -> {
                // Igual ao Python: basta ter mão no quadro e movimento acima do
                // limiar de início. As "trancas" extras de trajeto/amplitude da
                // mão foram removidas porque bloqueavam gestos de menor amplitude.
                val gestoComecou = bodyFrame.hasHand &&
                    bodyMovementBuffer.size >= BODY_MOV_WINDOW &&
                    movimento > BODY_START_MOTION

                if (gestoComecou) {
                    bodyStartCount++
                    if (bodyStartCount >= BODY_START_FRAMES) {
                        bodyState = BodyState.CAPTURANDO
                        bodyGestureBuffer.clear()
                        bodyGestureBuffer.addAll(bodyMovementBuffer)
                        bodyEndCount = 0
                        onFeedback("CAPTURANDO GESTO", FEEDBACK_NEUTRO)
                    }
                } else {
                    bodyStartCount = 0
                    onFeedback("PRONTO - FAÇA O SINAL", FEEDBACK_BOM)
                }
            }
            BodyState.CAPTURANDO -> {
                bodyGestureBuffer.add(normalized)
                bodyEndCount = if (movimento < BODY_END_MOTION) bodyEndCount + 1 else 0

                if (bodyEndCount >= BODY_END_FRAMES || bodyGestureBuffer.size >= BODY_MAX_FRAMES) {
                    val prediction = classifyBodyGesture(bodyGestureBuffer)
                    bodyState = BodyState.OCIOSO
                    bodyStartCount = 0
                    bodyEndCount = 0
                    bodyGestureBuffer.clear()

                    if (prediction != null && isReliableBodyPrediction(prediction)) {
                        onLetra(prediction.letra, prediction.confianca, "corpo")
                        onFeedback("CORPO: ${traduzirCorpo(prediction.letra).uppercase()}", FEEDBACK_BOM)
                        if (prediction.letra != "NEUTRO" &&
                            prediction.letra != ultimaClasseCorpo &&
                            agora - ultimoTempoCorpo > BODY_COOLDOWN) {
                            // Guarda o rótulo e re-traduz a frase inteira, para a
                            // concordância (artigo/gênero/verbo) sair certa.
                            bodyTokens.add(prediction.letra.uppercase())
                            frase = traduzirFrase(bodyTokens)
                            ultimoTempoCorpo = agora
                            ultimaClasseCorpo = prediction.letra
                            onFraseUpdate(frase)
                        }
                    } else {
                        onLetra("-", 0f, "corpo")
                        onFeedback("REPITA MAIS DEVAGAR", FEEDBACK_ALERTA)
                    }
                }
            }
        }

        if (bodyState == BodyState.OCIOSO) {
            onLetra("-", 0f, "corpo")
        }
    }

    private data class BodyFrame(
        val points: FloatArray,
        val hasHand: Boolean,
        val hasPose: Boolean,
        val hasLeftHand: Boolean,
        val hasRightHand: Boolean
    )

    private fun extractBodyFrame(
        handResult: com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult,
        poseResult: com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
    ): BodyFrame {
        val frame = FloatArray(BODY_FEATURES)
        val poses = poseResult.landmarks()
        val hasPose = poses.isNotEmpty()
        if (poses.isNotEmpty()) {
            poses[0].take(BODY_POSE_POINTS).forEachIndexed { index, lm ->
                writeBodyPoint(frame, index, lm.x() * aspectX, lm.y(), lm.z())
            }
        }

        var hasHand = false
        var hasLeftHand = false
        var hasRightHand = false
        handResult.landmarks().forEachIndexed { handIndex, landmarks ->
            val handedness = handResult.handednesses().getOrNull(handIndex)
                ?.firstOrNull()?.categoryName().orEmpty()
            val offset = if (handedness.equals("Left", ignoreCase = true)) {
                BODY_POSE_POINTS
            } else if (handedness.equals("Right", ignoreCase = true)) {
                BODY_POSE_POINTS + BODY_HAND_POINTS
            } else {
                val avgX = landmarks.map { it.x() }.average()
                if (avgX < 0.5) BODY_POSE_POINTS else BODY_POSE_POINTS + BODY_HAND_POINTS
            }
            landmarks.take(BODY_HAND_POINTS).forEachIndexed { index, lm ->
                writeBodyPoint(frame, offset + index, lm.x() * aspectX, lm.y(), lm.z())
            }
            hasHand = true
            if (offset == BODY_POSE_POINTS) {
                hasLeftHand = true
            } else {
                hasRightHand = true
            }
        }

        return BodyFrame(frame, hasHand, hasPose, hasLeftHand, hasRightHand)
    }

    private fun writeBodyPoint(frame: FloatArray, pointIndex: Int, x: Float, y: Float, z: Float) {
        val base = pointIndex * 3
        frame[base] = x
        frame[base + 1] = y
        frame[base + 2] = z
    }

    private fun normalizeBodyFrame(frame: FloatArray): FloatArray {
        val normalized = frame.copyOf()
        val leftShoulder = BODY_POINT_LEFT_SHOULDER * 3
        val rightShoulder = BODY_POINT_RIGHT_SHOULDER * 3
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
        for (point in 0 until BODY_TOTAL_POINTS) {
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
        for (point in BODY_POSE_POINTS until BODY_TOTAL_POINTS) {
            for (coord in 0..1) {
                val values = bodyMovementBuffer.map { it[point * 3 + coord] }
                total += std(values)
                count++
            }
        }
        return if (count == 0) 0f else total / count
    }

    private fun classifyBodyGesture(frames: List<FloatArray>): Prediction? {
        val interpreter = bodyInterpreter ?: return null
        val labels = labelsCorpo
        if (frames.size < BODY_MIN_FRAMES || labels.isEmpty()) return null

        val sampled = resampleBodyFrames(frames, BODY_WINDOW)
        val input = Array(1) { Array(BODY_WINDOW) { FloatArray(BODY_FEATURES) } }
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

    private fun isReliableBodyPrediction(prediction: Prediction): Boolean {
        // Python aceita a palavra apenas com confiança >= 0.85 (e != NEUTRO,
        // tratado em analisarCorpo). Sem exigência de margem.
        return prediction.letra != "-" && prediction.confianca >= BODY_CONFIDENCE
    }

    private fun resetBodyCapture() {
        bodyState = BodyState.OCIOSO
        bodyStartCount = 0
        bodyEndCount = 0
        bodyMovementBuffer.clear()
        bodyGestureBuffer.clear()
    }

    private fun resampleBodyFrames(frames: List<FloatArray>, count: Int): List<FloatArray> {
        if (frames.size == count) return frames
        return List(count) { index ->
            val sourceIndex = ((frames.size - 1) * index.toFloat() / (count - 1)).toInt()
            frames[sourceIndex]
        }
    }

    private fun traduzirCorpo(label: String): String {
        return when (label.uppercase()) {
            "AJUDAR" -> "ajuda"
            "COMPUTADOR" -> "computador"
            "CONVERSAR" -> "conversa"
            "PESSOA" -> "pessoa"
            "SURDO" -> "surdo"
            else -> label.lowercase()
        }
    }

    // ── Tradução de frase (concordância) — portado do traduzir_frase do Python ─
    private data class VocabEntry(
        val tipo: String,          // "subst" | "adj" | "verbo"
        val genero: String? = null,
        val palavra: String? = null,
        val artigo: String? = null,
        val masc: String? = null,
        val fem: String? = null,
        val conj: String? = null,
        val inf: String? = null
    )

    private val vocabulario = mapOf(
        "PESSOA"     to VocabEntry(tipo = "subst", genero = "f", palavra = "pessoa", artigo = "a"),
        "SURDO"      to VocabEntry(tipo = "adj", masc = "surdo", fem = "surda"),
        "CONVERSAR"  to VocabEntry(tipo = "verbo", conj = "conversa", inf = "conversar"),
        "COMPUTADOR" to VocabEntry(tipo = "subst", genero = "m", palavra = "computador", artigo = "o"),
        "AJUDAR"     to VocabEntry(tipo = "verbo", conj = "ajuda", inf = "ajudar")
    )

    private fun traduzirFrase(palavras: List<String>): String {
        val partes = ArrayList<String>()
        var ultGen: String? = null
        var ultTipo: String? = null
        palavras.forEachIndexed { i, raw ->
            val p = raw.uppercase()
            if (p == "NEUTRO") return@forEachIndexed
            val v = vocabulario[p]
            if (v == null) {
                partes.add(p.lowercase().replaceFirstChar { it.uppercase() })
                ultGen = "m"; ultTipo = "subst"
                return@forEachIndexed
            }
            when (v.tipo) {
                "subst" -> {
                    val art = if (i == 0) {
                        v.artigo.orEmpty().replaceFirstChar { it.uppercase() }
                    } else {
                        v.artigo.orEmpty()
                    }
                    partes.add("$art ${v.palavra}")
                    ultGen = v.genero; ultTipo = "subst"
                }
                "adj" -> {
                    partes.add(if (ultGen == "f") v.fem.orEmpty() else v.masc.orEmpty())
                    ultTipo = "adj"
                }
                "verbo" -> {
                    partes.add(if (ultTipo == "verbo") "a ${v.inf}" else v.conj.orEmpty())
                    ultTipo = "verbo"
                }
            }
        }
        if (partes.isEmpty()) return ""
        return partes.joinToString(" ").replaceFirstChar { it.uppercase() }
    }

    fun setModo(novoModo: Modo) {
        modoAtual = novoModo
        bodyState = BodyState.OCIOSO
        bodyStartCount = 0
        bodyEndCount = 0
        bodyMovementBuffer.clear()
        bodyGestureBuffer.clear()
        ultimaClasseCorpo = ""
        bodyNoHandSince = 0L
        ultimaPredicao = ""
        contadorEstabilidade = 0
        letraRepetidaPendente = ""
        // Troca de modo começa uma frase nova (letras e sinais de corpo não se
        // misturam na mesma frase).
        bodyTokens.clear()
        ultimaLetraAdicionada = ""
        frase = ""
        onFraseUpdate("")
        onRepeticaoPendente(null)
        onLetra("-", 0f, novoModo.name.lowercase())
        if (novoModo == Modo.CORPO) {
            onFeedback("MODO CORPO: ENQUADRE TRONCO E MAO", FEEDBACK_NEUTRO)
        } else {
            onFeedback("MODO LIBRAS: CENTRALIZE A MAO", FEEDBACK_NEUTRO)
        }
    }

    fun startCalibration(letra: String) {
        synchronized(calibrationLock) {
            calibrationTarget = letra.uppercase()
            calibrationBuffer.clear()
        }
        onFeedback("CALIBRANDO ${letra.uppercase()}: SEGURE A MAO", FEEDBACK_NEUTRO)
    }

    fun finishCalibration(): Boolean {
        synchronized(calibrationLock) {
            val letra = calibrationTarget ?: return false
            if (calibrationBuffer.size < CALIBRATION_MIN_FRAMES) {
                onFeedback("POUCAS AMOSTRAS PARA $letra", FEEDBACK_ALERTA)
                return false
            }

            val frames = calibrationBuffer.map { it.copyOf() }
            if (letra in labelsDinamico) {
                val sequencias = trainingStore.saveDynamicSamples(letra, frames)
                if (sequencias == 0) {
                    onFeedback("MOVIMENTE MAIS PARA $letra", FEEDBACK_ALERTA)
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
            onFeedback("$letra SALVA PARA TREINO", FEEDBACK_BOM)
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

    fun getTrainingDatasetPath(): String {
        return trainingStore.staticDatasetPath()
    }

    fun getDynamicTrainingDatasetPath(): String {
        return trainingStore.dynamicDatasetPath()
    }

    fun clearTrainingData() {
        synchronized(calibrationLock) {
            calibrationTarget = null
            calibrationBuffer.clear()
            trainingStore.clear()
        }
        onFeedback("TREINO ZERADO", FEEDBACK_NEUTRO)
    }

    fun aplicarSugestao(palavra: String) {
        val sugestao = palavra.trim()
        if (sugestao.isBlank()) return

        val prefixo = frase.substringBeforeLast(" ", missingDelimiterValue = "")
        frase = if (prefixo.isBlank()) {
            "$sugestao "
        } else {
            "$prefixo $sugestao "
        }

        ultimaPredicao = ""
        contadorEstabilidade = 0
        ultimaLetraAdicionada = ""
        letraRepetidaPendente = ""
        onRepeticaoPendente(null)
        onFraseUpdate(frase)
    }

    fun adicionarEspaco()  { frase += " "; letraRepetidaPendente = ""; onRepeticaoPendente(null); onFraseUpdate(frase) }
    fun repetirLetraPendente() {
        if (letraRepetidaPendente.isNotBlank()) {
            frase += letraRepetidaPendente
            letraRepetidaPendente = ""
            ultimaLetraAdicionada = ""
            onRepeticaoPendente(null)
            onFraseUpdate(frase)
        }
    }
    fun apagarUltima() {
        if (modoAtual == Modo.CORPO) {
            // No corpo apagamos o último SINAL (token) e re-traduzimos.
            if (bodyTokens.isNotEmpty()) {
                bodyTokens.removeAt(bodyTokens.size - 1)
                frase = traduzirFrase(bodyTokens)
                ultimaClasseCorpo = ""
                onFraseUpdate(frase)
            }
            return
        }
        if (frase.isNotEmpty()) {
            frase = frase.dropLast(1)
            letraRepetidaPendente = ""; ultimaLetraAdicionada = ""
            onRepeticaoPendente(null); onFraseUpdate(frase)
        }
    }
    fun limparFrase() {
        frase = ""; letraRepetidaPendente = ""; ultimaLetraAdicionada = ""
        bodyTokens.clear(); ultimaClasseCorpo = ""
        onRepeticaoPendente(null); onFraseUpdate(frase)
    }
    fun getFrase(): String = frase

    fun close() {
        handLandmarker.close()
        poseLandmarker?.close()
        bodyInterpreter?.close()
        flexDelegate?.close()
        sessionEstatico.close()
        sessionDinamico.close()
        ortEnv.close()
    }

    private fun loadAssetBuffer(assetName: String): ByteBuffer {
        val bytes = context.assets.open(assetName).use { it.readBytes() }
        return ByteBuffer.allocateDirect(bytes.size)
            .order(ByteOrder.nativeOrder())
            .put(bytes)
            .also { it.rewind() }
    }

}


