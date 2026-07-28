package com.visuall.app.libras

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
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

// Orquestrador do reconhecimento Libras: cada frame da câmera passa por
// aqui, mas a inteligência de cada modo vive num módulo dedicado —
// LetraEngine (alfabeto estático/dinâmico + calibração), BodyGestureEngine
// (sinais de corpo) e FaceMarkerEngine (marcador de sobrancelha/pergunta).
// Esta classe cuida só do que é DE FATO compartilhado entre os modos: captura
// de câmera, detecção de mão (handLandmarker), e o estado da frase/gesto de
// limpar, que é idêntico nos dois modos.
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
    ) -> Unit = { _, _, _ -> },
    // Sobrancelha levantada (marcador de pergunta) — igual ao Python, entra
    // nos DOIS modos. Chamado só quando o estado muda (não a cada frame).
    private val onInterrogativo: (ativo: Boolean) -> Unit = { }
) : ImageAnalysis.Analyzer {

    companion object {
        const val JANELA_MLP             = 10
        // Lado menor da imagem enviada ao MediaPipe. O preview continua na
        // resolução da câmera; este valor só reduz o bitmap analisado.
        // Estava em 255 (baixado de 360, o valor do Python) para ganhar
        // velocidade, mas isso não foi validado com teste real de acurácia
        // em dispositivo — 255 pode prejudicar letras difíceis (mão
        // pequena/longe do quadro perde detalhe do landmark). 300 é um
        // meio-termo: ~31% menos área que 360 (ainda ganha velocidade
        // real), mas ~38% mais área que 255 (perde menos detalhe). Antes de
        // fixar o valor final, comparar a taxa de acerto das letras
        // difíceis (E, I, U, F, G, P, Q, T, V, W, Y) nos três valores num
        // celular real.
        const val INPUT_SHORT_SIDE       = 300
        // O MLP é "superconfiante": cospe ~0.99 quase sempre, então só a
        // confiança filtra muito pouco. A MARGEM (1ª menos 2ª opção) é o
        // critério que realmente separa um sinal claro de um chute.
        const val CONFIANCA_MINIMA       = 0.90f
        const val MARGEM_ESTATICA_MINIMA = 0.25f
        // O modelo dinâmico só conhece 5 classes (H,J,K,X,Z) e não tem classe
        // "nenhuma". Meio-termo entre o valor solto do Rafael (0.90/0.20) e o
        // apertado que eu tinha deixado (0.95/0.35) — ainda pendente de
        // validação real (ver LIMIAR_MOVIMENTO abaixo para a mudança principal
        // que ataca o falso-J).
        const val CONFIANCA_DINAMICA     = 0.92f
        const val MARGEM_DINAMICA_MINIMA = 0.28f
        // Eu e o Rafael tínhamos valores incompatíveis aqui: 0.30 (dele, igual
        // ao Python) deixava o J disparar com qualquer tremida; 0.55 (meu)
        // travava gestos reais de H/J/K/X/Z no modelo estático. Os dois
        // usavam a MESMA variável pra duas coisas diferentes: magnitude do
        // movimento E se ele é intencional. Separamos isso: LIMIAR_MOVIMENTO
        // volta a 0.30 (não perde gesto real, como o Rafael queria), mas só
        // é CONFIADO depois de sustentado por alguns frames seguidos
        // (MOVIMENTO_SUSTENTADO_FRAMES) — uma tremida de 1 frame não basta,
        // um traço real de H/J/K/X/Z (que dura vários frames) sim. Precisa
        // validar em celular real; se ainda sair J fácil, subir
        // MOVIMENTO_SUSTENTADO_FRAMES antes de mexer em LIMIAR_MOVIMENTO de novo.
        const val LIMIAR_MOVIMENTO       = 0.30f
        const val MOVIMENTO_SUSTENTADO_FRAMES = 3
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

        // Marcador de sobrancelha (frase vira pergunta) — portado 1:1 de
        // m01_visuall_config.py / app.py (ler_marcador). Os índices são da
        // topologia de 468 pontos do FaceMesh, a mesma usada pelo
        // FaceLandmarker da Tasks API — não precisou remapear nada.
        const val LIMIAR_SOBRANCELHA = 0.38f
        const val JANELA_SOBR = 5
        const val IDX_BROW_L = 105
        const val IDX_BROW_R = 334
        const val IDX_EYE_TOP_L = 159
        const val IDX_EYE_TOP_R = 386
        const val IDX_EYE_OUT_L = 33
        const val IDX_EYE_OUT_R = 263
        // A sobrancelha não muda de estado tão rápido quanto uma letra: rodar
        // o 3º modelo (FaceLandmarker) 1 a cada N frames economiza latência
        // sem atraso perceptível no marcador de pergunta.
        const val FACE_DETECT_STRIDE = 3
    }

    enum class Modo {
        ALFABETO,
        CORPO
    }

    private val handLandmarker: HandLandmarker
    private val letraEngine = LetraEngine(context, onFeedback)
    private val bodyEngine = BodyGestureEngine(context)
    private val faceEngine = FaceMarkerEngine(context)

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

    private var ultimaPredicao        = ""
    private var contadorEstabilidade  = 0
    private var ultimaLetraAdicionada = ""
    private var ultimoTempoAdicao     = 0L
    private var tempoInicioEsticado   = 0L
    private var ultimoTempoLimpar     = 0L
    private var frase                 = ""
    private var framesSemMao          = 0
    private var letraRepetidaPendente = ""
    // Timestamp monotônico exigido pelo modo VIDEO do MediaPipe. Precisa ser
    // estritamente crescente a cada frame para o rastreamento funcionar.
    private var videoTimestamp        = 0L
    // Fator que corrige a proporção do quadro retrato do celular para a 4:3
    // usada no treino (multiplica o x). Substitui o antigo letterbox: em vez
    // de distorcer a imagem, corrigimos só as features. Calculado por frame.
    private var aspectX               = 0.5625f
    // Proporção real (largura/altura) do frame analisado, repassada ao overlay.
    private var frameAspect           = 0.75f

    // Campos de estado do reconhecimento de letra que precisam ser zerados
    // juntos sempre que a mão some do quadro ou o modo troca — extraído para
    // não depender de lembrar a mesma lista de resets em cada lugar.
    private fun resetEstadoAlfabeto() {
        ultimaPredicao = ""
        contadorEstabilidade = 0
        letraRepetidaPendente = ""
        ultimaLetraAdicionada = ""
        letraEngine.resetMovimentoSustentado()
    }

    private fun nextVideoTimestamp(): Long {
        val now = SystemClock.uptimeMillis()
        videoTimestamp = if (now <= videoTimestamp) videoTimestamp + 1 else now
        return videoTimestamp
    }

    @ExperimentalGetImage
    override fun analyze(imageProxy: ImageProxy) {
      // rawBitmap/preparedBitmap precisam existir fora do try para o finally
      // poder reciclá-los (variável declarada dentro do try não é visível lá).
      var rawBitmap: Bitmap? = null
      var preparedBitmap: Bitmap? = null
      try {
        // ── Converter YUV → RGBA_8888 (exigido pelo MediaPipe) ────────────
        // Os dois são reciclados no finally: detectForVideo é síncrono (sem
        // callback), então quando este método termina nenhum dos dois é mais
        // usado — reciclar aqui libera a memória nativa do bitmap na hora, em
        // vez de esperar o coletor de lixo (a alocação de 1-2 bitmaps por
        // frame de câmera é uma fonte real de pressão de GC).
        val raw = imageProxy.toBitmap()
        rawBitmap = raw
        val prepared = prepararBitmap(
            raw, imageProxy.imageInfo.rotationDegrees.toFloat(), espelharImagem)
        preparedBitmap = prepared
        val mpImage = BitmapImageBuilder(prepared).build()
        // Corrige o x para a proporção 4:3 do treino (quadro retrato -> 4:3).
        frameAspect = prepared.width.toFloat() / prepared.height
        aspectX = 0.75f * frameAspect

        val timestamp = nextVideoTimestamp()
        val result = handLandmarker.detectForVideo(mpImage, timestamp)

        // Rosto roda nos DOIS modos, igual ao Holistic do Python ("Em AMBOS
        // os modos o ROSTO entra como marcador não-manual") — ver
        // FaceMarkerEngine para o throttle de frames e o porte do ler_marcador.
        faceEngine.step(mpImage, timestamp, onInterrogativo)

        if (modoAtual == Modo.CORPO) {
            val poseDetector = bodyEngine.ensureLoaded()
            if (poseDetector == null) {
                onLetra("-", 0f, "corpo")
                onFeedback("MODELO DE CORPO INDISPONIVEL", FEEDBACK_ALERTA)
                return
            }
            val poseResult = poseDetector.detectForVideo(mpImage, timestamp)
            analisarCorpo(result, poseResult)
            return
        }

        if (result.landmarks().isEmpty()) {
            onLandmarks(emptyList(), null, frameAspect)
            framesSemMao++
            if (framesSemMao >= NO_HAND_TOLERANCE) {
                resetEstadoAlfabeto()
                tempoInicioEsticado = 0L
                letraEngine.limparBuffer()
                // Libera a mesma letra para ser digitada de novo: tirar a mão
                // do quadro e refazer o sinal é a forma natural de repetir.
                onRepeticaoPendente(null)
                onFeedback("MAO FORA DO QUADRO", FEEDBACK_ALERTA)
                onNoHand()
            }
            return
        }

        framesSemMao = 0
        onLandmarks(handsToArrays(result), null, frameAspect)
        val lms    = result.landmarks()[0]
        // x corrigido para 4:3 (features + geometria); o desenho usa o cru.
        val pontos = lms.map { Pair(it.x() * aspectX, it.y()) }
        val dedicosEsticados = LibrasMath.detectarDedosEsticados(pontos)

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

            val predicao = letraEngine.process(pontos)
            val letra = predicao.letra
            val confianca = predicao.confianca
            val modo = predicao.modo

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
      } finally {
        if (rawBitmap !== preparedBitmap) rawBitmap?.recycle()
        preparedBitmap?.recycle()
        imageProxy.close()
      }
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
    private fun handsToArrays(result: HandLandmarkerResult): List<FloatArray> =
        result.landmarks().map { hand ->
            FloatArray(hand.size * 2).also { arr ->
                hand.forEachIndexed { i, lm -> arr[i * 2] = lm.x(); arr[i * 2 + 1] = lm.y() }
            }
        }

    private fun poseToArray(poseResult: PoseLandmarkerResult): FloatArray? {
        val poses = poseResult.landmarks()
        if (poses.isEmpty()) return null
        val p = poses[0]
        return FloatArray(p.size * 2).also { arr ->
            p.forEachIndexed { i, lm -> arr[i * 2] = lm.x(); arr[i * 2 + 1] = lm.y() }
        }
    }

    private fun podeRepetirAutomaticamente(letra: String): Boolean {
        if (letra !in LETRAS_REPETICAO_AUTO) return false
        return frase.length < 2 || frase[frase.length - 2].toString() != letra
    }

    private fun analisarCorpo(handResult: HandLandmarkerResult, poseResult: PoseLandmarkerResult) {
        onLandmarks(handsToArrays(handResult), poseToArray(poseResult), frameAspect)
        val bodyFrame = bodyEngine.extractFrame(handResult, poseResult, aspectX)
        val agora = System.currentTimeMillis()

        if (!bodyFrame.hasPose || !bodyFrame.hasHand) {
            bodyEngine.onCorpoAusente(agora)
            onLetra("-", 0f, "corpo")
            onFeedback("ENQUADRE CORPO E MAO", FEEDBACK_ALERTA)
            onNoHand()
            return
        }

        // Gesto de limpar: mão toda aberta e parada por TEMPO_PRA_LIMPAR limpa
        // a frase — o mesmo gesto do modo alfabeto (a barra vermelha mostra o
        // progresso). Responde à dúvida "manter a mão aberta reseta as palavras".
        val maoAberta = handResult.landmarks().firstOrNull()?.let { hand ->
            LibrasMath.detectarDedosEsticados(hand.map { Pair(it.x() * aspectX, it.y()) })
        } ?: false
        if (maoAberta) {
            if (tempoInicioEsticado == 0L) tempoInicioEsticado = agora
            val progresso = ((agora - tempoInicioEsticado).toFloat() / TEMPO_PRA_LIMPAR)
                .coerceIn(0f, 1f)
            onGestoLimpar(progresso)
            if ((agora - tempoInicioEsticado) >= TEMPO_PRA_LIMPAR &&
                (agora - ultimoTempoLimpar) > 2_000L) {
                frase = ""; tempoInicioEsticado = 0L; ultimoTempoLimpar = agora
                bodyEngine.limparTokens()
                onFraseUpdate("")
            }
            bodyEngine.resetCapture()
            onLetra("-", 0f, "corpo")
            return
        } else {
            tempoInicioEsticado = 0L
            onGestoLimpar(0f)
        }

        val novaFrase = bodyEngine.processarFrame(bodyFrame, agora, onLetra, onFeedback)
        if (novaFrase != null) {
            frase = novaFrase
            onFraseUpdate(frase)
        }
    }

    fun setModo(novoModo: Modo) {
        modoAtual = novoModo
        bodyEngine.resetTudo()
        resetEstadoAlfabeto()
        // Troca de modo começa uma frase nova (letras e sinais de corpo não se
        // misturam na mesma frase).
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

    // ── Calibração pessoal (delegada ao LetraEngine) ───────────────────────
    fun startCalibration(letra: String) = letraEngine.startCalibration(letra)
    fun finishCalibration(): Boolean = letraEngine.finishCalibration()
    fun cancelCalibration() = letraEngine.cancelCalibration()
    fun getCalibrationCount(): Int = letraEngine.getCalibrationCount()
    fun getCalibrationFrameCount(): Int = letraEngine.getCalibrationFrameCount()
    fun getTrainingSampleCount(letra: String? = null): Int = letraEngine.getTrainingSampleCount(letra)
    fun getTrainingDatasetPath(): String = letraEngine.getTrainingDatasetPath()
    fun getDynamicTrainingDatasetPath(): String = letraEngine.getDynamicTrainingDatasetPath()
    fun clearTrainingData() = letraEngine.clearTrainingData()

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
            val novaFrase = bodyEngine.apagarUltimoToken() ?: return
            frase = novaFrase
            onFraseUpdate(frase)
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
        bodyEngine.limparTokens()
        onRepeticaoPendente(null); onFraseUpdate(frase)
    }
    fun getFrase(): String = frase

    fun close() {
        handLandmarker.close()
        letraEngine.close()
        bodyEngine.close()
        faceEngine.close()
    }
}
