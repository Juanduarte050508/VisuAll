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
import com.google.mediapipe.tasks.core.Delegate
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
        // A análise já chega em 640x480 (ver o ResolutionSelector do
        // LibrasFragment), então 480 aqui significa NÃO reduzir mais: o
        // detector recebe o frame como a câmera entregou.
        //
        // Passou por 360 (valor do Python), 255 e 300, todos escolhidos por
        // velocidade e nenhum medido contra acurácia. O risco que estava
        // escrito aqui desde então -- "mão pequena/longe do quadro perde
        // detalhe do landmark" -- foi o que apareceu no aparelho: no modo
        // corpo a pessoa recua pra caber o tronco, a mão passa a ocupar uma
        // fração pequena do quadro, e em AJUDAR (as duas mãos encostadas,
        // uma ocluindo a outra) os pontos da mão saíam errados. Landmark
        // ruim é irrecuperável: nenhum limiar ou modelo conserta depois.
        //
        // O custo é real -- 640x480 tem 2.5x a área de 400x300 -- e cai em
        // cima do MediaPipe, que é o gargalo. Se a latência incomodar no
        // modo alfabeto (onde a mão está perto e detalhe sobra), o caminho é
        // reduzir só nesse modo, não voltar a reduzir no corpo.
        const val INPUT_SHORT_SIDE       = 480
        // O MLP é "superconfiante": cospe ~0.99 quase sempre, então só a
        // confiança filtra muito pouco. A MARGEM (1ª menos 2ª opção) é o
        // critério que realmente separa um sinal claro de um chute.
        //
        // Subidos de 0.90/0.25 depois de relato de reconhecimento fácil
        // demais (letra confirmada mesmo sem estar sendo feita). Ainda
        // pendente de validação real — se ficar difícil demais de acertar
        // letras de verdade, é o primeiro lugar pra abaixar de novo.
        const val CONFIANCA_MINIMA       = 0.93f
        const val MARGEM_ESTATICA_MINIMA = 0.30f
        // O modelo dinâmico só conhece 5 classes (H,J,K,X,Z) e não tem classe
        // "nenhuma". Subidos de 0.92/0.28 pelo mesmo motivo do
        // CONFIANCA_MINIMA acima (ver LIMIAR_MOVIMENTO abaixo para a mudança
        // que ataca o falso-J por outro ângulo, o de tempo/histerese).
        const val CONFIANCA_DINAMICA     = 0.95f
        const val MARGEM_DINAMICA_MINIMA = 0.32f
        // Usado só pelos modelos INDIVIDUAIS (um classificador binário "é
        // esta letra ou não" por letra, treinado pela ferramenta em
        // treino/). Diferente do modelo geral (multiclasse, softmax
        // sobre TODAS as letras reais), um binário nunca viu "mão se
        // mexendo sem sinalizar nada" como exemplo negativo — só viu outras
        // letras reais. Isso o deixa mais propenso a "confiante" demais
        // (overconfident) em movimento que não é sinal nenhum, porque nunca
        // aprendeu a rejeitar o que não é nenhuma das classes que conhece.
        // Por isso a barra aqui é mais alta que a do modelo geral.
        const val CONFIANCA_INDIVIDUAL   = 0.97f
        // Usado quando existe UM ÚNICO modelo individual treinado. Nesse caso
        // não há segundo colocado, então a margem vira igual à confiança e o
        // portão de margem não filtra nada — qualquer resposta acima de
        // CONFIANCA_INDIVIDUAL passa. E esse é o primeiro cenário que vai
        // acontecer, não um caso de borda: treinar uma letra só pra medir se
        // gravar resolve gera exatamente um modelo. Como não há com quem
        // comparar, a exigência sobe. Precisa validar em celular real: se a
        // letra recém-treinada não aparecer nunca, é o primeiro valor a baixar.
        const val CONFIANCA_INDIVIDUAL_SEM_RIVAL = 0.99f
        // Eu e o Rafael tínhamos valores incompatíveis aqui: 0.30 (dele, igual
        // ao Python) deixava o J disparar com qualquer tremida; 0.55 (meu)
        // travava gestos reais de H/J/K/X/Z no modelo estático. Os dois
        // usavam a MESMA variável pra duas coisas diferentes: magnitude do
        // movimento E se ele é intencional. Separamos isso: LIMIAR_MOVIMENTO
        // volta a 0.30 (não perde gesto real, como o Rafael queria), mas só
        // é CONFIADO depois de sustentado por um tempo mínimo
        // (MOVIMENTO_SUSTENTADO_MS) — uma tremida de 1 frame não basta, um
        // traço real de H/J/K/X/Z (que dura ~300-500ms) sim.
        const val LIMIAR_MOVIMENTO       = 0.30f
        // Em MILISSEGUNDOS, não em frames (ver ESTAB_MIN_* abaixo pro mesmo
        // motivo): num aparelho que analisa poucos frames por segundo, uma
        // contagem em frames vira uma janela de tempo bem maior do que
        // pretendido, e o app perde gestos rápidos. Tempo fixo se comporta
        // igual não importa a taxa de quadros real do dispositivo. Precisa
        // validar em celular real; se ainda sair J fácil, subir este valor
        // antes de mexer em LIMIAR_MOVIMENTO de novo.
        const val MOVIMENTO_SUSTENTADO_MS = 130L
        // Por quanto tempo, DEPOIS que o movimento cai, a janela do gesto que
        // acabou continua sendo classificada como dinâmica (estado ENCERRANDO
        // do MovementGate). Precisa ser maior que ESTAB_MIN_DINAMICO_MS: é
        // dentro dessa janela que a letra tem que se manter estável pra entrar
        // na frase. Com o portão fechando no mesmo quadro em que o movimento
        // parava, o gesto terminava antes de a letra ser aceita e a letra se
        // perdia — o sintoma relatado em teste era o app "continuar analisando"
        // depois do fim do movimento e não capturar a letra.
        const val MOVIMENTO_ENCERRAMENTO_MS = 400L
        const val TEMPO_PRA_LIMPAR       = 3_000L
        // No modo corpo a referencia exige mais tempo (TEMPO_LIMPAR_CORPO =
        // 5.0 em m01_visuall_config.py): um sinal de corpo dura muito mais que
        // uma letra, entao 3s de palma aberta acontecem sem querer.
        const val TEMPO_PRA_LIMPAR_CORPO = 5_000L
        // Espera antes de a mao aberta poder limpar de novo. Sem ela, manter a
        // palma aberta depois de limpar dispararia a cada quadro seguinte.
        // Estava escrita como 2_000L solto em dois lugares diferentes.
        const val ESPERA_ENTRE_LIMPEZAS_MS = 2_000L
        // Tempo mínimo com a MESMA letra reconhecida antes de comitar na
        // frase — em milissegundos, não em frames, pelo mesmo motivo do
        // MOVIMENTO_SUSTENTADO_MS acima. Dinâmicas são transitórias; exigir
        // tempo demais faz a janela passar do gesto antes da letra ser
        // adicionada.
        const val ESTAB_MIN_DINAMICO_MS  = 260L
        const val ESTAB_MIN_ESTATICO_MS  = 850L
        const val COOLDOWN_DINAMICO      = 700L
        const val COOLDOWN_ESTATICO      = 1_100L
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
        // Baixado de 0.030 depois de MEDIR o movimento real de cada sinal nos
        // 200 clipes gravados (treino/diagnostico/mede_movimento.py). O valor
        // Abaixo daqui por BODY_END_FRAMES quadros seguidos, a captura
        // encerra e o trecho vai pro modelo. 0.030 e o LIMIAR_FIM do pipeline
        // Python de referencia (m01_visuall_config.py).
        //
        // Ja tentei 0.015 aqui, com o argumento de que o movimento mediano do
        // AJUDAR e 0.0267 e portanto ele seria cortado no meio. Foi pior, e a
        // medicao (treino/diagnostico/mede_parada.py) mostra por que: baixar o
        // limiar nao faz a captura durar ate o fim do sinal, faz ela NUNCA
        // encerrar sozinha e bater no teto de BODY_MAX_FRAMES. Corte limpo por
        // parada, sobre os 200 clipes: 76% em 0.030 contra 65% em 0.015, e o
        // COMPUTADOR caindo de 19/36 pra 13/36. No aparelho isso apareceu como
        // "a palavra so e capturada quando eu comeco o proximo sinal" -- a
        // janela entregue ao modelo pegava o fim de um sinal colado no comeco
        // do outro, e a classificacao desabava pra CONVERSAR.
        //
        // Medianas medidas: AJUDAR 0.0267 | COMPUTADOR 0.0599 | CONVERSAR
        // 0.0448 | PESSOA 0.0133 | SURDO 0.0106 | NEUTRO 0.0083.
        //
        // Este valor tem um gemeo em treino/treinar_corpo.py (END_MOTION). Os
        // dois PRECISAM andar juntos: e o recorte que o modelo aprende a
        // classificar. Mudar um sem o outro foi exatamente o erro acima.
        const val BODY_END_MOTION        = 0.030f
        // Quanto a mao pode se AFASTAR do ponto onde parou e ainda contar
        // como "parada no lugar", pra limpar a frase. Fracao do quadro (0..1).
        //
        // Rede de seguranca, NAO a trava principal. Quem impede um sinal de
        // encher a barra e gestoEmAndamento (a captura esta gravando). Isto
        // aqui so cobre o intervalo entre o movimento comecar e a captura
        // engatar, e um sinal que nunca passe de BODY_START_MOTION.
        //
        // Ja foi 0.06 e era apertado demais: com 5s de mao levantada, a deriva
        // natural passava disso, a contagem reiniciava e limpar ficou
        // impossivel -- relatado como "a barra comeca a carregar e para".
        const val LIMPAR_DESLOCAMENTO_MAXIMO = 0.15f
        // Por quantos quadros a ultima posicao conhecida de uma mao e mantida
        // quando o MediaPipe a perde. ~0.4s na taxa medida no aparelho (13
        // quadros/s). Ver preencheMaoPerdida no BodyGestureEngine.
        const val MAX_QUADROS_MAO_PERDIDA = 5
        const val BODY_START_FRAMES      = 3
        const val BODY_END_FRAMES        = 5
        const val BODY_MIN_FRAMES        = 10
        const val BODY_MAX_FRAMES        = 60
        // De volta a 0.85, o valor do pipeline Python citado logo acima.
        // Tinha sido subido pra 0.90 junto com os limiares de mão, mas o modo
        // corpo não tem portão de margem (ver isReliable em BodyGestureEngine):
        // aqui a confiança é o único corte, então 0.05 a mais não filtra
        // dúvida, apaga os sinais mais fracos inteiros. Relatado no aparelho:
        // AJUDAR, PESSOA e CONVERSAR pararam de sair, enquanto SURDO e
        // COMPUTADOR continuaram. Se voltar a reconhecer fácil demais, o
        // caminho é exigir margem como as letras fazem, não subir isto de novo.
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
        // Subiu de 3 para 5: a taxa de quadros efetiva de análise (mão +
        // ONNX + às vezes rosto) é o gargalo pra pegar letras/gestos com
        // movimento rápido — cada frame que o FaceLandmarker NÃO roda é um
        // frame a mais de orçamento pro que realmente importa (mão +
        // classificação). A sobrancelha ainda muda de estado bem mais devagar
        // que isso.
        const val FACE_DETECT_STRIDE = 5
    }

    enum class Modo {
        ALFABETO,
        CORPO
    }

    private val handLandmarker: HandLandmarker
    // onLetra/onFeedback passam por notificarLetra/notificarFeedback (abaixo)
    // em vez de serem chamados direto — os dois seriam disparados a CADA
    // frame analisado (mão parada no mesmo sinal, corpo ocioso esperando
    // gesto, etc.), e cada chamada posta uma Runnable pra thread de UI que
    // reescreve texto/cor. Numa taxa de análise mais alta isso inunda a
    // thread principal com trabalho redundante — a causa mais provável de
    // travamento ao subir os fps, já que nada na tela de fato mudou entre
    // um frame e outro na maioria das vezes.
    private val letraEngine = LetraEngine(context, ::notificarFeedback)
    private val bodyEngine = BodyGestureEngine(context)
    private val faceEngine = FaceMarkerEngine(context)

    private var ultimaLetraNotificada = ""
    private var ultimaPorcentagemNotificada = -1
    private var ultimaFeedbackMensagem: String? = null
    private var ultimaFeedbackNivel = -1

    // Só repassa pro callback do Fragment quando o valor EXIBIDO muda de
    // verdade. Compara por porcentagem arredondada (o que a UI mostra), não
    // pela confiança crua — o float varia um pouco a cada frame só por
    // jitter do landmark, então comparar o float não deduplicaria quase nada.
    private fun notificarLetra(letra: String, confianca: Float, modo: String) {
        val porcentagem = if (letra != "-") (confianca * 100).toInt().coerceIn(0, 100) else -1
        if (letra == ultimaLetraNotificada && porcentagem == ultimaPorcentagemNotificada) return
        ultimaLetraNotificada = letra
        ultimaPorcentagemNotificada = porcentagem
        onLetra(letra, confianca, modo)
    }

    private fun notificarFeedback(mensagem: String, nivel: Int) {
        if (mensagem == ultimaFeedbackMensagem && nivel == ultimaFeedbackNivel) return
        ultimaFeedbackMensagem = mensagem
        ultimaFeedbackNivel = nivel
        onFeedback(mensagem, nivel)
    }

    @Volatile private var modoAtual = Modo.ALFABETO
    // Espelha a imagem na horizontal quando estamos na câmera frontal. O
    // dataset foi gravado com a webcam espelhada (cv2.flip no Python), então
    // a câmera frontal precisa do mesmo espelhamento para o "lado" da mão
    // bater com o treino. Câmera traseira não é espelhada por natureza.
    @Volatile private var espelharImagem = true

    private fun handOptions(delegate: Delegate) = HandLandmarkerOptions.builder()
        .setBaseOptions(BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .setDelegate(delegate)
            .build())
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

    init {
        // GPU costuma ser bem mais rápido que CPU pra esses modelos de
        // landmark, mas nem todo aparelho/driver aceita o delegate (falha na
        // hora de montar o grafo, não durante a inferência — é assim que o
        // MediaPipe documenta essa falha, então pegar aqui no createFromOptions
        // é suficiente). Tenta GPU primeiro, cai pra CPU se der erro. HandLandmarker
        // é obrigatório (sem ele não tem reconhecimento nenhum), então mesmo
        // a tentativa em CPU pode falhar — mesmo comportamento de antes desta
        // mudança nesse caso extremo.
        handLandmarker = try {
            HandLandmarker.createFromOptions(context, handOptions(Delegate.GPU))
        } catch (e: Throwable) {
            android.util.Log.w("LibrasAnalyzer", "GPU indisponivel pro HandLandmarker, usando CPU", e)
            HandLandmarker.createFromOptions(context, handOptions(Delegate.CPU))
        }
    }

    // Os portões de estabilidade/cooldown que decidem se uma letra entra na
    // frase — ver LetterCommitGate.
    private val commitGate            = LetterCommitGate()
    private val clearGate             = ClearGestureGate()
    private var tempoInicioEsticado   = 0L
    private var ultimoTempoLimpar     = 0L
    // A frase e as regras de como ela muda (repetição, sugestão, apagar) moram
    // no SentenceBuilder, que é testável em JVM.
    private val sentence              = SentenceBuilder()
    private var framesSemMao          = 0
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
        commitGate.reset()
        sentence.limparPendente()
        letraEngine.resetMovimentoSustentado()
        // Sem isso, tirar a mão do quadro e voltar a fazer O MESMO sinal (ex.:
        // "A" a 87%) seria filtrado pelo dedup de notificarLetra por parecer
        // idêntico ao último valor notificado antes da mão sumir — mesmo o
        // chip tendo sido escondido nesse meio-tempo (onNoHand). Resetar o
        // cache aqui garante que a primeira detecção após a mão voltar sempre
        // notifica de novo.
        ultimaLetraNotificada = ""
        ultimaPorcentagemNotificada = -1
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
                notificarLetra("-", 0f, "corpo")
                // Inclui o motivo em vez de só "indisponível": depois de um
                // retreino, saber se o arquivo sumiu ou se saiu no formato
                // errado é a diferença entre corrigir em um minuto e ficar
                // procurando problema na frente da câmera.
                val motivo = bodyEngine.motivoFalha
                notificarFeedback(
                    if (motivo.isNullOrBlank()) "MODELO DE CORPO INDISPONIVEL"
                    else "MODELO DE CORPO INDISPONIVEL: $motivo",
                    FEEDBACK_ALERTA
                )
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
                notificarFeedback("MAO FORA DO QUADRO", FEEDBACK_ALERTA)
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
                sentence.limpar(); commitGate.liberarRepeticao()
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

            notificarLetra(letra, confianca, modo)

            val agora = System.currentTimeMillis()
            if (commitGate.avaliar(letra, modo, agora)) {
                when (sentence.aceitarLetra(letra)) {
                    SentenceBuilder.Resultado.ADICIONADA -> {
                        onRepeticaoPendente(null)
                        onFraseUpdate(sentence.frase)
                    }
                    SentenceBuilder.Resultado.AGUARDANDO_CONFIRMACAO ->
                        onRepeticaoPendente(letra)
                }
                commitGate.registrarComite(letra, agora)
            }

            // NÃO limpamos ultimaLetraAdicionada por tempo: fazer isso digitava
            // a mesma letra repetidamente enquanto a mão ficava parada no sinal.
            // Ela só é liberada quando a mão sai do quadro (ver bloco sem mão)
            // ou pelo botão REPETIR, deixando a repetição sempre intencional.
        }
      } catch (error: Throwable) {
        android.util.Log.e("LibrasAnalyzer", "Falha ao analisar frame de Libras", error)
        notificarLetra("-", 0f, modoAtual.name.lowercase())
        notificarFeedback("ERRO NO RECONHECIMENTO", FEEDBACK_ALERTA)
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
        // filter=true (bilinear). Era nearest-neighbor, com a justificativa
        // de que a imagem só serve de entrada pro detector e não é exibida --
        // mas é justamente por servir ao detector que ela precisa do filtro:
        // vizinho-mais-próximo serrilha bordas finas, e borda fina é o que
        // um dedo é. Com INPUT_SHORT_SIDE em 480 a rotação e o espelho
        // passam a ser as únicas transforms, então o custo do bilinear aqui
        // é pequeno e a qualidade da borda é o que o MediaPipe consome.
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

    // A mao que conta pro gesto de limpar: a primeira ABERTA no quadro.
    //
    // Ja tentei "a primeira da lista" (a ordem do MediaPipe muda entre quadros)
    // e depois "direita, senao esquerda", copiando o
    // right_hand_landmarks or left_hand_landmarks da referencia. As duas estao
    // erradas aqui, e a segunda foi pior: a referencia usa o Holistic, onde so
    // existe UMA mao de cada lado, enquanto aqui as duas maos aparecem em 395
    // de 463 quadros medidos no aparelho -- entao "direita" escolhia a mao
    // parada ao lado do corpo e a mao aberta erguida nunca era testada.
    // Resultado: aberta=true em 0 de 463 quadros, com o gesto sendo feito.
    //
    // Perguntar "alguma mao esta aberta?" e o que o gesto quer dizer, e nao
    // depende de rotulo de lado -- que numa camera frontal espelhada nem
    // corresponde a mao anatomica.
    private fun maoAbertaNoQuadro(handResult: HandLandmarkerResult) =
        handResult.landmarks().firstOrNull { hand ->
            LibrasMath.detectarDedosEsticados(hand.map { Pair(it.x() * aspectX, it.y()) })
        }

    private fun analisarCorpo(handResult: HandLandmarkerResult, poseResult: PoseLandmarkerResult) {
        onLandmarks(handsToArrays(handResult), poseToArray(poseResult), frameAspect)
        val bodyFrame = bodyEngine.extractFrame(handResult, poseResult, aspectX)
        val agora = System.currentTimeMillis()

        if (!bodyFrame.hasPose || !bodyFrame.hasHand) {
            bodyEngine.onCorpoAusente(agora)
            notificarLetra("-", 0f, "corpo")
            notificarFeedback("ENQUADRE CORPO E MAO", FEEDBACK_ALERTA)
            onNoHand()
            return
        }

        // Movimento PRIMEIRO: o gesto de limpar precisa saber se a mão está
        // parada, e essa informação vem da mesma janela que a captura usa.
        bodyEngine.registrarMovimento(bodyFrame)

        // Gesto de limpar: mão toda aberta E PARADA por TEMPO_PRA_LIMPAR limpa
        // a frase (a barra vermelha mostra o progresso).
        //
        // A exigência de estar parada é o que conserta o AJUDAR. Antes só a
        // abertura da mão era checada, apesar de o comentário aqui já dizer
        // "e parada" — então um sinal feito de palma aberta caía no contador de
        // limpar, o resetCapture() abaixo zerava a captura, e o gesto NUNCA
        // chegava ao modelo. Relatado no aparelho exatamente assim: "AJUDAR não
        // funciona, ele fica contando os segundos pra limpar o texto".
        val mao = maoAbertaNoQuadro(handResult)
        val maoAberta = mao != null
        // Pulso (ponto 0): é o que menos se mexe quando só os dedos mudam de
        // forma, então serve de referência estável de "a mão está neste lugar".
        val pulso = mao?.firstOrNull()
        // gestoEmAndamento e a trava principal: no AJUDAR a mao aberta e a de
        // APOIO e mal sai do lugar, entao o deslocamento sozinho nao segurava a
        // barra. Se a captura esta gravando, a pessoa esta sinalizando.
        val limpeza = clearGate.avaliar(
            maoAberta, pulso?.x() ?: 0f, pulso?.y() ?: 0f, agora,
            gestoEmAndamento = bodyEngine.gestoEmAndamento
        )
        onGestoLimpar(limpeza.progresso)
        if (limpeza.limpar) {
            sentence.limpar()
            bodyEngine.limparTokens()
            onFraseUpdate("")
        }
        // NÃO interrompe a captura só porque a barra começou a encher. Foi o
        // erro da primeira tentativa de consertar o AJUDAR: medindo os clipes,
        // 70% dos quadros de AJUDAR ficam abaixo do limiar de "parado", então
        // a barra engatilhava quase sempre e o return abortava a captura --
        // pior que o bug original.
        //
        // Os dois não precisam se excluir: pra limpar é preciso a palma aberta
        // e imóvel por 3s seguidos, e nessa condição o movimento fica abaixo
        // de BODY_START_MOTION, então a captura não começa de qualquer jeito.
        // Deixar os dois correrem em paralelo é o que permite um sinal de palma
        // aberta ser classificado.
        if (limpeza.limpar) {
            bodyEngine.resetCapture()
            notificarLetra("-", 0f, "corpo")
            return
        }

        val novaFrase = bodyEngine.processarFrame(bodyFrame, agora, ::notificarLetra, ::notificarFeedback)
        if (novaFrase != null) {
            sentence.definir(novaFrase)
            onFraseUpdate(sentence.frase)
        }
    }

    fun setModo(novoModo: Modo) {
        modoAtual = novoModo
        bodyEngine.resetTudo()
        resetEstadoAlfabeto()
        // Troca de modo começa uma frase nova (letras e sinais de corpo não se
        // misturam na mesma frase).
        sentence.limpar()
        onFraseUpdate("")
        onRepeticaoPendente(null)
        // resetEstadoAlfabeto() (acima) já zera o cache de dedup da letra;
        // falta só o do feedback, pra troca de modo reafirmar a mensagem na
        // UI mesmo que coincida por acaso com a última já notificada.
        ultimaFeedbackMensagem = null
        ultimaFeedbackNivel = -1
        notificarLetra("-", 0f, novoModo.name.lowercase())
        if (novoModo == Modo.CORPO) {
            notificarFeedback("MODO CORPO: ENQUADRE TRONCO E MAO", FEEDBACK_NEUTRO)
        } else {
            notificarFeedback("MODO LIBRAS: CENTRALIZE A MAO", FEEDBACK_NEUTRO)
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

    // Vêm dos labels.txt que acompanham os modelos — ver comentário em
    // LetraEngine. A UI de calibração usa estes em vez de uma lista própria,
    // pra não haver duas versões do alfabeto podendo divergir.
    fun labelsAlfabeto(): List<String> = letraEngine.labelsAlfabeto
    fun labelsDinamicas(): Set<String> = letraEngine.labelsDinamicasSet

    fun aplicarSugestao(palavra: String) {
        if (!sentence.aplicarSugestao(palavra)) return
        commitGate.reset()
        onRepeticaoPendente(null)
        onFraseUpdate(sentence.frase)
    }

    fun adicionarEspaco() {
        sentence.adicionarEspaco()
        onRepeticaoPendente(null)
        onFraseUpdate(sentence.frase)
    }

    fun repetirLetraPendente() {
        if (!sentence.confirmarRepeticao()) return
        commitGate.liberarRepeticao()
        onRepeticaoPendente(null)
        onFraseUpdate(sentence.frase)
    }
    fun apagarUltima() {
        if (modoAtual == Modo.CORPO) {
            // No corpo apagamos o último SINAL (token) e re-traduzimos.
            val novaFrase = bodyEngine.apagarUltimoToken() ?: return
            sentence.definir(novaFrase)
            onFraseUpdate(sentence.frase)
            return
        }
        if (!sentence.apagarUltima()) return
        commitGate.liberarRepeticao()
        onRepeticaoPendente(null)
        onFraseUpdate(sentence.frase)
    }

    fun limparFrase() {
        sentence.limpar()
        commitGate.liberarRepeticao()
        bodyEngine.limparTokens()
        onRepeticaoPendente(null)
        onFraseUpdate(sentence.frase)
    }

    fun getFrase(): String = sentence.frase

    fun close() {
        runCatching { handLandmarker.close() }
            .onFailure { android.util.Log.w("LibrasAnalyzer", "Falha ao fechar HandLandmarker", it) }
        runCatching { letraEngine.close() }
            .onFailure { android.util.Log.w("LibrasAnalyzer", "Falha ao fechar LetraEngine", it) }
        runCatching { bodyEngine.close() }
            .onFailure { android.util.Log.w("LibrasAnalyzer", "Falha ao fechar BodyGestureEngine", it) }
        runCatching { faceEngine.close() }
            .onFailure { android.util.Log.w("LibrasAnalyzer", "Falha ao fechar FaceMarkerEngine", it) }
    }
}
