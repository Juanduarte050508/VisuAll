package com.visuall.app.libras

import android.Manifest
import android.animation.ValueAnimator
import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.content.res.ColorStateList
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.RecognizerIntent
import android.speech.tts.TextToSpeech
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.util.Log
import android.util.Range
import android.view.MotionEvent
import android.view.WindowManager
import android.view.Surface
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts

import android.util.Size
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewCompositionStrategy
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.visuall.app.R
import com.visuall.app.databinding.FragmentLibrasBinding
import com.visuall.app.oculos.EnderecoDosOculos
import com.visuall.app.oculos.MensagemDeErro
import com.visuall.app.oculos.MjpegClient
import com.visuall.app.oculos.NetworkStreamSource
import com.visuall.app.ui.EncaixeDeQuadro
import com.visuall.app.ui.ScanFrameView
import com.visuall.app.ui.compose.LibrasLandscapeHud
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.roundToInt

class LibrasFragment : Fragment(), TextToSpeech.OnInitListener {

    private companion object {
        // Cores fixas do chip de confiança: só 3 buckets possíveis, então não
        // há motivo pra alocar um ColorStateList novo (via valueOf) a cada
        // atualização — onLetraDetectada roda a cada letra reconhecida.
        val TINT_CONFIANCA_ALTA = ColorStateList.valueOf(0xFFF5C842.toInt())
        val TINT_CONFIANCA_MEDIA = ColorStateList.valueOf(0xFFE8A020.toInt())
        val TINT_CONFIANCA_BAIXA = ColorStateList.valueOf(0xFF8E6A26.toInt())
        val TINT_CONFIANCA_FUNDO = ColorStateList.valueOf(0x33242424)

        // Quanto tempo o dedo precisa ficar na lixeira pra limpar a frase
        // inteira. Mais curto que o gesto de mao aberta
        // (LibrasAnalyzer.TEMPO_PRA_LIMPAR_CORPO, 5s): ali o tempo longo existe
        // pra nao confundir com um sinal sendo feito, e aqui nao ha essa
        // duvida -- o dedo esta no botao de proposito.
        const val TEMPO_HOLD_LIMPAR_MS = 3_000L
    }

    private var _binding: FragmentLibrasBinding? = null
    private val binding get() = _binding!!

    private lateinit var cameraExecutor: ExecutorService
    private var librasAnalyzer: LibrasAnalyzer? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var tts: TextToSpeech? = null
    private var landscapeHud: View? = null
    private var bindRetryPosted = false
    private var cameraStartRequested = false
    private var bindInProgress = false
    private var pendingBind = false

    private var lensFacing = CameraSelector.LENS_FACING_FRONT

    // Camera dos oculos (ESP32 por Wi-Fi). Quando ligada, o CameraX e
    // desvinculado e os quadros passam a vir da rede -- ver ligarOculos().
    private var fonteOculos: NetworkStreamSource? = null
    private var usandoOculos = false

    /** Ultima falha ja avisada, pra o laco de reconexao nao repetir o aviso. */
    private var ultimoAvisoOculos: String? = null

    /** Tela escurecida e surda a toques, com o reconhecimento rodando. */
    private var telaBloqueada = false
    private var yInicialDestrave = 0f

    // Com o celular no bolso o gesto de voltar dispara sozinho na borda da
    // tela. Deixar passar tiraria a pessoa do modo oculos no meio de uma
    // conversa -- e ela so ia descobrir ao tirar o celular do bolso.
    private val ignorarVoltarBloqueado = object : OnBackPressedCallback(false) {
        override fun handleOnBackPressed() = Unit
    }

    private val historyStore by lazy { ConversationHistoryStore(requireContext().applicationContext) }

    // Controle de histórico: rastreia a frase completa anterior
    private var fraseAnterior = ""
    // Texto bruto vindo do analyzer (letras/tokens, sem "?"). O "?" do
    // marcador de sobrancelha é só de EXIBIÇÃO — igual ao Python, que nunca
    // grava a interrogação no texto, só no que é mostrado na tela.
    private var fraseBase = ""
    private var interrogativoAtivo = false
    private var ultimaLetraChip = ""
    private var linhasAtivas = true
    private var modoAtual = LibrasAnalyzer.Modo.ALFABETO

    // Hold da lixeira: animador da barra de progresso e a trava que impede o
    // ACTION_UP de apagar uma letra depois de a limpeza ja ter disparado (ou
    // depois de o dedo sair de cima do botao).
    private var holdAnimator: ValueAnimator? = null
    private var toqueLixeiraConsumido = false

    // O painel de resposta esta aberto? Guardado a parte porque o relayout do
    // HUD mexe na visibilidade das views e precisa saber ao que voltar.
    private var painelRespostaAberto = false

    // Texto da resposta atual. Vivia no tv_reply da bolha; com a bolha fora da
    // tela ele passa a ser estado do fragmento, sem view espelhando.
    private var respostaAtual = ""


    // Acabamos de abrir uma activity nossa (o reconhecedor de fala)? Se sim, o
    // onResume seguinte nao deve religar a camera. Ver o comentario no onResume.
    private var voltandoDeActivityNossa = false

    // Com a resposta ocupando a tela inteira, o "voltar" do sistema precisa
    // fechar o PAINEL antes de qualquer outra coisa. Sem isto ele sai do modo
    // Libras inteiro: quem esta respondendo aperta voltar esperando sair da
    // resposta e perde a camera junto, tendo que reabrir tudo.
    private val fecharRespostaNoVoltar = object : OnBackPressedCallback(false) {
        override fun handleOnBackPressed() {
            saveReplyToScreen()
            closeReplyPanel()
        }
    }
    private val speechLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode != Activity.RESULT_OK || _binding == null) {
            return@registerForActivityResult
        }

        val texto = result.data
            ?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            ?.firstOrNull()
            ?.trim()
            .orEmpty()

        if (texto.isNotBlank()) {
            openReplyPanel(focus = false)
            binding.etReply.setText(texto)
            binding.etReply.setSelection(texto.length)
            // Sem falar de volta: a resposta existe pra ser LIDA por quem nao
            // ouve. Repetir em voz alta o que a pessoa acabou de dizer nao
            // entrega a mensagem a ninguem.
            saveReplyToScreen()
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentLibrasBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        tts = TextToSpeech(requireContext(), this)
        historyStore.load()
        view.post {
            applyPreviewAspectRatio()
            applyHudLayout()
        }
        requireActivity().onBackPressedDispatcher
            .addCallback(viewLifecycleOwner, fecharRespostaNoVoltar)
        // Depois do de cima: o ultimo registrado tem prioridade, e enquanto a
        // tela esta travada nada mais deve responder ao voltar.
        requireActivity().onBackPressedDispatcher
            .addCallback(viewLifecycleOwner, ignorarVoltarBloqueado)
        setupButtons()
        updateModeButtons()
    }

    // ── Inicia provider uma única vez ──────────────────────────────────────
    private fun hasCameraPermission(): Boolean {
        val context = context ?: return false
        return ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
    }

    private fun startCamera() {
        val context = context ?: return
        // Ver a explicação longa no CameraFragment.startCamera: sem permissão
        // o provider falha e o future.get() abaixo estoura na main thread,
        // derrubando o app. O onResume tenta de novo depois.
        if (!hasCameraPermission()) return
        if (cameraStartRequested) return
        cameraStartRequested = true
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            // Zerar antes de qualquer return, senão a flag trava novas
            // tentativas pra sempre.
            cameraStartRequested = false
            if (!isAdded || _binding == null) return@addListener
            cameraProvider = try {
                future.get()
            } catch (e: Exception) {
                Log.e("LibrasFragment", "CameraX nao inicializou no modo Libras", e)
                Toast.makeText(context, "Nao consegui abrir a camera de Libras", Toast.LENGTH_SHORT).show()
                return@addListener
            }
            scheduleBindCamera()
        }, ContextCompat.getMainExecutor(context))
    }

    private fun scheduleBindCamera(delayMs: Long = 0L) {
        val root = _binding?.root ?: return
        root.removeCallbacks(bindRunnable)
        if (delayMs > 0L) {
            root.postDelayed(bindRunnable, delayMs)
        } else {
            root.post(bindRunnable)
        }
    }

    // Dispara quando o dedo completa TEMPO_HOLD_LIMPAR_MS sobre a lixeira.
    private val limparTudoRunnable = Runnable {
        toqueLixeiraConsumido = true
        librasAnalyzer?.limparFrase()
        vibrateConfirmation()
        pararProgressoHold()
    }

    private val bindRunnable = Runnable {
        if (_binding == null || !isAdded) return@Runnable
        bindCamera()
    }
    private fun cameraSelectorForAvailableLens(preferredLensFacing: Int): CameraSelector {
        val provider = cameraProvider
        val preferred = CameraSelector.Builder()
            .requireLensFacing(preferredLensFacing)
            .build()
        if (provider == null || runCatching { provider.hasCamera(preferred) }.getOrDefault(false)) {
            return preferred
        }

        val fallbackLensFacing = if (preferredLensFacing == CameraSelector.LENS_FACING_FRONT) {
            CameraSelector.LENS_FACING_BACK
        } else {
            CameraSelector.LENS_FACING_FRONT
        }
        val fallback = CameraSelector.Builder()
            .requireLensFacing(fallbackLensFacing)
            .build()
        return if (runCatching { provider.hasCamera(fallback) }.getOrDefault(false)) {
            lensFacing = fallbackLensFacing
            fallback
        } else {
            preferred
        }
    }

    private fun bindCamera() {
        val provider = cameraProvider ?: return
        if (!isAdded || _binding == null) return
        if (!hasCameraPermission()) return
        if (bindInProgress) {
            pendingBind = true
            return
        }
        if (!binding.previewView.isAttachedToWindow ||
            binding.previewView.width == 0 ||
            binding.previewView.height == 0
        ) {
            if (!bindRetryPosted) {
                bindRetryPosted = true
                binding.previewView.post {
                    bindRetryPosted = false
                    bindCamera()
                }
            }
            return
        }
        bindInProgress = true
        pendingBind = false

        val requestedLensFacing = lensFacing
        val selector = cameraSelectorForAvailableLens(requestedLensFacing)
        val targetRotation = currentTargetRotation()

        val resolutionSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(640, 480),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                )
            )
            .build()

        val preview = Preview.Builder()
            .setResolutionSelector(resolutionSelector)
            .setTargetRotation(targetRotation)
            .build()
            .also { p -> p.setSurfaceProvider(binding.previewView.surfaceProvider) }

        // Para o analyzer atual — seta null ANTES de unbindAll para evitar
        // que o executor entregue frames enquanto o provider já foi desvinculado
        val oldAnalyzer = librasAnalyzer
        librasAnalyzer = null

        // Recria executor se foi encerrado
        if (cameraExecutor.isShutdown) {
            cameraExecutor = Executors.newSingleThreadExecutor()
        }

        // Pede uma análise de baixa resolução (~640x480, 4:3) para a inferência
        // ficar rápida — igual à referência Python (480x360). Menos pixels =
        // muito menos latência no MediaPipe/ONNX.
        val analysisBuilder = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setResolutionSelector(resolutionSelector)
            .setTargetRotation(targetRotation)

        // Fixa o FPS numa faixa que a câmera realmente suporta. Pedir 60fps
        // fixo sem checar antes falha silenciosamente em muitos aparelhos (o
        // Camera2Interop simplesmente ignora o pedido) e, mesmo quando aceito,
        // força exposição curta — pior imagem em ambiente escuro. Consultamos
        // as faixas disponíveis e escolhemos a melhor, avisando no Logcat
        // quando 60fps fixo não é suportado nativamente.
        selecionarFaixaFps(requestedLensFacing)?.let { faixa ->
            Camera2Interop.Extender(analysisBuilder)
                .setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, faixa)
        }

        val analysis = analysisBuilder.build()

        val usandoCameraFrontal = lensFacing == CameraSelector.LENS_FACING_FRONT

        val boundAnalysis = try {
            provider.unbindAll()
            provider.bindToLifecycle(viewLifecycleOwner, selector, preview, analysis)
            analysis
        } catch (e: Exception) {
            Log.w("LibrasFragment", "Bind otimizado da camera de Libras falhou; tentando fallback simples", e)
            try {
                provider.unbindAll()
                val fallbackPreview = Preview.Builder()
                    .setTargetRotation(targetRotation)
                    .build()
                    .also { p -> p.setSurfaceProvider(binding.previewView.surfaceProvider) }
                val fallbackAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setTargetRotation(targetRotation)
                    .build()
                provider.bindToLifecycle(viewLifecycleOwner, selector, fallbackPreview, fallbackAnalysis)
                fallbackAnalysis
            } catch (fallbackError: Exception) {
                Log.e("LibrasFragment", "Nao consegui abrir a camera de Libras", fallbackError)
                Toast.makeText(requireContext(), "Nao consegui abrir a camera de Libras", Toast.LENGTH_SHORT).show()
                oldAnalyzer?.close()
                bindInProgress = false
                null
            }
        } ?: return

        if (!isAdded || _binding == null) {
            oldAnalyzer?.close()
            bindInProgress = false
            return
        }

        val appContext = requireContext().applicationContext
        val rootView = binding.root
        cameraExecutor.execute {
            oldAnalyzer?.close()

            val newAnalyzer = try {
                LibrasAnalyzer(
                    context       = appContext,
                    onLetra       = { letra, conf, _ -> onLetraDetectada(letra, conf) },
                    onFraseUpdate = { frase -> onFraseAtualizada(frase) },
                    onNoHand      = { onSemMao() },
                    onGestoLimpar = { prog -> onGestoLimpar(prog) },
                    onRepeticaoPendente = { letra -> onRepeticaoPendente(letra) },
                    onFeedback = { mensagem, nivel -> onFeedback(mensagem, nivel) },
                    onLandmarks = { hands, pose, frameAspect ->
                        onLandmarksDetected(hands, pose, frameAspect)
                    },
                    onInterrogativo = { ativo -> onInterrogativoAtualizado(ativo) }
                ).also {
                    it.setModo(modoAtual)
                    it.setEspelhamento(usandoCameraFrontal)
                }
            } catch (error: Throwable) {
                Log.e("LibrasFragment", "Nao consegui iniciar o reconhecedor de Libras", error)
                rootView.post {
                    context?.let { safeContext ->
                        Toast.makeText(
                            safeContext,
                            "Reconhecimento de Libras nao iniciou",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
                bindInProgress = false
                if (pendingBind) rootView.postDelayed(bindRunnable, 180L)
                return@execute
            }

            rootView.post {
                if (!isAdded || _binding == null || cameraExecutor.isShutdown) {
                    newAnalyzer.close()
                    bindInProgress = false
                    return@post
                }
                librasAnalyzer = newAnalyzer
                // O analyzer e recriado a cada bind; sem isto, um religamento
                // com a resposta aberta voltaria reconhecendo.
                newAnalyzer.pausado = painelRespostaAberto
                boundAnalysis.setAnalyzer(cameraExecutor, newAnalyzer)
                sincronizarAlfabeto(newAnalyzer)
                bindInProgress = false
                if (pendingBind) scheduleBindCamera(180L)
            }
        }
    }

    // Loga quando o alfabeto que os modelos realmente conhecem (vem dos
    // labels.txt exportados junto de cada modelo) muda — útil para
    // diagnosticar treinos que alteraram o conjunto de letras suportado.
    private fun sincronizarAlfabeto(analyzer: LibrasAnalyzer) {
        if (_binding == null) return
        val doModelo = analyzer.labelsAlfabeto()
        if (doModelo.isEmpty()) return
        Log.i("LibrasFragment", "Alfabeto dos modelos: $doModelo")
    }

    // Rotacao FIXA, de proposito. display.rotation e recalculado a cada giro
    // fisico do aparelho e, entregue ao CameraX, girava tanto a preview quanto
    // os frames do ImageAnalysis -- enquanto as molduras de enquadramento das
    // maos, que seguem a Activity (travada em retrato), ficavam paradas. Fora o
    // desalinhamento visual, a rotacao dos frames muda o referencial dos
    // landmarks que alimentam o reconhecimento. Como a janela e sempre retrato,
    // ROTATION_0 e a resposta certa sempre.
    private fun currentTargetRotation(): Int = Surface.ROTATION_0

    // Consulta as faixas de FPS que a câmera realmente suporta (Camera2) e
    // escolhe a melhor opção em vez de forçar 60fps às cegas. Retorna null se
    // não conseguir consultar (a câmera então usa o auto padrão do CameraX).
    private fun selecionarFaixaFps(lensFacing: Int): Range<Int>? {
        val desejada = Range(60, 60)
        return try {
            val manager = requireContext()
                .getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraId = manager.cameraIdList.firstOrNull { id ->
                val facingCam = manager.getCameraCharacteristics(id)
                    .get(CameraCharacteristics.LENS_FACING)
                val facingDesejado = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                    CameraCharacteristics.LENS_FACING_FRONT
                } else {
                    CameraCharacteristics.LENS_FACING_BACK
                }
                facingCam == facingDesejado
            } ?: return null

            val faixas = manager.getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
                ?: return null

            faixas.firstOrNull { it == desejada } ?: run {
                // Sem 60fps fixo nativo: usa a faixa de maior teto disponível
                // em vez de insistir num valor que a câmera vai ignorar.
                val melhor = faixas.maxByOrNull { it.upper }
                Log.w(
                    "LibrasFragment",
                    "Camera sem suporte nativo a 60fps fixo; usando $melhor"
                )
                melhor
            }
        } catch (e: Exception) {
            Log.w("LibrasFragment", "Falha ao consultar faixas de FPS da camera", e)
            null
        }
    }

    private fun applyPreviewAspectRatio() {
        val params = binding.previewView.layoutParams
                as? ConstraintLayout.LayoutParams ?: return
        if (params.dimensionRatio != null) {
            params.dimensionRatio = null
            binding.previewView.layoutParams = params
        } else {
            binding.previewView.requestLayout()
        }
    }

    private fun applyHudLayout() {
        if (!isLandscapeHudCompact()) {
            applyPortraitHudLayout()
            return
        }

        setPortraitHudVisible(false)
        ensureLandscapeHud()
    }

    private fun dp(value: Int): Int {
        return (value * resources.displayMetrics.density).roundToInt()
    }

    private fun isLandscapeHudCompact(): Boolean {
        return isLandscapeByBounds()
    }

    private fun applyPortraitHudLayout() {
        removeLandscapeHud()
        setPortraitHudVisible(true)
    }

    // As views marcadas com `?.` só existem em uma das variantes do layout, e
    // aí o ViewBinding as declara anuláveis: tv_live/action_row/controls_row
    // ficaram só no layout-land (o desenho retrato trocou o antigo "AO VIVO"
    // pelo botão LINHAS no topo e agrupou feedback+REPETIR+apagar no
    // feedback_row), enquanto feedback_row só existe no retrato.
    private fun setPortraitHudVisible(visible: Boolean) {
        val visibility = if (visible) View.VISIBLE else View.GONE
        binding.gradTop.visibility = visibility
        binding.gradBottom.visibility = visibility
        binding.btnExitLibras.visibility = visibility
        binding.tvLive?.visibility = visibility
        binding.btnLines.visibility = visibility
        // "Modo Libras" sai no modo oculos. Com o cadeado, sao cinco coisas
        // disputando a barra de cima, e o rotulo era o que sobrava espremido --
        // aparecia como "M..." ou "...", que nao informa nada. Ele e um rotulo
        // de estado, e no modo oculos o estado ja esta dito pelos icones.
        binding.tvModeLabel.visibility =
            if (visible && !usandoOculos) View.VISIBLE else View.GONE
        binding.btnHistory.visibility = visibility
        binding.scanFrame.visibility = visibility
        binding.actionRow?.visibility = visibility
        binding.feedbackRow?.visibility = visibility
        binding.modesRow.visibility = visibility
        binding.btnReply.visibility = visibility
        binding.btnFlip.visibility = visibility
        binding.btnOculos.visibility = visibility
        // So existe dentro do modo oculos; fora dele nao ha o que travar.
        binding.btnBloquear.visibility =
            if (visible && usandoOculos) View.VISIBLE else View.GONE
        binding.controlsRow?.visibility = visibility
        binding.chipResult.visibility = if (visible) View.INVISIBLE else View.GONE
        binding.progressConfidence.visibility = if (visible) View.INVISIBLE else View.GONE
        binding.tvFeedback.visibility = if (visible) View.INVISIBLE else View.GONE
        binding.progressClear.visibility = View.GONE

        // Estas tres NAO seguem o `visible` do HUD: elas tem estado proprio
        // (ha resposta? ha frase? o painel estava aberto?). Zera-las aqui era o
        // que fazia a resposta piscar e sumir ao voltar do microfone -- o
        // onResume chama este metodo dentro de um post(), ou seja DEPOIS de o
        // resultado da fala ja ter preenchido a caixa. A conversa na tela
        // precisa durar ate alguem apagar, nao ate a proxima troca de layout.
        binding.phraseBubble.isVisible = visible && fraseBase.isNotBlank()
        binding.replyPanel.isVisible = visible && painelRespostaAberto
        atualizarBotaoResponder()
    }

    private fun ensureLandscapeHud() {
        if (landscapeHud != null) return
        val hud = ComposeView(requireContext()).apply {
            id = R.id.hud_libras_land_root
            isClickable = false
            isFocusable = false
            setViewCompositionStrategy(ViewCompositionStrategy.DisposeOnDetachedFromWindow)
            setContent {
                LibrasLandscapeHud(
                    onExitClick = { exitLibrasMode() },
                    onFlipClick = {
                        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT)
                            CameraSelector.LENS_FACING_BACK
                        else CameraSelector.LENS_FACING_FRONT
                        scheduleBindCamera(180L)
                    }
                )
            }
        }
        binding.root.addView(
            hud,
            ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        )
        landscapeHud = hud
    }

    private fun removeLandscapeHud() {
        landscapeHud?.let { binding.root.removeView(it) }
        landscapeHud = null
    }

    // O app e travado em retrato no manifesto, entao isto so responde
    // "sim" em janela realmente deitada -- multi-janela / desktop mode. NAO
    // consulta mais nem o sensor de orientacao nem display.rotation: os dois
    // seguem a rotacao FISICA do aparelho mesmo com a Activity travada, e era
    // por eles que o HUD de paisagem entrava com a tela ainda em retrato.
    private fun isLandscapeByBounds(): Boolean {
        val rootWidth = _binding?.root?.width ?: 0
        val rootHeight = _binding?.root?.height ?: 0
        return if (rootWidth > 0 && rootHeight > 0) {
            rootWidth > rootHeight
        } else {
            resources.configuration.orientation == Configuration.ORIENTATION_LANDSCAPE
        }
    }

    override fun onResume() {
        super.onResume()

        // Voltando do reconhecedor de fala: nada da camera mudou, e religar
        // significa unbindAll() -- a preview perde a superficie e a tela PISCA
        // PRETO ate o primeiro quadro novo. Fora o custo de recriar o analyzer
        // e recarregar os modelos. O CameraX ja religa sozinho pelo lifecycle
        // quando a activity volta pro STARTED, entao aqui basta nao atrapalhar.
        val religarCamera = !voltandoDeActivityNossa
        voltandoDeActivityNossa = false

        _binding?.root?.post {
            applyPreviewAspectRatio()
            applyHudLayout()
            if (!religarCamera) return@post
            if (cameraProvider != null) {
                scheduleBindCamera(250L)
            } else {
                _binding?.root?.postDelayed({ startCamera() }, 250L)
            }
        }
    }

    // ── Botões ─────────────────────────────────────────────────────────────
    private fun setupButtons() {
        binding.btnExitLibras.setOnClickListener { exitLibrasMode() }

        binding.btnReply.setOnClickListener {
            if (binding.replyPanel.isVisible) {
                saveReplyToScreen()
                closeReplyPanel()
            } else {
                openReplyPanel(focus = true)
            }
        }

        // Atalho: responder por voz custava dois toques (abrir o painel, depois
        // achar o microfone dentro dele) e e o caminho mais comum de quem ouve.
        // Segurar o botao pula o painel e vai direto pro reconhecimento de fala.
        binding.btnReply.setOnLongClickListener {
            startSpeechReply()
            true
        }

        binding.btnConfirmLetter.setOnClickListener {
            librasAnalyzer?.repetirLetraPendente()
        }

        binding.btnReplyAudio.setOnClickListener { startSpeechReply() }

        // PRONTO guarda o texto e esconde o teclado, mas NAO sai da tela cheia:
        // e exatamente aqui que o celular e virado pra pessoa surda ler.
        binding.btnReplyClose.setOnClickListener {
            saveReplyToScreen()
            binding.etReply.clearFocus()
            hideKeyboard()
        }

        // Sair da tela cheia e uma acao separada, no X do canto.
        binding.btnReplyExit.setOnClickListener {
            saveReplyToScreen()
            closeReplyPanel()
        }

        binding.btnReplyErase.setOnClickListener {
            esvaziarResposta()
            binding.etReply.setText("")
            focusReplyText()
        }
        binding.etReply.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_DONE) {
                // Mesmo efeito do PRONTO: guarda e sai do teclado, sem fechar a
                // tela cheia, que e justamente o que vai ser mostrado.
                saveReplyToScreen()
                binding.etReply.clearFocus()
                hideKeyboard()
                true
            } else {
                false
            }
        }

        binding.btnModeAlphabet.setOnClickListener {
            modoAtual = LibrasAnalyzer.Modo.ALFABETO
            librasAnalyzer?.setModo(modoAtual)
            updateModeButtons()
        }

        binding.btnModeBody.setOnClickListener {
            modoAtual = LibrasAnalyzer.Modo.CORPO
            librasAnalyzer?.setModo(modoAtual)
            updateModeButtons()
            Toast.makeText(requireContext(), "Modo corpo ativo", Toast.LENGTH_SHORT).show()
        }

        binding.btnLines.setOnClickListener {
            linhasAtivas = !linhasAtivas
            binding.landmarkOverlay.isVisible = linhasAtivas
            if (!linhasAtivas) binding.landmarkOverlay.clear()
            binding.btnLines.text = if (linhasAtivas) "LINHAS: ON" else "LINHAS: OFF"
        }

        binding.btnFlip.setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT)
                CameraSelector.LENS_FACING_BACK
            else CameraSelector.LENS_FACING_FRONT
            scheduleBindCamera(180L)
        }

        binding.btnOculos.setOnClickListener {
            if (usandoOculos) desligarOculos()
            // Sem endereco gravado nao ha o que conectar; pedir na hora e
            // melhor que ligar e falhar sem dizer por que.
            else if (enderecoOculos().isBlank()) pedirEnderecoOculos()
            else ligarOculos()
        }
        binding.btnBloquear.setOnClickListener { bloquearTela() }
        binding.oculosLock.setOnTouchListener { capa, evento -> aoTocarNaCapa(capa, evento) }

        binding.btnOculos.setOnLongClickListener {
            pedirEnderecoOculos()
            true
        }

        // Toque: apaga só a última letra/sinal. Segurar 3s: limpa a frase toda.
        //
        // O toque longo do Android dispara em ~500ms e não existe API pra
        // alongar isso, então o hold é cronometrado aqui: o ACTION_DOWN agenda
        // a limpeza pra daqui a TEMPO_HOLD_LIMPAR_MS e a barra progress_clear
        // enche nesse intervalo — o mesmo retorno visual que o gesto de mão
        // aberta já dá. Soltar antes cancela o agendamento e vale como toque.
        binding.btnDeleteLetter.setOnClickListener {
            librasAnalyzer?.apagarUltima()
        }
        binding.btnDeleteLetter.setOnTouchListener { view, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    view.isPressed = true
                    toqueLixeiraConsumido = false
                    iniciarProgressoHold()
                    view.postDelayed(limparTudoRunnable, TEMPO_HOLD_LIMPAR_MS)
                }

                // Dedo arrastou pra fora: desiste do hold sem apagar nada. Sem
                // isso a limpeza dispararia mesmo com o dedo longe do botão,
                // já que o toque continua sendo entregue a esta view.
                MotionEvent.ACTION_MOVE -> {
                    val dentro = event.x >= 0f && event.y >= 0f &&
                        event.x <= view.width && event.y <= view.height
                    if (!dentro && !toqueLixeiraConsumido) {
                        toqueLixeiraConsumido = true
                        view.isPressed = false
                        cancelarHoldLimpar(view)
                    }
                }

                MotionEvent.ACTION_UP -> {
                    view.isPressed = false
                    cancelarHoldLimpar(view)
                    // performClick em vez de chamar apagarUltima direto: mantém
                    // o caminho de acessibilidade, que ativa a view por clique.
                    if (!toqueLixeiraConsumido) view.performClick()
                }

                MotionEvent.ACTION_CANCEL -> {
                    view.isPressed = false
                    cancelarHoldLimpar(view)
                }

                else -> return@setOnTouchListener false
            }
            true
        }

        // Duas metades da conversa, dois historicos. O botao do topo abre o
        // que foi SINALIZADO; o de dentro da tela de resposta abre o que foi
        // RESPONDIDO. Antes uma lista so misturava os dois e obrigava a
        // garimpar pra achar o que se procurava.
        binding.btnHistory.setOnClickListener {
            abrirHistorico("LIBRAS", "Libras")
        }
        binding.btnReplyHistory.setOnClickListener {
            abrirHistorico("RESPOSTA", "Respostas")
        }
    }

    private fun abrirHistorico(origem: String, titulo: String) {
        HistoryBottomSheet.newInstance(historyStore.entriesDe(origem), titulo)
            .also { sheet ->
                sheet.onClearConversation = { historyStore.limpar(origem) }
            }
            .show(childFragmentManager, "history_$origem")
    }

    private fun updateModeButtons() {
        val dark = ContextCompat.getColor(requireContext(), R.color.text_on_gold)
        val light = ContextCompat.getColor(requireContext(), R.color.text_primary)

        if (modoAtual == LibrasAnalyzer.Modo.ALFABETO) {
            binding.btnModeAlphabet.setBackgroundResource(R.drawable.vf_bg_mode_active)
            binding.btnModeAlphabet.setTextColor(dark)
            binding.btnModeBody.setBackgroundResource(R.drawable.vf_bg_mode_inactive)
            binding.btnModeBody.setTextColor(light)
            binding.btnModeBody.text = "CORPO"
            binding.tvModeLabel.text = "Modo Libras"
        } else {
            binding.btnModeAlphabet.setBackgroundResource(R.drawable.vf_bg_mode_inactive)
            binding.btnModeAlphabet.setTextColor(light)
            binding.btnModeBody.setBackgroundResource(R.drawable.vf_bg_mode_active)
            binding.btnModeBody.setTextColor(dark)
            binding.btnModeBody.text = "CORPO"
            binding.tvModeLabel.text = "Modo Corpo"
        }
        binding.btnModeAlphabet.alpha = 1f
        binding.btnModeBody.alpha = 1f
    }

    // ── Hold da lixeira ────────────────────────────────────────────────────

    /** Enche progress_clear ao longo dos 3s, pra o hold ter fim visível. */
    private fun iniciarProgressoHold() {
        holdAnimator?.cancel()
        val barra = _binding?.progressClear ?: return
        barra.progress = 0
        barra.isVisible = true
        holdAnimator = ValueAnimator.ofInt(0, 100).apply {
            duration = TEMPO_HOLD_LIMPAR_MS
            addUpdateListener { anim ->
                _binding?.progressClear?.progress = anim.animatedValue as Int
            }
            start()
        }
    }

    private fun pararProgressoHold() {
        holdAnimator?.cancel()
        holdAnimator = null
        _binding?.progressClear?.isVisible = false
    }

    private fun cancelarHoldLimpar(view: View) {
        view.removeCallbacks(limparTudoRunnable)
        pararProgressoHold()
    }

    private fun startSpeechReply() {
        voltandoDeActivityNossa = true
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
            )
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "pt-BR")
            putExtra(RecognizerIntent.EXTRA_PROMPT, "Fale a resposta")
        }

        try {
            speechLauncher.launch(intent)
        } catch (e: ActivityNotFoundException) {
            Toast.makeText(
                requireContext(),
                "Reconhecimento de voz indisponivel neste celular",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    /**
     * Deixa visivel o estado do painel no proprio botao: fechado ele e um
     * contorno dourado (acao disponivel), aberto ele fica preenchido, porque
     * ali o toque ja nao abre nada -- guarda a resposta e fecha.
     */
    private fun atualizarBotaoResponder() {
        val aberto = binding.replyPanel.isVisible
        fecharRespostaNoVoltar.isEnabled = aberto
        // Com a resposta aberta ninguem esta sinalizando: o que a camera
        // reconhecer ali e acidente, e entrava na frase sendo falado em voz alta.
        librasAnalyzer?.pausado = aberto
        binding.btnReply.setBackgroundResource(
            if (aberto) R.drawable.vf_bg_action_primary_on else R.drawable.vf_bg_action_primary
        )
        binding.btnReply.imageTintList = ColorStateList.valueOf(
            ContextCompat.getColor(
                requireContext(),
                // Fechado: branco, como todo icone de botao do app. Aberto: escuro,
                // porque o fundo passa a ser dourado preenchido.
                if (aberto) R.color.text_on_gold else R.color.text_primary
            )
        )
    }

    private fun openReplyPanel(focus: Boolean = false) {
        if (isLandscapeHudCompact()) {
            binding.replyPanel.isVisible = false
            painelRespostaAberto = false
            Toast.makeText(requireContext(), "Resposta ocultada no HUD compacto", Toast.LENGTH_SHORT).show()
            return
        }
        binding.replyPanel.isVisible = true
        painelRespostaAberto = true
        atualizarBotaoResponder()
        if (binding.etReply.text.isNullOrBlank() && respostaAtual.isNotBlank()) {
            binding.etReply.setText(respostaAtual)
            binding.etReply.setSelection(respostaAtual.length)
        }
        if (focus) focusReplyText()
    }

    private fun focusReplyText() {
        binding.replyPanel.isVisible = true
        painelRespostaAberto = true
        atualizarBotaoResponder()
        binding.etReply.requestFocus()
        val imm = requireContext().getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
        imm.showSoftInput(binding.etReply, InputMethodManager.SHOW_IMPLICIT)
    }

    private fun hideKeyboard() {
        val imm = requireContext().getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(binding.root.windowToken, 0)
    }

    private fun closeReplyPanel() {
        binding.replyPanel.isVisible = false
        painelRespostaAberto = false
        atualizarBotaoResponder()
        binding.etReply.clearFocus()
        hideKeyboard()
    }

    /**
     * Espelha o campo na bolha e no historico. Campo vazio significa "nao ha
     * resposta", e nao "feche tudo": quem fecha e quem sai.
     */
    private fun saveReplyToScreen(): Boolean {
        val texto = binding.etReply.text?.toString().orEmpty().trim()
        if (texto.isBlank()) {
            esvaziarResposta()
            return false
        }
        respostaAtual = texto
        historyStore.registrarMensagemResposta(texto)
        return true
    }

    /** Apaga a resposta em todos os lugares onde ela existe, sem mexer no painel. */
    private fun esvaziarResposta() {
        historyStore.removerRespostaAtual()
        respostaAtual = ""
    }

    // ── Callbacks ──────────────────────────────────────────────────────────
    private fun onLetraDetectada(letra: String, confianca: Float) {
        activity?.runOnUiThread {
            if (_binding == null) return@runOnUiThread
            if (isLandscapeHudCompact()) {
                binding.chipResult.visibility = View.GONE
                binding.progressConfidence.visibility = View.GONE
                return@runOnUiThread
            }
            if (letra != "-") {
                val porcentagem = (confianca * 100).toInt().coerceIn(0, 100)
                binding.chipResult.text = "$letra   $porcentagem% conf"
                binding.chipResult.visibility = View.VISIBLE
                binding.progressConfidence.visibility = View.VISIBLE
                binding.progressConfidence.progress = porcentagem

                binding.progressConfidence.progressTintList = confidenceTint(confianca)
                binding.progressConfidence.progressBackgroundTintList = TINT_CONFIANCA_FUNDO
            } else {
                ultimaLetraChip = ""
                binding.chipResult.visibility = View.INVISIBLE
                binding.progressConfidence.visibility = View.INVISIBLE
                binding.progressConfidence.progress = 0
            }
        }
    }

    private fun confidenceTint(confianca: Float): ColorStateList =
        when {
            confianca >= 0.92f -> TINT_CONFIANCA_ALTA
            confianca >= 0.84f -> TINT_CONFIANCA_MEDIA
            else -> TINT_CONFIANCA_BAIXA
        }

    private fun onFeedback(mensagem: String, nivel: Int) {
        activity?.runOnUiThread {
            if (_binding == null) return@runOnUiThread
            if (isLandscapeHudCompact()) {
                binding.tvFeedback.visibility = View.GONE
                return@runOnUiThread
            }
            if (mensagem.isBlank()) {
                binding.tvFeedback.visibility = View.INVISIBLE
                binding.scanFrame.setFeedbackLevel(ScanFrameView.FEEDBACK_NEUTRO)
                return@runOnUiThread
            }
            binding.tvFeedback.text = mensagem
            binding.tvFeedback.visibility = View.VISIBLE
            binding.tvFeedback.setTextColor(feedbackColor(nivel))
            binding.scanFrame.setFeedbackLevel(nivel)
        }
    }

    private fun feedbackColor(nivel: Int): Int {
        val ctx = requireContext()
        return when (nivel) {
            LibrasAnalyzer.FEEDBACK_BOM -> ContextCompat.getColor(ctx, R.color.gold_light)
            LibrasAnalyzer.FEEDBACK_ALERTA -> ContextCompat.getColor(ctx, R.color.feedback_alerta)
            else -> ContextCompat.getColor(ctx, R.color.text_primary)
        }
    }

    private fun onFraseAtualizada(frase: String) {
        activity?.runOnUiThread {
            if (_binding == null) return@runOnUiThread
            fraseBase = frase
            if (isLandscapeHudCompact()) {
                binding.phraseBubble.isVisible = false
                fraseAnterior = frase
                return@runOnUiThread
            }
            binding.tvPhrase.text = fraseExibida()
            binding.phraseBubble.isVisible = frase.isNotBlank()

            // TTS: fala o que acabou de entrar. O trecho e a pronúncia dele
            // são decididos no PhraseOutput — textoParaVoz é o que garante que
            // o motor de voz não leia "AV" como "avenida".
            val trecho = PhraseOutput.trechoParaFalar(frase, fraseAnterior)
            val fala = PhraseOutput.textoParaVoz(trecho)
            if (frase.length > fraseAnterior.length) {
                if (fala.isNotBlank()) {
                    tts?.speak(fala, TextToSpeech.QUEUE_FLUSH, null, null)
                }
                vibrateConfirmation()
            }

            historyStore.registrarMensagemLibras(frase)
            fraseAnterior = frase
        }
    }

    private fun fraseExibida(): String = PhraseOutput.exibicao(fraseBase, interrogativoAtivo)

    private fun onInterrogativoAtualizado(ativo: Boolean) {
        activity?.runOnUiThread {
            if (_binding == null) return@runOnUiThread
            if (interrogativoAtivo == ativo) return@runOnUiThread
            interrogativoAtivo = ativo
            if (!isLandscapeHudCompact()) {
                binding.tvPhrase.text = fraseExibida()
            }
        }
    }

    private fun onSemMao() {
        activity?.runOnUiThread {
            if (_binding == null) return@runOnUiThread
            binding.chipResult.visibility = if (isLandscapeHudCompact()) View.GONE else View.INVISIBLE
            binding.progressConfidence.visibility = if (isLandscapeHudCompact()) View.GONE else View.INVISIBLE
            binding.progressConfidence.progress = 0
            binding.progressClear.isVisible = false
        }
    }

    // Chamado da thread do analyzer. update()/clear() usam postInvalidate(),
    // então são seguros fora da UI thread.
    private fun onLandmarksDetected(
        hands: List<FloatArray>,
        pose: FloatArray?,
        frameAspect: Float
    ) {
        if (!linhasAtivas) return
        _binding?.landmarkOverlay?.update(hands, pose, frameAspect)
    }

    private fun onRepeticaoPendente(letra: String?) {
        activity?.runOnUiThread {
            if (_binding == null) return@runOnUiThread
            binding.btnConfirmLetter.isVisible = letra != null
            binding.btnConfirmLetter.text = if (letra != null) "REPETIR $letra" else "REPETIR"
        }
    }

    private fun onGestoLimpar(progresso: Float) {
        activity?.runOnUiThread {
            if (_binding == null) return@runOnUiThread
            // Um hold na lixeira está usando a mesma barra: não deixa o gesto
            // de mão aberta sobrescrever o progresso dele.
            if (holdAnimator != null) return@runOnUiThread
            binding.progressClear.isVisible = progresso > 0f
            binding.progressClear.progress  = (progresso * 100).toInt()
            if (progresso == 0f) {
                // Frase foi limpa — reseta referência
                fraseAnterior = ""
            }
        }
    }

    // ── Saída segura do modo Libras ────────────────────────────────────────
    @Suppress("DEPRECATION")
    private fun vibrateConfirmation() {
        val vibrator = requireContext().getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
            ?: return
        if (!vibrator.hasVibrator()) return

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(35L, 90))
        } else {
            vibrator.vibrate(35L)
        }
    }

    // Desvincula a câmera ANTES de fechar o analyzer, e fecha o analyzer no
    // mesmo executor (single-thread) que entrega os frames em analyze().
    // Isso garante que nenhum frame ainda esteja em processamento quando os
    // recursos nativos (ONNX Runtime, MediaPipe, TFLite) forem liberados —
    // fechar esses recursos enquanto analyze() roda em paralelo em outra
    // thread é undefined behavior e foi a causa raiz dos crashes nativos
    // intermitentes ao sair do modo Libras (mais frequentes em x86_64).
    private fun closeAnalyzerSafely() {
        cameraProvider?.unbindAll()
        cameraProvider = null

        val analyzerToClose = librasAnalyzer
        librasAnalyzer = null

        if (analyzerToClose != null) {
            if (!cameraExecutor.isShutdown) {
                cameraExecutor.execute { analyzerToClose.close() }
            } else {
                analyzerToClose.close()
            }
        }
        if (!cameraExecutor.isShutdown) cameraExecutor.shutdown()
    }

    // A limpeza da câmera/analyzer (closeAnalyzerSafely) não acontece aqui:
    // popBackStack()/navigate() disparam onDestroyView() do fragmento, que já
    // faz essa limpeza. Fazer duas vezes era redundante e o código antigo
    // ainda arriscava a mesma race condition se a ordem das chamadas mudasse.
    // ── Camera dos oculos (ESP32 por Wi-Fi) ──────────────────────────────
    //
    // A troca e so de FONTE. Nada do reconhecimento muda: os quadros vao pro
    // mesmo LibrasAnalyzer.analisarQuadro que a camera do celular alimenta.

    /** Brilho da tela travada. Zero cravado alguns aparelhos leem como "automatico". */
    private val BRILHO_TRAVADO = 0.01f

    /** Quanto da altura da tela o dedo precisa subir pra destravar. */
    private val FRACAO_DESTRAVE = 0.22f

    private fun prefsOculos() =
        requireContext().getSharedPreferences("oculos", Context.MODE_PRIVATE)

    private fun enderecoOculos(): String =
        prefsOculos().getString("url", "").orEmpty()

    private fun pedirEnderecoOculos() {
        val campo = EditText(requireContext()).apply {
            // Enquanto a placa nao chega, o endereco e o do mock rodando no PC
            // (oculos/mock_esp32_cam.py). Depois vira o IP do proprio ESP32.
            setText(enderecoOculos().ifBlank { "http://192.168.15.10:8080/stream" })
            setSelection(text.length)
        }
        AlertDialog.Builder(requireContext())
            .setTitle("Endereco dos oculos")
            .setMessage(
                "Endereco do stream.\n\n" +
                    "Oculos: ${EnderecoDosOculos.URL}\n" +
                    "O app entra na rede ${EnderecoDosOculos.REDE} sozinho, e o " +
                    "celular continua na rede de casa pro resto.\n\n" +
                    "Mock no PC: o IP do PC com /stream no fim. Ai o app usa o " +
                    "Wi-Fi em que o celular ja esta.")
            .setView(campo)
            .setPositiveButton("Conectar") { _, _ ->
                val url = campo.text.toString().trim()
                prefsOculos().edit().putString("url", url).apply()
                if (url.isNotBlank()) ligarOculos()
            }
            .setNegativeButton("Cancelar", null)
            .show()
    }

    private fun ligarOculos() {
        // O analyzer nasce junto com a camera e carrega os modelos, o que
        // demora e nao pode acontecer na thread da interface. Se ainda nao
        // existe, e porque a camera acabou de abrir -- pedir pra tentar de novo
        // e melhor que congelar a tela carregando modelo aqui.
        val endereco = enderecoOculos()
        val analyzer = librasAnalyzer
        if (analyzer == null) {
            Toast.makeText(requireContext(),
                "Espere a camera abrir e tente de novo", Toast.LENGTH_SHORT).show()
            return
        }

        usandoOculos = true
        // Solta a camera do celular: sem isto ela continua ligada gastando
        // bateria e entregando quadros que ninguem quer.
        cameraProvider?.unbindAll()
        // INVISIBLE, nao GONE, e a diferenca entre funcionar e nao funcionar.
        //
        // iv_oculos e landmark_overlay tem largura e altura 0dp amarradas as
        // QUATRO bordas do preview_view. No ConstraintLayout um view GONE
        // colapsa num ponto, e quem esta preso a ele colapsa junto: os dois
        // viravam 0x0. O stream continuava chegando, o reconhecimento
        // continuava rodando a 13 quadros/s, e a tela ficava preta -- sem erro
        // nenhum, porque erro nao houve; a imagem era desenhada num retangulo
        // de tamanho zero.
        //
        // INVISIBLE nao desenha mas mantem as medidas, que e exatamente o que
        // esses dois precisam do preview_view.
        binding.previewView.visibility = View.INVISIBLE
        binding.ivOculos.isVisible = true
        // A ImageView dos oculos usa fitCenter: a imagem 4:3 cabe inteira e
        // sobram faixas em cima e embaixo. O desenho dos landmarks precisa
        // saber disso, senao mapeia como se a imagem enchesse a tela e joga o
        // esqueleto centenas de pixels acima da mao.
        binding.landmarkOverlay.setEncaixe(EncaixeDeQuadro.Modo.INTEIRA)
        binding.btnBloquear.isVisible = true
        binding.tvModeLabel.isVisible = false   // ver setPortraitHudVisible
        // A tela nao pode apagar. Apagando, o Android para a activity: o
        // reconhecimento morre e a conexao com os oculos cai junto. Quem quiser
        // guardar o celular no bolso usa o cadeado, que escurece sem apagar.
        manterTelaAcesa(true)
        binding.btnOculos.alpha = 1f
        binding.landmarkOverlay.clear()

        // Camera que aponta pra FORA, como a traseira do celular: nao espelha.
        // A frontal espelha pra casar com o dataset, que foi gravado em selfie.
        analyzer.setEspelhamento(false)

        fonteOculos?.parar()
        // Apontando pra placa, o app pede pro Android entrar na rede dela --
        // so pro app, sem tirar o celular da rede de casa. Apontando pro mock,
        // usa o Wi-Fi em que o celular ja esta.
        val naPlaca = EnderecoDosOculos.ehAPlaca(endereco)
        fonteOculos = NetworkStreamSource(
            context = requireContext().applicationContext,
            url = endereco,
            ssidOculos = if (naPlaca) EnderecoDosOculos.REDE else null,
            senhaOculos = if (naPlaca) EnderecoDosOculos.SENHA else null,
            aoQuadro = { quadro ->
                // Duas copias porque os dois consumidores tem donos diferentes:
                // o analisarQuadro RECICLA o bitmap que recebe, e a tela precisa
                // do dela viva ate o proximo quadro chegar.
                // Com a tela travada ninguem esta olhando. Pular a copia
                // corta uns 300 KB por quadro, treze vezes por segundo, mais um
                // redesenho de tela inteira -- e o unico gasto que da pra tirar
                // sem parar de entender os sinais.
                if (!telaBloqueada) {
                    val paraTela = quadro.copy(android.graphics.Bitmap.Config.ARGB_8888, false)
                    view?.post {
                        if (usandoOculos && _binding != null && !telaBloqueada) {
                            binding.ivOculos.setImageBitmap(paraTela)
                        } else {
                            paraTela.recycle()
                        }
                    }
                }
                // rotacao 0: a camera fica fixa na armacao, ao contrario da do
                // celular, que informa a rotacao a cada quadro.
                librasAnalyzer?.analisarQuadro(quadro, 0f, false) ?: quadro.recycle()
            },
            aoEstado = { estado -> view?.post { mostrarEstadoOculos(estado) } }
        ).also { it.iniciar() }
    }

    /**
     * O que a pessoa ve enquanto os oculos conectam, caem e voltam.
     *
     * Falha NAO vai pro chip de feedback, e isso e o principal aqui. Duas
     * razoes, as duas vistas na tela: aquele chip e do reconhecimento
     * ("CENTRALIZE A MAO"), e um texto de rede em vermelho ali se le como se a
     * mao estivesse errada; e o motivo cru era comprido o bastante pra esticar
     * a linha do chip e empurrar a lixeira pra fora do celular.
     *
     * Falha vira aviso passageiro, em portugues, dizendo o que fazer. O motivo
     * cru vai pro logcat, que e onde ele serve.
     */
    private fun mostrarEstadoOculos(estado: MjpegClient.Estado) {
        if (!usandoOculos || _binding == null) return
        when (estado) {
            is MjpegClient.Estado.Conectando ->
                onFeedback("PROCURANDO OCULOS", LibrasAnalyzer.FEEDBACK_NEUTRO)

            is MjpegClient.Estado.Recebendo -> {
                // Nao anuncia nada: a imagem aparecendo ja e o aviso, e no
                // quadro seguinte o proprio reconhecimento ocupa o chip.
                ultimoAvisoOculos = null
                onFeedback("", LibrasAnalyzer.FEEDBACK_NEUTRO)
            }

            is MjpegClient.Estado.Erro -> {
                Log.w("Oculos", "falha na conexao: ${estado.motivo}")
                val aviso = MensagemDeErro.emPortugues(estado.motivo)
                // O laco tenta de novo a cada poucos segundos; sem esta guarda
                // seria um aviso atras do outro, pra sempre.
                if (aviso != ultimoAvisoOculos) {
                    ultimoAvisoOculos = aviso
                    Toast.makeText(requireContext(), aviso, Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    // ── Tela travada (modo oculos) ───────────────────────────────────────
    //
    // Pra que serve: no uso de verdade o celular vai pro bolso e os oculos
    // continuam vendo. A tela precisa continuar ACESA -- apagando, o Android
    // para a activity e o reconhecimento junto --, mas uma tela acesa no bolso
    // aceita toque e gasta bateria. Travada, ela fica preta, surda e barata.

    private fun bloquearTela() {
        if (!usandoOculos || telaBloqueada || _binding == null) return
        telaBloqueada = true
        binding.oculosLock.isVisible = true
        // Coberto pela capa, mas o Android nao sabe disso e redesenharia a cada
        // quadro assim mesmo.
        binding.landmarkOverlay.visibility = View.INVISIBLE
        ignorarVoltarBloqueado.isEnabled = true
        // Preto numa tela OLED e pixel desligado; com o brilho no minimo, o
        // custo de manter a tela acesa cai pra perto de nada.
        ajustarBrilho(BRILHO_TRAVADO)
    }

    private fun desbloquearTela() {
        if (!telaBloqueada) return
        telaBloqueada = false
        _binding?.oculosLock?.isVisible = false
        _binding?.landmarkOverlay?.visibility = View.VISIBLE
        ignorarVoltarBloqueado.isEnabled = false
        ajustarBrilho(WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE)
    }

    /**
     * Destrava com um arrasto pra cima, e engole todo o resto.
     *
     * Arrasto, e nao toque: no bolso, um toque acontece sozinho o tempo todo.
     * Um movimento longo e deliberado, nao.
     */
    private fun aoTocarNaCapa(capa: View, evento: MotionEvent): Boolean {
        when (evento.actionMasked) {
            MotionEvent.ACTION_DOWN -> yInicialDestrave = evento.rawY
            MotionEvent.ACTION_UP -> {
                val subiu = yInicialDestrave - evento.rawY
                if (subiu >= capa.height * FRACAO_DESTRAVE) desbloquearTela()
            }
        }
        return true
    }

    private fun ajustarBrilho(valor: Float) {
        val janela = activity?.window ?: return
        janela.attributes = janela.attributes.apply { screenBrightness = valor }
    }

    private fun manterTelaAcesa(manter: Boolean) {
        val janela = activity?.window ?: return
        if (manter) janela.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else janela.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private fun desligarOculos() {
        usandoOculos = false
        ultimoAvisoOculos = null
        desbloquearTela()
        manterTelaAcesa(false)
        _binding?.btnBloquear?.isVisible = false
        _binding?.tvModeLabel?.isVisible = true
        fonteOculos?.parar()
        fonteOculos = null
        binding.ivOculos.setImageDrawable(null)
        binding.ivOculos.isVisible = false
        // De volta pra PreviewView, que e fillCenter.
        binding.landmarkOverlay.setEncaixe(EncaixeDeQuadro.Modo.CORTANDO)
        binding.previewView.isVisible = true
        binding.btnOculos.alpha = 0.6f
        binding.landmarkOverlay.clear()
        // Religa a camera do celular; o bind recria o analyzer com o
        // espelhamento certo pra lente em uso.
        scheduleBindCamera(120L)
    }

    private fun exitLibrasMode() {
        if (!isAdded || view == null) return
        val navController = findNavController()
        if (navController.currentDestination?.id != R.id.nav_libras) return

        val voltouParaCamera = navController.popBackStack(R.id.nav_camera, false)
        if (!voltouParaCamera && navController.currentDestination?.id == R.id.nav_libras) {
            navController.navigate(R.id.action_libras_to_camera)
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) tts?.language = Locale("pt", "BR")
    }

    override fun onDestroyView() {
        super.onDestroyView()
        // Brilho e tela acesa sao da JANELA, nao do fragmento: sem devolver
        // aqui, sair do modo Libras deixaria o app inteiro escuro e sem apagar.
        desbloquearTela()
        manterTelaAcesa(false)
        // Antes de tudo: as threads dos oculos seguram uma referencia ao
        // analyzer, e ele e fechado logo abaixo.
        fonteOculos?.parar()
        fonteOculos = null
        landscapeHud = null
        _binding?.btnDeleteLetter?.removeCallbacks(limparTudoRunnable)
        holdAnimator?.cancel()
        holdAnimator = null
        try {
            closeAnalyzerSafely()
        } catch (e: Exception) {
            e.printStackTrace()
        }
        tts?.shutdown()
        _binding = null
    }
}
