package com.visuall.app.libras

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.content.res.Configuration
import android.content.res.ColorStateList
import android.net.Uri
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
import android.view.Surface
import android.view.LayoutInflater
import android.view.OrientationEventListener
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
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
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.visuall.app.R
import com.visuall.app.databinding.FragmentLibrasBinding
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.Normalizer
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.roundToInt

class LibrasFragment : Fragment(), TextToSpeech.OnInitListener {

    private var _binding: FragmentLibrasBinding? = null
    private val binding get() = _binding!!

    private lateinit var cameraExecutor: ExecutorService
    private var librasAnalyzer: LibrasAnalyzer? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var tts: TextToSpeech? = null
    private var isPhysicalLandscape = false
    private var orientationListener: OrientationEventListener? = null
    private var landscapeHud: View? = null

    private var lensFacing = CameraSelector.LENS_FACING_FRONT

    private val historicoEntries = arrayListOf<HistoryEntry>()
    private var ultimaMensagemLibras = ""
    private var indiceLibrasAtual = -1
    private var indiceRespostaAtual = -1

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
    private val letrasCalibracao = listOf(
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
    )
    private val letrasDinamicasTreino = setOf("H", "J", "K", "X", "Z")
    private val palavrasSugeridas = listOf(
        "ajuda", "ajudar", "agua", "amigo", "amanha", "aprender", "aqui",
        "banheiro", "bom", "boa", "casa", "comida", "computador", "conversa",
        "conversar", "desculpa", "dor", "escola", "estou", "familia", "feliz",
        "hoje", "jovi", "libras", "mae", "medico", "nao", "obrigado", "obrigada",
        "oi", "onde", "pai", "pessoa", "por favor", "preciso", "professor",
        "quero", "responder", "sim", "surdo", "tudo", "voce", "voltar"
    )
    private val sugestoesContextuais = mapOf(
        "bom" to listOf("dia"),
        "boa" to listOf("tarde", "noite"),
        "por" to listOf("favor"),
        "eu" to listOf("preciso", "quero", "estou"),
        "voce" to listOf("pode", "quer", "entendeu"),
        "preciso" to listOf("ajuda", "agua", "medico"),
        "quero" to listOf("comida", "agua", "conversar")
    )
    private var indiceCalibracao = letrasCalibracao.indexOf("E").coerceAtLeast(0)

    private data class TrainingProgress(
        val percent: Int,
        val missingLetters: List<String>,
        val trainedLetters: Int,
        val totalSamples: Int
    )

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
            binding.tvReply.text = texto
            binding.replyBubble.isVisible = true
            speakReply(texto)
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
        carregarConversaSalva()
        view.post {
            applyPreviewAspectRatio()
            applyHudLayout()
        }
        startCamera()
        setupButtons()
        updateModeButtons()
        setupOrientationHudListener()
    }

    // ── Inicia provider uma única vez ──────────────────────────────────────
    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(requireContext())
        future.addListener({
            cameraProvider = future.get()
            bindCamera()
        }, ContextCompat.getMainExecutor(requireContext()))
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

        // Desvincula câmera (para entrega de frames no pipeline)
        provider.unbindAll()

        // Recria executor se foi encerrado
        if (cameraExecutor.isShutdown) {
            cameraExecutor = Executors.newSingleThreadExecutor()
        }

        // Pede uma análise de baixa resolução (~640x480, 4:3) para a inferência
        // ficar rápida — igual à referência Python (480x360). Menos pixels =
        // muito menos latência no MediaPipe/ONNX.
        val analysisBuilder = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
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

        try {
            provider.bindToLifecycle(viewLifecycleOwner, selector, preview, analysis)
        } catch (e: Exception) {
            e.printStackTrace()
            oldAnalyzer?.close()
            return
        }

        val appContext = requireContext().applicationContext
        cameraExecutor.execute {
            oldAnalyzer?.close()

            val newAnalyzer = LibrasAnalyzer(
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

            if (!isAdded || _binding == null || cameraExecutor.isShutdown) {
                newAnalyzer.close()
                return@execute
            }

            librasAnalyzer = newAnalyzer
            analysis.setAnalyzer(cameraExecutor, newAnalyzer)
        }
    }

    private fun currentTargetRotation(): Int {
        return _binding?.previewView?.display?.rotation
            ?: activity?.windowManager?.defaultDisplay?.rotation
            ?: Surface.ROTATION_0
    }

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
        val ratio = if (isLandscapeByBounds()) "4:3" else "3:4"
        val params = binding.previewView.layoutParams
                as? ConstraintLayout.LayoutParams ?: return
        // Só mexe se a proporção realmente mudou. Reaplicar layout à toa
        // destrói e recria a surface do preview.
        if (params.dimensionRatio == ratio) return
        // Alteramos APENAS os params do preview. Antes usávamos
        // ConstraintSet.clone()/applyTo() na raiz inteira, o que reconstruía
        // todo o layout logo após o bind da câmera e derrubava a surface —
        // era por isso que a câmera "fechava" ao entrar no modo Libras e só
        // voltava ao inverter a câmera (que refaz o bind).
        params.dimensionRatio = ratio
        binding.previewView.layoutParams = params
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

    private fun setPortraitHudVisible(visible: Boolean) {
        val visibility = if (visible) View.VISIBLE else View.GONE
        binding.gradTop.visibility = visibility
        binding.gradBottom.visibility = visibility
        binding.btnExitLibras.visibility = visibility
        binding.tvLive.visibility = visibility
        binding.tvModeLabel.visibility = visibility
        binding.scanFrame.visibility = visibility
        binding.actionRow.visibility = visibility
        binding.modesRow.visibility = visibility
        binding.controlsRow.visibility = visibility
        binding.chipResult.visibility = if (visible) View.INVISIBLE else View.GONE
        binding.progressConfidence.visibility = if (visible) View.INVISIBLE else View.GONE
        binding.tvFeedback.visibility = if (visible) View.INVISIBLE else View.GONE
        binding.progressClear.visibility = View.GONE
        binding.replyBubble.visibility = View.GONE
        binding.phraseBubble.visibility = View.GONE
        binding.suggestionsRow.visibility = View.GONE
        binding.replyPanel.visibility = View.GONE
        binding.calibrationPanel.visibility = View.GONE
    }

    private fun ensureLandscapeHud() {
        if (landscapeHud != null) return
        val hud = layoutInflater.inflate(R.layout.hud_libras_land, binding.root, false)
        hud.id = R.id.hud_libras_land_root
        binding.root.addView(
            hud,
            ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        )
        hud.findViewById<View>(R.id.btn_exit_libras_land).setOnClickListener {
            exitLibrasMode()
        }
        hud.findViewById<View>(R.id.btn_flip_libras_land).setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT)
                CameraSelector.LENS_FACING_BACK
            else CameraSelector.LENS_FACING_FRONT
            bindCamera()
        }
        landscapeHud = hud
    }

    private fun removeLandscapeHud() {
        landscapeHud?.let { binding.root.removeView(it) }
        landscapeHud = null
    }

    private fun isLandscapeByBounds(): Boolean {
        if (isPhysicalLandscape) return true

        val displayRotation = _binding?.previewView?.display?.rotation
            ?: activity?.windowManager?.defaultDisplay?.rotation
        if (displayRotation == Surface.ROTATION_90 || displayRotation == Surface.ROTATION_270) {
            return true
        }

        val rootWidth = _binding?.root?.width ?: 0
        val rootHeight = _binding?.root?.height ?: 0
        return if (rootWidth > 0 && rootHeight > 0) {
            rootWidth > rootHeight
        } else {
            resources.configuration.orientation == Configuration.ORIENTATION_LANDSCAPE
        }
    }

    private fun setupOrientationHudListener() {
        val context = context ?: return
        orientationListener = object : OrientationEventListener(context) {
            override fun onOrientationChanged(orientation: Int) {
                if (orientation == ORIENTATION_UNKNOWN) return
                val landscape = orientation in 60..120 || orientation in 240..300
                if (landscape == isPhysicalLandscape) return

                isPhysicalLandscape = landscape
                _binding?.root?.post {
                    applyPreviewAspectRatio()
                    applyHudLayout()
                }
            }
        }.also { listener ->
            if (listener.canDetectOrientation()) listener.enable()
        }
    }

    override fun onResume() {
        super.onResume()
        _binding?.root?.post {
            applyPreviewAspectRatio()
            applyHudLayout()
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
                openReplyPanel()
            }
        }

        binding.replyBubble.setOnClickListener {
            openReplyPanel(focus = true)
        }

        binding.btnReplyClear.setOnClickListener {
            clearReply()
        }

        binding.btnCalibrate.setOnClickListener {
            openCalibrationPanel()
        }

        binding.btnSuggestion1.setOnClickListener { applySuggestionFrom(binding.btnSuggestion1) }
        binding.btnSuggestion2.setOnClickListener { applySuggestionFrom(binding.btnSuggestion2) }
        binding.btnSuggestion3.setOnClickListener { applySuggestionFrom(binding.btnSuggestion3) }

        binding.btnCalibrationPrev.setOnClickListener {
            indiceCalibracao = if (indiceCalibracao == 0) {
                letrasCalibracao.lastIndex
            } else {
                indiceCalibracao - 1
            }
            updateCalibrationPanel()
        }

        binding.btnCalibrationNext.setOnClickListener {
            indiceCalibracao = if (indiceCalibracao == letrasCalibracao.lastIndex) {
                0
            } else {
                indiceCalibracao + 1
            }
            updateCalibrationPanel()
        }

        binding.btnCalibrationRecord.setOnClickListener {
            val letra = currentCalibrationLetter()
            librasAnalyzer?.startCalibration(letra)
            updateCalibrationCaptureProgress()
            val instrucao = if (letra in letrasDinamicasTreino) {
                "Faca o movimento de $letra na camera"
            } else {
                "Segure o sinal de $letra por 2 segundos"
            }
            Toast.makeText(requireContext(), instrucao, Toast.LENGTH_SHORT).show()
        }

        binding.btnCalibrationSave.setOnClickListener {
            val letra = currentCalibrationLetter()
            val saved = librasAnalyzer?.finishCalibration() == true
            if (saved) {
                Toast.makeText(requireContext(), "$letra salva para calibracao e treino", Toast.LENGTH_SHORT).show()
                indiceCalibracao = proximaLetraParaTreinar()
                updateCalibrationPanel()
            } else {
                updateCalibrationCaptureProgress()
                Toast.makeText(requireContext(), "Grave mais alguns frames da mao", Toast.LENGTH_SHORT).show()
            }
        }

        binding.btnCalibrationClose.setOnClickListener {
            closeCalibrationPanel()
        }

        binding.btnCalibrationExport.setOnClickListener {
            exportTrainingData()
        }

        binding.btnCalibrationNextWeak.setOnClickListener {
            goToNextWeakLetter()
        }

        binding.btnCalibrationResetTraining.setOnClickListener {
            confirmResetTraining()
        }

        binding.btnConfirmLetter.setOnClickListener {
            librasAnalyzer?.repetirLetraPendente()
        }

        binding.btnReplyAudio.setOnClickListener { startSpeechReply() }
        binding.btnReplyText.setOnClickListener { focusReplyText() }
        binding.btnReplySpeak.setOnClickListener { speakReply() }
        binding.btnReplyClose.setOnClickListener {
            saveReplyToScreen()
            closeReplyPanel()
        }
        binding.etReply.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_DONE) {
                saveReplyToScreen()
                closeReplyPanel()
                true
            } else {
                false
            }
        }

        binding.btnModeAlphabet.setOnClickListener {
            modoAtual = LibrasAnalyzer.Modo.ALFABETO
            librasAnalyzer?.setModo(modoAtual)
            updateModeButtons()
            updateCalibrationVisibility()
        }

        binding.btnModeBody.setOnClickListener {
            modoAtual = LibrasAnalyzer.Modo.CORPO
            librasAnalyzer?.setModo(modoAtual)
            updateModeButtons()
            closeCalibrationPanel()
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
            bindCamera()
        }

        // Toque: limpa a frase inteira (a lixeira zera tudo de uma vez).
        // Toque longo: apaga só a última letra/sinal, para quem quer corrigir.
        binding.btnDeleteLetter.setOnClickListener {
            librasAnalyzer?.limparFrase()
        }
        binding.btnDeleteLetter.setOnLongClickListener {
            librasAnalyzer?.apagarUltima()
            true
        }

        binding.btnHistory.setOnClickListener {
            HistoryBottomSheet.newInstance(historicoEntries)
                .also { sheet ->
                    sheet.onClearConversation = { limparConversa() }
                }
                .show(childFragmentManager, "history")
        }
    }

    private fun updateModeButtons() {
        val dark = 0xFF070707.toInt()
        val light = 0xFFF5F1E8.toInt()

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
        updateCalibrationVisibility()
    }

    private fun currentCalibrationLetter(): String = letrasCalibracao[indiceCalibracao]

    private fun openCalibrationPanel() {
        if (isLandscapeHudCompact()) {
            closeReplyPanel()
            binding.calibrationPanel.isVisible = false
            Toast.makeText(requireContext(), "Calibracao ocultada no HUD compacto", Toast.LENGTH_SHORT).show()
            return
        }
        if (modoAtual != LibrasAnalyzer.Modo.ALFABETO) {
            modoAtual = LibrasAnalyzer.Modo.ALFABETO
            librasAnalyzer?.setModo(modoAtual)
            updateModeButtons()
        }
        closeReplyPanel()
        binding.calibrationPanel.isVisible = true
        updateCalibrationPanel()
    }

    private fun closeCalibrationPanel() {
        librasAnalyzer?.cancelCalibration()
        binding.calibrationPanel.isVisible = false
    }

    private fun updateCalibrationPanel() {
        val letra = currentCalibrationLetter()
        binding.tvCalibrationLetter.text = letra
        val total = librasAnalyzer?.getCalibrationCount() ?: 0
        val amostrasLetra = librasAnalyzer?.getTrainingSampleCount(letra) ?: 0
        val amostrasTotal = librasAnalyzer?.getTrainingSampleCount() ?: 0
        val statusLetra = trainingLevel(amostrasLetra)
        binding.tvCalibrationStatus.text = if (letra in letrasDinamicasTreino) {
            "$total LETRAS CALIBRADAS | $letra DINAMICA"
        } else {
            "$total LETRAS CALIBRADAS"
        }
        binding.tvTrainingStatus.text =
            "$letra $amostrasLetra/${LibrasAnalyzer.TRAINING_STRONG_TARGET_SAMPLES} $statusLetra  |  TOTAL $amostrasTotal"
        binding.progressCalibration.progress = 0
        updateTrainingDashboard()
        librasAnalyzer?.cancelCalibration()
    }

    private fun updateCalibrationCaptureProgress() {
        if (!binding.calibrationPanel.isVisible) return
        val letra = currentCalibrationLetter()
        val frames = librasAnalyzer?.getCalibrationFrameCount() ?: 0
        val progress = (frames * 100 / LibrasAnalyzer.CALIBRATION_TARGET_FRAMES).coerceIn(0, 100)
        binding.progressCalibration.progress = progress
        val dinamica = letra in letrasDinamicasTreino
        binding.tvCalibrationStatus.text = when {
            frames == 0 && dinamica -> "TOQUE EM GRAVAR E MOVA $letra"
            frames == 0 -> "TOQUE EM GRAVAR E SEGURE $letra"
            frames < LibrasAnalyzer.CALIBRATION_MIN_FRAMES -> "GRAVANDO $letra  $frames/${LibrasAnalyzer.CALIBRATION_TARGET_FRAMES}"
            frames < LibrasAnalyzer.CALIBRATION_TARGET_FRAMES && dinamica -> "BOM, REPITA O MOVIMENTO  $frames/${LibrasAnalyzer.CALIBRATION_TARGET_FRAMES}"
            frames < LibrasAnalyzer.CALIBRATION_TARGET_FRAMES -> "BOM, CONTINUE FIRME  $frames/${LibrasAnalyzer.CALIBRATION_TARGET_FRAMES}"
            else -> "PRONTO PARA SALVAR $letra"
        }
    }

    private fun exportTrainingData() {
        val analyzer = librasAnalyzer
        val files = listOfNotNull(
            analyzer?.getTrainingDatasetPath()?.let { File(it) },
            analyzer?.getDynamicTrainingDatasetPath()?.let { File(it) }
        ).filter { file -> file.exists() && file.length() > 0L }

        if (files.isEmpty()) {
            Toast.makeText(
                requireContext(),
                "Calibre algumas letras antes de exportar",
                Toast.LENGTH_SHORT
            ).show()
            return
        }

        val authority = "${requireContext().packageName}.fileprovider"
        val uris = ArrayList<Uri>(
            files.map { file ->
                FileProvider.getUriForFile(requireContext(), authority, file)
            }
        )

        val intent = if (uris.size == 1) {
            Intent(Intent.ACTION_SEND).apply {
                type = "text/csv"
                putExtra(Intent.EXTRA_STREAM, uris.first())
            }
        } else {
            Intent(Intent.ACTION_SEND_MULTIPLE).apply {
                type = "text/csv"
                putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
            }
        }.apply {
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            putExtra(Intent.EXTRA_SUBJECT, "Dados de treino VisuAll")
            putExtra(
                Intent.EXTRA_TEXT,
                "Dados coletados no celular para melhorar o reconhecimento de Libras."
            )
        }

        startActivity(Intent.createChooser(intent, "Exportar dados VisuAll"))
    }

    private fun updateTrainingDashboard() {
        val progress = trainingProgress()
        binding.progressTrainingTotal.progress = progress.percent
        val faltam = if (progress.missingLetters.isEmpty()) {
            "TODAS PRONTAS"
        } else {
            progress.missingLetters.take(10).joinToString(" ")
        }
        binding.tvTrainingDashboard.text =
            "FORTE ${progress.percent}% | ${progress.trainedLetters}/${letrasCalibracao.size} LETRAS\nFRACAS: $faltam"
        binding.btnCalibrationNextWeak.isEnabled = progress.missingLetters.isNotEmpty()
        binding.btnCalibrationNextWeak.alpha = if (progress.missingLetters.isNotEmpty()) 1f else 0.45f
        binding.btnCalibrationExport.isEnabled = progress.totalSamples > 0
        binding.btnCalibrationExport.alpha = if (progress.totalSamples > 0) 1f else 0.45f
    }

    private fun trainingProgress(): TrainingProgress {
        val analyzer = librasAnalyzer
        val target = LibrasAnalyzer.TRAINING_STRONG_TARGET_SAMPLES
        var trainedLetters = 0
        var totalSamples = 0
        var cappedSamples = 0
        val missing = mutableListOf<Pair<String, Int>>()

        letrasCalibracao.forEach { letra ->
            val count = analyzer?.getTrainingSampleCount(letra) ?: 0
            totalSamples += count
            cappedSamples += count.coerceAtMost(target)
            if (count >= target) {
                trainedLetters++
            } else {
                missing += letra to count
            }
        }

        val percent = if (letrasCalibracao.isEmpty()) {
            0
        } else {
            (cappedSamples * 100 / (letrasCalibracao.size * target)).coerceIn(0, 100)
        }
        val sortedMissing = missing
            .sortedWith(compareBy<Pair<String, Int>> { it.second }.thenBy { it.first })
            .map { it.first }
        return TrainingProgress(percent, sortedMissing, trainedLetters, totalSamples)
    }

    private fun goToNextWeakLetter() {
        val next = indiceProximaLetraFraca(includeCurrent = true)
        if (next == null) {
            Toast.makeText(requireContext(), "Todas as letras ja estao fortes", Toast.LENGTH_SHORT).show()
            updateTrainingDashboard()
            return
        }
        indiceCalibracao = next
        updateCalibrationPanel()
    }

    private fun confirmResetTraining() {
        AlertDialog.Builder(requireContext())
            .setTitle("Zerar treino?")
            .setMessage("Isso apaga as amostras e calibracoes salvas neste celular.")
            .setPositiveButton("Zerar") { _, _ ->
                librasAnalyzer?.clearTrainingData()
                indiceCalibracao = letrasCalibracao.indexOf("E").coerceAtLeast(0)
                updateCalibrationPanel()
                Toast.makeText(requireContext(), "Treino zerado", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Cancelar", null)
            .show()
    }

    private fun proximaLetraParaTreinar(): Int {
        return indiceProximaLetraFraca(includeCurrent = false)
            ?: ((indiceCalibracao + 1) % letrasCalibracao.size)
    }

    private fun indiceProximaLetraFraca(includeCurrent: Boolean): Int? {
        if (letrasCalibracao.isEmpty()) return null
        val offset = if (includeCurrent) 0 else 1
        return letrasCalibracao.indices
            .map { (indiceCalibracao + offset + it) % letrasCalibracao.size }
            .firstOrNull { index ->
                val letra = letrasCalibracao[index]
                (librasAnalyzer?.getTrainingSampleCount(letra) ?: 0) <
                    LibrasAnalyzer.TRAINING_STRONG_TARGET_SAMPLES
            }
    }

    private fun trainingLevel(count: Int): String {
        return when {
            count >= LibrasAnalyzer.TRAINING_STRONG_TARGET_SAMPLES -> "FORTE"
            count >= LibrasAnalyzer.TRAINING_BASIC_TARGET_SAMPLES -> "BASICO"
            count > 0 -> "INICIO"
            else -> "SEM DADOS"
        }
    }

    private fun updateCalibrationVisibility() {
        val alfabeto = modoAtual == LibrasAnalyzer.Modo.ALFABETO
        binding.btnCalibrate.isVisible = alfabeto
        if (!alfabeto) {
            closeCalibrationPanel()
            hideSuggestions()
        }
    }

    private fun updateWordSuggestions(frase: String) {
        if (modoAtual != LibrasAnalyzer.Modo.ALFABETO) {
            hideSuggestions()
            return
        }

        val textoNormal = frase.lowercase(Locale("pt", "BR"))
        val prefixo = textoNormal.substringAfterLast(' ').trim()
        val prefixoBusca = normalizarBusca(prefixo)
        val ultimaCompleta = textoNormal.trim().split(" ").lastOrNull().orEmpty()
        val contextuais = if (frase.endsWith(" ")) {
            sugestoesContextuais[normalizarBusca(ultimaCompleta)].orEmpty()
        } else {
            emptyList()
        }

        val sugestoes = (contextuais + palavrasSugeridas)
            .distinct()
            .mapNotNull { palavra ->
                val busca = normalizarBusca(palavra)
                val score = when {
                    prefixoBusca.isBlank() && palavra in contextuais -> 100
                    prefixoBusca.length < 2 -> -1
                    busca.startsWith(prefixoBusca) -> 80 - (busca.length - prefixoBusca.length)
                    busca.contains(prefixoBusca) -> 35 - busca.indexOf(prefixoBusca)
                    else -> -1
                }
                if (score < 0 || busca == prefixoBusca) null else palavra to score
            }
            .sortedWith(compareByDescending<Pair<String, Int>> { it.second }.thenBy { it.first.length })
            .map { it.first }
            .take(3)

        val botoes = listOf(binding.btnSuggestion1, binding.btnSuggestion2, binding.btnSuggestion3)
        botoes.forEachIndexed { index, botao ->
            val palavra = sugestoes.getOrNull(index)
            botao.text = palavra.orEmpty()
            botao.tag = palavra
            botao.isVisible = palavra != null
        }
        binding.suggestionsRow.isVisible = sugestoes.isNotEmpty()
    }

    private fun normalizarBusca(texto: String): String {
        return Normalizer.normalize(texto.trim().lowercase(Locale("pt", "BR")), Normalizer.Form.NFD)
            .replace("\\p{Mn}+".toRegex(), "")
    }

    private fun hideSuggestions() {
        binding.suggestionsRow.isVisible = false
        listOf(binding.btnSuggestion1, binding.btnSuggestion2, binding.btnSuggestion3).forEach { botao ->
            botao.text = ""
            botao.tag = null
        }
    }

    private fun applySuggestionFrom(botao: TextView) {
        val palavra = botao.tag as? String ?: return
        librasAnalyzer?.aplicarSugestao(palavra)
        hideSuggestions()
    }

    private fun startSpeechReply() {
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

    private fun openReplyPanel(focus: Boolean = false) {
        if (isLandscapeHudCompact()) {
            binding.replyPanel.isVisible = false
            Toast.makeText(requireContext(), "Resposta ocultada no HUD compacto", Toast.LENGTH_SHORT).show()
            return
        }
        binding.replyPanel.isVisible = true
        val respostaAtual = binding.tvReply.text?.toString().orEmpty()
        if (binding.etReply.text.isNullOrBlank() && respostaAtual.isNotBlank()) {
            binding.etReply.setText(respostaAtual)
            binding.etReply.setSelection(respostaAtual.length)
        }
        if (focus) focusReplyText()
    }

    private fun focusReplyText() {
        binding.replyPanel.isVisible = true
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
        binding.etReply.clearFocus()
        hideKeyboard()
    }

    private fun saveReplyToScreen(): Boolean {
        val texto = binding.etReply.text?.toString().orEmpty().trim()
        if (texto.isBlank()) {
            clearReply()
            return false
        }
        binding.tvReply.text = texto
        binding.replyBubble.isVisible = true
        registrarMensagemResposta(texto)
        return true
    }

    private fun clearReply() {
        removerRespostaAtual()
        binding.etReply.setText("")
        binding.tvReply.text = ""
        binding.replyBubble.isVisible = false
        closeReplyPanel()
    }

    private fun speakReply(textoManual: String? = null) {
        if (textoManual != null) {
            binding.etReply.setText(textoManual)
            binding.etReply.setSelection(textoManual.length)
        }
        if (!saveReplyToScreen()) {
            Toast.makeText(requireContext(), "Digite ou grave uma resposta", Toast.LENGTH_SHORT).show()
            return
        }
        val texto = binding.tvReply.text.toString()
        tts?.speak(texto, TextToSpeech.QUEUE_FLUSH, null, "reply")
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
                binding.progressConfidence.progressTintList =
                    ColorStateList.valueOf(confidenceColor(confianca))
                binding.progressConfidence.progressBackgroundTintList =
                    ColorStateList.valueOf(0x33242424)
                // Bump: pequena pulsada quando a letra reconhecida muda.
                if (letra != ultimaLetraChip) {
                    binding.chipResult.animate().scaleX(1.14f).scaleY(1.14f).setDuration(90)
                        .withEndAction {
                            _binding?.chipResult?.animate()?.scaleX(1f)?.scaleY(1f)
                                ?.setDuration(90)?.start()
                        }.start()
                }
                ultimaLetraChip = letra
                updateCalibrationCaptureProgress()
            } else {
                ultimaLetraChip = ""
                binding.chipResult.visibility = View.INVISIBLE
                binding.progressConfidence.visibility = View.INVISIBLE
                binding.progressConfidence.progress = 0
                updateCalibrationCaptureProgress()
            }
        }
    }

    private fun confidenceColor(confianca: Float): Int {
        return when {
            confianca >= 0.92f -> 0xFFF5C842.toInt()
            confianca >= 0.84f -> 0xFFE8A020.toInt()
            else -> 0xFF8E6A26.toInt()
        }
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
                return@runOnUiThread
            }
            binding.tvFeedback.text = mensagem
            binding.tvFeedback.visibility = View.VISIBLE
            binding.tvFeedback.setTextColor(feedbackColor(nivel))
            if (binding.calibrationPanel.isVisible) {
                updateCalibrationCaptureProgress()
            }
        }
    }

    private fun feedbackColor(nivel: Int): Int {
        return when (nivel) {
            LibrasAnalyzer.FEEDBACK_BOM -> 0xFFF5C842.toInt()
            LibrasAnalyzer.FEEDBACK_ALERTA -> 0xFFFF8A80.toInt()
            else -> 0xFFF5F1E8.toInt()
        }
    }

    private fun onFraseAtualizada(frase: String) {
        activity?.runOnUiThread {
            if (_binding == null) return@runOnUiThread
            fraseBase = frase
            if (isLandscapeHudCompact()) {
                binding.phraseBubble.isVisible = false
                hideSuggestions()
                fraseAnterior = frase
                return@runOnUiThread
            }
            binding.tvPhrase.text = fraseExibida()
            binding.phraseBubble.isVisible = frase.isNotBlank()
            updateWordSuggestions(frase)

            // TTS: fala a última letra adicionada
            if (frase.length > fraseAnterior.length) {
                val trechoNovo = frase.removePrefix(fraseAnterior).trim()
                val fala = if (frase.endsWith(" ")) {
                    frase.trim().substringAfterLast(' ')
                } else {
                    trechoNovo.ifBlank { frase.lastOrNull()?.toString().orEmpty() }
                }
                if (fala.isNotBlank()) {
                    tts?.speak(fala, TextToSpeech.QUEUE_FLUSH, null, null)
                }
                vibrateConfirmation()
            }

            registrarMensagemLibras(frase)
            fraseAnterior = frase
        }
    }

    // Porta de montar_exibicao (app.py): acrescenta "?" ao texto mostrado
    // quando a sobrancelha está levantada, sem mexer na frase armazenada.
    private fun fraseExibida(): String {
        val base = fraseBase
        return if (interrogativoAtivo && base.isNotBlank() && !base.trimEnd().endsWith("?")) {
            base.trimEnd() + "?"
        } else {
            base
        }
    }

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

    /**
     * HISTÓRICO SIMPLIFICADO:
     * - Divide a frase em palavras por espaço
     * - A última palavra (incompleta ou não) é sempre mantida/atualizada
     *   na última entrada do histórico — nunca duplica
     * - Palavras anteriores (já com espaço depois) ficam fixas
     */
    private fun salvarPalavrasNoHistorico(fraseNova: String) {
        if (fraseNova.isEmpty()) return

        val tokens = fraseNova.trim().split(" ").filter { it.isNotEmpty() }
        if (tokens.isEmpty()) return

        val terminalComEspaco = fraseNova.endsWith(" ")

        if (terminalComEspaco) {
            // Todas as palavras estão completas — garante que cada uma existe
            // no histórico exatamente uma vez (sem duplicar)
            tokens.forEach { token ->
                val jaExiste = historicoEntries.any { it.word == token }
                if (!jaExiste) historicoEntries.add(HistoryEntry(token))
            }
        } else {
            // A última palavra ainda está sendo digitada
            val palavrasCompletas = tokens.dropLast(1)
            val ultimaPalavra = tokens.last()

            // Garante palavras completas no histórico
            palavrasCompletas.forEach { token ->
                val jaExiste = historicoEntries.any { it.word == token }
                if (!jaExiste) historicoEntries.add(HistoryEntry(token))
            }

            // Atualiza ou cria a entrada da última palavra (em construção)
            val ultimoIdx = historicoEntries.indexOfLast { entry ->
                ultimaPalavra.startsWith(entry.word) || entry.word.startsWith(ultimaPalavra)
            }
            if (ultimoIdx >= 0) {
                // Substitui no lugar — mantém o timestamp original
                historicoEntries[ultimoIdx] = HistoryEntry(ultimaPalavra, historicoEntries[ultimoIdx].timestamp)
            } else {
                historicoEntries.add(HistoryEntry(ultimaPalavra))
            }
        }
    }

    private fun registrarMensagemLibras(fraseNova: String) {
        val texto = fraseNova.trim()
        if (texto.isBlank()) {
            ultimaMensagemLibras = ""
            indiceLibrasAtual = -1
            return
        }
        if (texto == ultimaMensagemLibras) return

        if (indiceLibrasAtual in historicoEntries.indices &&
            historicoEntries[indiceLibrasAtual].source == "LIBRAS") {
            val anterior = historicoEntries[indiceLibrasAtual]
            historicoEntries[indiceLibrasAtual] = HistoryEntry(texto, anterior.timestamp, "LIBRAS")
        } else {
            historicoEntries.add(HistoryEntry(texto, source = "LIBRAS"))
            indiceLibrasAtual = historicoEntries.lastIndex
            indiceRespostaAtual = -1
        }
        ultimaMensagemLibras = texto
        limitarConversa()
        salvarConversa()
    }

    private fun registrarMensagemResposta(textoResposta: String) {
        val texto = textoResposta.trim()
        if (texto.isBlank()) return

        if (indiceRespostaAtual in historicoEntries.indices &&
            historicoEntries[indiceRespostaAtual].source == "RESPOSTA") {
            val anterior = historicoEntries[indiceRespostaAtual]
            historicoEntries[indiceRespostaAtual] = HistoryEntry(texto, anterior.timestamp, "RESPOSTA")
        } else {
            historicoEntries.add(HistoryEntry(texto, source = "RESPOSTA"))
            indiceRespostaAtual = historicoEntries.lastIndex
            indiceLibrasAtual = -1
        }
        limitarConversa()
        salvarConversa()
    }

    private fun removerRespostaAtual() {
        if (indiceRespostaAtual in historicoEntries.indices &&
            historicoEntries[indiceRespostaAtual].source == "RESPOSTA") {
            historicoEntries.removeAt(indiceRespostaAtual)
        }
        indiceRespostaAtual = -1
        salvarConversa()
    }

    private fun limitarConversa() {
        while (historicoEntries.size > 80) {
            historicoEntries.removeAt(0)
            if (indiceRespostaAtual > 0) {
                indiceRespostaAtual--
            } else if (indiceRespostaAtual == 0) {
                indiceRespostaAtual = -1
            }
            if (indiceLibrasAtual > 0) {
                indiceLibrasAtual--
            } else if (indiceLibrasAtual == 0) {
                indiceLibrasAtual = -1
            }
        }
    }

    private fun limparConversa() {
        historicoEntries.clear()
        ultimaMensagemLibras = ""
        indiceLibrasAtual = -1
        indiceRespostaAtual = -1
        salvarConversa()
    }

    private fun salvarConversa() {
        val json = JSONArray()
        historicoEntries.forEach { entry ->
            json.put(JSONObject().apply {
                put("word", entry.word)
                put("timestamp", entry.timestamp)
                put("source", entry.source)
            })
        }
        requireContext()
            .getSharedPreferences("visuall_conversa", Context.MODE_PRIVATE)
            .edit()
            .putString("entries", json.toString())
            .apply()
    }

    private fun carregarConversaSalva() {
        val raw = requireContext()
            .getSharedPreferences("visuall_conversa", Context.MODE_PRIVATE)
            .getString("entries", null)
            ?: return

        try {
            val json = JSONArray(raw)
            historicoEntries.clear()
            for (index in 0 until json.length()) {
                val item = json.optJSONObject(index) ?: continue
                val texto = item.optString("word").trim()
                if (texto.isBlank()) continue
                historicoEntries.add(
                    HistoryEntry(
                        word = texto,
                        timestamp = item.optLong("timestamp", System.currentTimeMillis()),
                        source = item.optString("source", "LIBRAS")
                    )
                )
            }
            indiceLibrasAtual = -1
            indiceRespostaAtual = -1
            ultimaMensagemLibras = ""
            limitarConversa()
        } catch (e: Exception) {
            historicoEntries.clear()
            salvarConversa()
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
        orientationListener?.disable()
        orientationListener = null
        landscapeHud = null
        try {
            librasAnalyzer?.close()
            librasAnalyzer = null
            cameraProvider?.unbindAll()
            cameraProvider = null
            if (!cameraExecutor.isShutdown) cameraExecutor.shutdown()
        } catch (e: Exception) {
            e.printStackTrace()
        }
        tts?.shutdown()
        _binding = null
    }
}
