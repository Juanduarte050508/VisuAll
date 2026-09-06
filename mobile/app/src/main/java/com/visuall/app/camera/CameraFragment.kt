package com.visuall.app.camera

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.BitmapFactory
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import android.view.Surface
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.SeekBar
import android.widget.Toast
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.Camera
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.core.ZoomState
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import com.visuall.app.ui.ProporcaoDaCamera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.MediaStoreOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewCompositionStrategy
import androidx.core.content.ContextCompat
import androidx.constraintlayout.widget.ConstraintSet
import androidx.fragment.app.Fragment
import androidx.lifecycle.LiveData
import androidx.lifecycle.Observer
import androidx.navigation.fragment.findNavController
import com.visuall.app.R
import com.visuall.app.databinding.FragmentCameraBinding
import com.visuall.app.ui.compose.CameraLandscapeHud
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.roundToInt

class CameraFragment : Fragment() {

    private var _binding: FragmentCameraBinding? = null
    private val binding get() = _binding!!

    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var preview: Preview? = null
    private var imageCapture: ImageCapture? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private var zoomStateLiveData: LiveData<ZoomState>? = null
    private var zoomStateObserver: Observer<ZoomState>? = null

    private var lensFacing  = CameraSelector.LENS_FACING_BACK
    private var flashMode   = ImageCapture.FLASH_MODE_AUTO
    private var isVideoMode = false
    private var isRecording = false
    private var landscapeHud: View? = null
    private var bindRetryPosted = false
    private var cameraStartRequested = false

    // Aspect ratio: true = 4:3 (padrão), false = 16:9
    private var is4to3 = true

    // Modo atual. PRO captura foto igual ao FOTO; o que ele acrescenta é o
    // controle de exposição, que fica escondido nos outros modos.
    private var modo = Modo.FOTO

    private var gradeLigada = false

    // Temporizador de disparo, em segundos. 0 = desligado.
    private var temporizador = 0
    private var contagemEmCurso = false

    // Zoom pedido pelo chip, em ratio da câmera atual. Guardado porque cada
    // rebind (troca de proporção, de modo, de lente) zera o zoom da sessão.
    private var zoomAtual = 1f

    // Id da câmera física ultra-wide, quando existir.
    //
    // Não dá pra chegar na ultra-wide por zoom: este aparelho reporta
    // zoomRatioRange [1.0, 8.0] em TODAS as câmeras, ou seja, o mínimo é 1x e
    // a grande-angular é outra câmera física. Por isso o chip 0.6x TROCA de
    // câmera, e só aparece se descobrirUltraWide() achar uma (ver lá o
    // critério). Em aparelho sem ultra-wide o chip simplesmente não existe,
    // em vez de virar um botão que não faz nada.
    private var idUltraWide: String? = null
    private var usandoUltraWide = false
    // Rótulo do chip da grande-angular, medido no aparelho (ver
    // descobrirUltraWide). O valor inicial só existe até a medição acontecer.
    private var rotuloUltraWide = "0.6x"

    private enum class Modo { FOTO, VIDEO, PRO }

    private val timerHandler = Handler(Looper.getMainLooper())
    private var timerSeconds = 0
    private val timerRunnable = object : Runnable {
        override fun run() {
            timerSeconds++
            _binding?.tvTimer?.text = "● %02d:%02d".format(timerSeconds / 60, timerSeconds % 60)
            timerHandler.postDelayed(this, 1000)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentCameraBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        // A escolha e da pessoa, nao da sessao: volta como estava.
        is4to3 = ProporcaoDaCamera.ehQuatroPorTres(requireContext())
        view.post {
            applyHudLayout()
            applyAspectRatioToPreview()
        }
        startCamera()
        setupButtons()
        loadLastThumbnail()
    }

    // ── Câmera ─────────────────────────────────────────────────────────────
    private fun hasCameraPermission(): Boolean {
        val context = context ?: return false
        return ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
    }

    private fun startCamera() {
        val context = context ?: return
        // Numa instalação nova, este fragment é criado DENTRO do setContentView
        // da MainActivity, ou seja, antes de o usuário sequer ver o diálogo de
        // permissão. Inicializar o CameraX aí faz o provider falhar e o
        // future.get() abaixo estourar na main thread -- que é exatamente o
        // motivo de o app fechar sozinho ao ser instalado do zero (no
        // dispositivo do dev não aparecia, porque a permissão já estava
        // concedida de execuções anteriores). Sem permissão a gente
        // simplesmente não começa: o onResume tenta de novo assim que o
        // usuário responde o diálogo.
        if (!hasCameraPermission()) return
        if (cameraStartRequested) return
        cameraStartRequested = true
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            // Zerar a flag ANTES de qualquer return: se sair sem zerar, toda
            // tentativa posterior de abrir a câmera fica bloqueada pra sempre.
            cameraStartRequested = false
            if (_binding == null || !isAdded) return@addListener
            cameraProvider = try {
                future.get()
            } catch (e: Exception) {
                Log.e("CameraFragment", "CameraX nao inicializou", e)
                Toast.makeText(context, "Nao consegui abrir a camera", Toast.LENGTH_SHORT).show()
                return@addListener
            }
            bindCamera()
        }, ContextCompat.getMainExecutor(context))
    }

    private fun bindCamera() {
        val currentBinding = _binding ?: return
        if (!isAdded || view == null) return
        // Mesmo motivo do startCamera: sem permissão o bind sempre falha e o
        // único efeito seria um Toast de erro a cada tentativa.
        if (!hasCameraPermission()) return
        val provider = cameraProvider ?: return
        if (!currentBinding.previewView.isAttachedToWindow ||
            currentBinding.previewView.width == 0 ||
            currentBinding.previewView.height == 0
        ) {
            if (!bindRetryPosted) {
                bindRetryPosted = true
                currentBinding.previewView.post {
                    bindRetryPosted = false
                    bindCamera()
                }
            }
            return
        }
        // A grande-angular é uma câmera física, não um zoom (ver
        // descobrirUltraWide), então ela entra aqui, na escolha do seletor.
        val idWide = idUltraWide
        val selector = if (usandoUltraWide && idWide != null &&
            lensFacing == CameraSelector.LENS_FACING_BACK
        ) {
            seletorDaCamera(idWide)
        } else {
            usandoUltraWide = false
            cameraSelectorForAvailableLens(lensFacing)
        }
        val targetRotation = currentTargetRotation()

        val aspectRatioStrategy = if (is4to3)
            AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
        else
            AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY

        val resolutionSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(aspectRatioStrategy)
            .build()

        val previewUseCase = Preview.Builder()
            .setResolutionSelector(resolutionSelector)
            .setTargetRotation(targetRotation)
            .build()
            .also { it.setSurfaceProvider(currentBinding.previewView.surfaceProvider) }
        preview = previewUseCase

        try {
            provider.unbindAll()
            camera = if (isVideoMode) {
                videoCapture = VideoCapture.withOutput(
                    Recorder.Builder()
                        .setQualitySelector(QualitySelector.from(Quality.HIGHEST))
                        .build()
                )
                provider.bindToLifecycle(viewLifecycleOwner, selector, preview, videoCapture!!)
            } else {
                imageCapture = ImageCapture.Builder()
                    .setFlashMode(flashMode)
                    .setResolutionSelector(resolutionSelector)
                    .setTargetRotation(targetRotation)
                    .build()
                provider.bindToLifecycle(viewLifecycleOwner, selector, preview, imageCapture!!)
            }
            camera?.cameraInfo?.let { info -> descobrirUltraWide(provider, info) }
            configurarZoomEExposicao()
            applyAspectRatioToPreview()
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(requireContext(), "Nao consegui abrir a camera", Toast.LENGTH_SHORT).show()
        }
    }


    // ── Aspect Ratio: altura derivada da largura (match_parent) ────────────
    private fun applyAspectRatioToPreview() {
        // Chegam aqui runnables postados na view e o retry de bind postado na
        // preview, que podem disparar depois do onDestroyView -- aí o
        // `binding` (que é `_binding!!`) estouraria.
        val currentBinding = _binding ?: return
        // Usa a resolução real entregue pela câmera (não uma suposição fixa
        // de "3:4"/"9:16"): o sensor raramente entrega exatamente esse
        // valor, e a caixa da preview precisa bater com o que é capturado
        // de fato — senão a moldura mostrada não corresponde à foto/vídeo
        // final e a imagem aparece cortada/deslocada.
        val resolution = preview?.resolutionInfo?.resolution
        val ratio = if (resolution != null && resolution.width > 0 && resolution.height > 0) {
            val longSide = maxOf(resolution.width, resolution.height)
            val shortSide = minOf(resolution.width, resolution.height)
            "H,$shortSide:$longSide"
        } else if (is4to3) {
            "H,3:4"
        } else {
            "H,9:16"
        }
        // ConstraintSet.clone() exige que TODO filho direto do root tenha id e
        // lança RuntimeException se algum não tiver. Foi assim que o app passou
        // a fechar sozinho na abertura: o novo desenho da câmera trouxe uma
        // View de contorno sem id (hoje @id/viewfinder_border), e este método
        // roda no onViewCreated. O id já está lá, mas ajustar a proporção da
        // preview é cosmético demais pra poder derrubar o app de novo se uma
        // view nova entrar sem id -- então a falha fica só no log.
        try {
            val cs = ConstraintSet()
            cs.clone(currentBinding.root)
            if (is4to3) {
                // 4:3 vive numa faixa entre a barra do topo e a fileira de
                // modos, com a proporção real da câmera.
                cs.setDimensionRatio(R.id.preview_view, ratio)
                cs.connect(R.id.preview_view, ConstraintSet.TOP,
                    R.id.top_chips, ConstraintSet.BOTTOM, dp(12))
                cs.connect(R.id.preview_view, ConstraintSet.BOTTOM,
                    R.id.modes_row, ConstraintSet.TOP, dp(12))
                // Sobreposições presas à própria faixa da preview.
                cs.connect(R.id.tv_mode_badge, ConstraintSet.TOP,
                    R.id.preview_view, ConstraintSet.TOP, dp(10))
                cs.connect(R.id.tv_timer, ConstraintSet.TOP,
                    R.id.preview_view, ConstraintSet.TOP, dp(10))
                cs.connect(R.id.zoom_bar, ConstraintSet.BOTTOM,
                    R.id.preview_view, ConstraintSet.BOTTOM, dp(14))
            } else {
                // 16:9 ocupa a tela inteira e os controles passam a flutuar
                // por cima da imagem. Sem razão de aspecto: quem decide o
                // enquadramento é o scaleType fillCenter da PreviewView, que
                // preenche a tela cortando o excedente.
                cs.setDimensionRatio(R.id.preview_view, null)
                cs.connect(R.id.preview_view, ConstraintSet.TOP,
                    ConstraintSet.PARENT_ID, ConstraintSet.TOP, 0)
                cs.connect(R.id.preview_view, ConstraintSet.BOTTOM,
                    ConstraintSet.PARENT_ID, ConstraintSet.BOTTOM, 0)
                // Em tela cheia "topo da preview" passa a ser o topo do
                // aparelho, onde já estão o flash e os chips -- então estas
                // três se ancoram nos controles, não na preview, pra não
                // colidirem com eles nem com a barra de status.
                cs.connect(R.id.tv_mode_badge, ConstraintSet.TOP,
                    R.id.top_chips, ConstraintSet.BOTTOM, dp(14))
                cs.connect(R.id.tv_timer, ConstraintSet.TOP,
                    R.id.top_chips, ConstraintSet.BOTTOM, dp(14))
                cs.connect(R.id.zoom_bar, ConstraintSet.BOTTOM,
                    R.id.modes_row, ConstraintSet.TOP, dp(14))
            }
            cs.applyTo(currentBinding.root)
        } catch (e: RuntimeException) {
            Log.e("CameraFragment", "Nao consegui ajustar a proporcao da preview " +
                "(alguma view do fragment_camera.xml esta sem android:id?)", e)
        }

        // A moldura delimita a área de captura dentro da faixa; em tela cheia
        // ela viraria um contorno dourado na borda do aparelho, sem função.
        currentBinding.viewfinderBorder.visibility = if (is4to3) View.VISIBLE else View.GONE
        val veu = if (is4to3) View.GONE else View.VISIBLE
        currentBinding.scrimTop.visibility = veu
        currentBinding.scrimBottom.visibility = veu

        currentBinding.btnAspectRatio.text = if (is4to3) "4:3" else "16:9"
    }

    private fun cameraSelectorForAvailableLens(preferredLensFacing: Int): CameraSelector {
        val provider = cameraProvider
        val preferred = CameraSelector.Builder()
            .requireLensFacing(preferredLensFacing)
            .build()
        if (provider == null || runCatching { provider.hasCamera(preferred) }.getOrDefault(false)) {
            return preferred
        }

        val fallbackLensFacing = if (preferredLensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
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

    private fun applyHudLayout() {
        // Mesma proteção do applyAspectRatioToPreview: daqui pra baixo tudo
        // usa `binding`, e este método é chamado de runnables postados.
        if (_binding == null) return
        if (!isLandscapeByBounds()) {
            applyPortraitHudLayout()
            return
        }

        setPortraitHudVisible(false)
        ensureLandscapeHud()
    }

    private fun applyPortraitHudLayout() {
        removeLandscapeHud()
        setPortraitHudVisible(true)
    }

    private fun dp(value: Int): Int {
        return (value * resources.displayMetrics.density).roundToInt()
    }

    private fun aspectRatioLabel(): String = if (is4to3) "4:3" else "16:9"

    private fun setPortraitHudVisible(visible: Boolean) {
        val visibility = if (visible) View.VISIBLE else View.GONE
        binding.btnFlash.visibility = visibility
        binding.topChips.visibility = visibility
        binding.btnVisuall.visibility = visibility
        binding.tvTimer.visibility = if (visible && isRecording) View.VISIBLE else View.GONE
        binding.zoomBar.visibility = visibility
        binding.modesRow.visibility = visibility
        binding.controlsRow.visibility = visibility
        // Estes dois têm dono próprio (o modo e o botão da grade); esconder
        // junto é seguro, mas mostrar de volta só quem estava ligado é que
        // mantém o HUD coerente ao voltar de paisagem.
        binding.proPanel.visibility = if (visible && modo == Modo.PRO) View.VISIBLE else View.GONE
        binding.gridOverlay.visibility = if (visible && gradeLigada) View.VISIBLE else View.GONE
    }

    private fun ensureLandscapeHud() {
        if (landscapeHud != null) return
        val hud = ComposeView(requireContext()).apply {
            id = R.id.hud_camera_land_root
            isClickable = false
            isFocusable = false
            setViewCompositionStrategy(ViewCompositionStrategy.DisposeOnDetachedFromWindow)
            setContent {
                var ratioLabel by remember { mutableStateOf(aspectRatioLabel()) }

                CameraLandscapeHud(
                    aspectRatioLabel = ratioLabel,
                    onLibrasClick = {
                        releaseCamera()
                        findNavController().navigate(R.id.action_camera_to_libras)
                    },
                    onFlipClick = {
                        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK)
                            CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
                        bindCamera()
                    },
                    onAspectRatioClick = {
                        is4to3 = !is4to3
                        ProporcaoDaCamera.guardar(requireContext(), is4to3)
                        ratioLabel = aspectRatioLabel()
                        bindCamera()
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
        _binding?.root?.post {
            applyHudLayout()
            applyAspectRatioToPreview()
        }
        if (_binding != null && camera == null) {
            if (cameraProvider != null) {
                bindCamera()
            } else {
                startCamera()
            }
        }
    }

    // Rotacao FIXA, de proposito. display.rotation e recalculado a cada giro
    // fisico do aparelho e, entregue ao CameraX, girava a imagem capturada
    // enquanto os guias na tela (que seguem a Activity, travada em retrato)
    // ficavam parados -- a area capturada deixava de bater com a moldura.
    // Como a janela e sempre retrato, ROTATION_0 e a resposta certa sempre.
    private fun currentTargetRotation(): Int = Surface.ROTATION_0

    // ── Zoom, ultra-wide e exposição ───────────────────────────────────────

    // Abertura angular de uma câmera, como largura-do-sensor / distância-focal.
    //
    // Este quociente é proporcional à tangente de metade do campo de visão, e
    // é ele -- não a distância focal sozinha -- que diz quanto a câmera "pega"
    // de cena. Comparar só focais leva a erro grosseiro entre câmeras de
    // sensores diferentes: no A52 a ultra-wide tem focal 3x menor que a
    // principal, mas o sensor dela também é bem menor, então o campo real é
    // pouco menos que o dobro, não o triplo.
    private fun aberturaDe(manager: CameraManager, id: String): Float? = try {
        val cc = manager.getCameraCharacteristics(id)
        val focal = cc.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.minOrNull()
        val largura = cc.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.width
        if (focal != null && largura != null && focal > 0f) largura / focal else null
    } catch (e: Exception) {
        null
    }

    // Procura uma grande-angular entre as câmeras traseiras.
    //
    // O critério é comparativo, não um número mágico: a ultra-wide é a câmera
    // traseira com o MAIOR campo de visão, e ele precisa ser pelo menos 20%
    // maior que o da principal (a que o CameraX acabou de abrir) pra não
    // confundir com variação entre sensores parecidos. Exijo também que o
    // CameraX enxergue aquela câmera -- se ela não estiver em
    // availableCameraInfos não há como vincular nela, e aí o chip não deve
    // existir em vez de existir sem funcionar.
    //
    // O rótulo sai da medição, não de um "0.6x" fixo: é a razão entre os
    // campos de visão, que é exatamente o "quantas vezes" da câmera.
    private fun descobrirUltraWide(provider: ProcessCameraProvider, atual: CameraInfo) {
        if (idUltraWide != null) return
        val manager = context?.getSystemService(Context.CAMERA_SERVICE) as? CameraManager ?: return
        try {
            val idPrincipal = Camera2CameraInfo.from(atual).cameraId
            val aberturaPrincipal = aberturaDe(manager, idPrincipal) ?: return

            val visiveisAoCameraX = provider.availableCameraInfos.mapNotNull { info ->
                runCatching { Camera2CameraInfo.from(info).cameraId }.getOrNull()
            }.toSet()

            val candidata = manager.cameraIdList
                .filter { id ->
                    id in visiveisAoCameraX &&
                        manager.getCameraCharacteristics(id)
                            .get(CameraCharacteristics.LENS_FACING) ==
                        CameraCharacteristics.LENS_FACING_BACK
                }
                .mapNotNull { id -> aberturaDe(manager, id)?.let { id to it } }
                .filter { (_, abertura) -> abertura > aberturaPrincipal * 1.2f }
                .maxByOrNull { (_, abertura) -> abertura }

            if (candidata == null) {
                Log.i("CameraFragment", "Sem ultra-wide utilizavel; chip de grande-angular fica oculto")
                return
            }
            idUltraWide = candidata.first
            rotuloUltraWide = "%.1fx".format(Locale.US, aberturaPrincipal / candidata.second)
            Log.i("CameraFragment", "Ultra-wide: camera ${candidata.first} ($rotuloUltraWide)")
        } catch (e: Exception) {
            Log.w("CameraFragment", "Falha ao procurar a camera grande-angular", e)
        }
    }

    // Seletor que prende o bind numa câmera física específica.
    private fun seletorDaCamera(id: String): CameraSelector =
        CameraSelector.Builder()
            .addCameraFilter { infos ->
                infos.filter { runCatching { Camera2CameraInfo.from(it).cameraId }.getOrNull() == id }
            }
            .build()

    // Chama a cada bind: mostra só os chips que esta câmera aguenta e reaplica
    // o zoom escolhido, já que cada rebind zera o zoom da sessão.
    private fun configurarZoomEExposicao() {
        val cam = camera ?: return
        val b = _binding ?: return

        val maxZoom = cam.cameraInfo.zoomState.value?.maxZoomRatio ?: 1f
        b.chipZoomWide.text = rotuloUltraWide
        // A grande-angular é uma câmera traseira: na frontal o chip não teria
        // o que fazer, então some em vez de ficar lá sem efeito.
        val temWide = idUltraWide != null && lensFacing == CameraSelector.LENS_FACING_BACK
        b.chipZoomWide.visibility = if (temWide) View.VISIBLE else View.GONE
        b.chipZoom2.visibility = if (maxZoom >= 2f) View.VISIBLE else View.GONE
        b.chipZoom5.visibility = if (maxZoom >= 5f) View.VISIBLE else View.GONE

        cam.cameraControl.setZoomRatio(zoomAtual.coerceIn(1f, maxZoom))
        atualizarChipsZoom()
        configurarExposicao(cam)
    }

    private fun atualizarChipsZoom() {
        val b = _binding ?: return
        val selecionado = listOf(
            b.chipZoomWide to usandoUltraWide,
            b.chipZoom1 to (!usandoUltraWide && zoomAtual < 1.5f),
            b.chipZoom2 to (!usandoUltraWide && zoomAtual >= 1.5f && zoomAtual < 3.5f),
            b.chipZoom5 to (!usandoUltraWide && zoomAtual >= 3.5f)
        )
        val ouro = ContextCompat.getColor(requireContext(), R.color.gold_primary)
        val claro = ContextCompat.getColor(requireContext(), R.color.text_primary)
        selecionado.forEach { (chip, ativo) ->
            chip.setBackgroundResource(if (ativo) R.drawable.bg_zoom_chip_on else 0)
            chip.setTextColor(if (ativo) ouro else claro)
        }
    }

    // Troca o zoom pedido por um chip. O 0.6x é o único que não é zoom: ele
    // troca de câmera física, então precisa de rebind.
    private fun aplicarZoom(ratio: Float, ultraWide: Boolean) {
        if (ultraWide && idUltraWide == null) return
        val precisaRebind = ultraWide != usandoUltraWide
        usandoUltraWide = ultraWide
        zoomAtual = if (ultraWide) 1f else ratio
        if (precisaRebind) {
            bindCamera()
        } else {
            val cam = camera ?: return
            val maxZoom = cam.cameraInfo.zoomState.value?.maxZoomRatio ?: 1f
            cam.cameraControl.setZoomRatio(zoomAtual.coerceIn(1f, maxZoom))
            atualizarChipsZoom()
        }
    }

    // Compensação de exposição: é o controle do modo PRO.
    //
    // Nem toda câmera suporta -- e quando não suporta, o modo PRO inteiro sai
    // da fileira de modos em vez de virar um botão que não faz nada.
    private fun configurarExposicao(cam: Camera) {
        val b = _binding ?: return
        val estado = cam.cameraInfo.exposureState
        val suportado = estado.isExposureCompensationSupported
        b.modePro.visibility = if (suportado) View.VISIBLE else View.GONE

        if (!suportado) {
            if (modo == Modo.PRO) aplicarModo(Modo.FOTO)
            return
        }

        val faixa = estado.exposureCompensationRange
        b.seekEv.max = faixa.upper - faixa.lower
        b.seekEv.progress = estado.exposureCompensationIndex - faixa.lower
        atualizarRotuloEv(cam, estado.exposureCompensationIndex)

        b.seekEv.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progresso: Int, doUsuario: Boolean) {
                if (!doUsuario) return
                val indice = faixa.lower + progresso
                camera?.let {
                    it.cameraControl.setExposureCompensationIndex(indice)
                    atualizarRotuloEv(it, indice)
                }
            }
            override fun onStartTrackingTouch(sb: SeekBar?) = Unit
            override fun onStopTrackingTouch(sb: SeekBar?) = Unit
        })
    }

    // O índice de exposição é um passo, não um valor em EV: o valor real é
    // índice * passo, e o passo é uma fração que varia por aparelho.
    private fun atualizarRotuloEv(cam: Camera, indice: Int) {
        val passo = cam.cameraInfo.exposureState.exposureCompensationStep
        val ev = indice * passo.numerator.toFloat() / passo.denominator.toFloat()
        _binding?.tvEvLabel?.text = "EV %+.1f".format(Locale.getDefault(), ev)
    }

    // ── Botões ─────────────────────────────────────────────────────────────

    private fun setupButtons() {
        binding.modeFoto.setOnClickListener  { aplicarModo(Modo.FOTO) }
        binding.modeVideo.setOnClickListener { aplicarModo(Modo.VIDEO) }
        binding.modePro.setOnClickListener   { aplicarModo(Modo.PRO) }

        binding.btnVisuall.setOnClickListener {
            releaseCamera()
            findNavController().navigate(R.id.action_camera_to_libras)
        }

        // Botão aspect ratio: alterna entre 4:3 e 16:9.
        // Guardado em disco porque o modo Libras usa a MESMA escolha — ver
        // ProporcaoDaCamera e applyPreviewAspectRatio no LibrasFragment.
        binding.btnAspectRatio.setOnClickListener {
            is4to3 = !is4to3
            ProporcaoDaCamera.guardar(requireContext(), is4to3)
            bindCamera()
        }

        binding.btnGrid.setOnClickListener { alternarGrade() }
        binding.btnTimer.setOnClickListener { alternarTemporizador() }

        binding.chipZoomWide.setOnClickListener { aplicarZoom(1f, ultraWide = true) }
        binding.chipZoom1.setOnClickListener    { aplicarZoom(1f, ultraWide = false) }
        binding.chipZoom2.setOnClickListener    { aplicarZoom(2f, ultraWide = false) }
        binding.chipZoom5.setOnClickListener    { aplicarZoom(5f, ultraWide = false) }

        binding.btnShutter.setOnTouchListener { v, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    binding.btnShutter.setImageResource(R.drawable.ic_shutter_pressed)
                    true
                }
                MotionEvent.ACTION_UP -> {
                    binding.btnShutter.setImageResource(
                        if (isVideoMode) R.drawable.ic_shutter_video else R.drawable.ic_shutter)
                    dispararComTemporizador()
                    v.performClick()
                    true
                }
                MotionEvent.ACTION_CANCEL -> {
                    binding.btnShutter.setImageResource(
                        if (isVideoMode) R.drawable.ic_shutter_video else R.drawable.ic_shutter)
                    true
                }
                else -> false
            }
        }

        binding.btnFlip.setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK)
                CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
            bindCamera()
        }

        binding.btnFlash.setOnClickListener {
            flashMode = when (flashMode) {
                ImageCapture.FLASH_MODE_AUTO -> ImageCapture.FLASH_MODE_ON
                ImageCapture.FLASH_MODE_ON   -> ImageCapture.FLASH_MODE_OFF
                else                          -> ImageCapture.FLASH_MODE_AUTO
            }
            imageCapture?.flashMode = flashMode
            atualizarIconeFlash()
        }
        atualizarIconeFlash()

        binding.btnGallery.setOnClickListener { openGallery() }
    }

    // ── Modo, grade e temporizador ─────────────────────────────────────────

    private fun aplicarModo(novo: Modo) {
        // Trocar de modo no meio de uma gravação deixaria o arquivo pela
        // metade e o botão fora de sincronia com o estado real.
        if (isRecording) return
        cancelarContagem()

        modo = novo
        isVideoMode = novo == Modo.VIDEO

        val ouro = ContextCompat.getColor(requireContext(), R.color.gold_primary)
        val apagado = ContextCompat.getColor(requireContext(), R.color.text_dim)
        listOf(
            binding.modeFoto to (novo == Modo.FOTO),
            binding.modeVideo to (novo == Modo.VIDEO),
            binding.modePro to (novo == Modo.PRO)
        ).forEach { (view, ativo) ->
            view.setTextColor(if (ativo) ouro else apagado)
            view.setTypeface(null, if (ativo) android.graphics.Typeface.BOLD else android.graphics.Typeface.NORMAL)
        }

        binding.tvModeBadge.text = when (novo) {
            Modo.FOTO -> "FOTO"
            Modo.VIDEO -> "VÍDEO"
            Modo.PRO -> "PRO"
        }
        binding.proPanel.visibility = if (novo == Modo.PRO) View.VISIBLE else View.GONE
        binding.btnShutter.setImageResource(
            if (isVideoMode) R.drawable.ic_shutter_video else R.drawable.ic_shutter)

        // O temporizador é de foto: em vídeo ele confundiria com o tempo de
        // gravação, então volta pra OFF ao entrar no modo vídeo.
        if (isVideoMode && temporizador != 0) {
            temporizador = 0
            binding.btnTimer.text = "OFF"
        }
        binding.btnTimer.visibility = if (isVideoMode) View.GONE else View.VISIBLE

        // Vídeo e foto usam use cases diferentes, então precisam de rebind.
        bindCamera()
    }

    // Ícone e leitura do flash sempre saem do MESMO lugar, junto.
    //
    // Antes o ícone era trocado só dentro do clique, e AUTO e ON usavam o
    // mesmo raio mudando de cor -- então dava pra estar em AUTO achando que
    // estava desligado, e o flash disparar no escuro. Como este é um app de
    // acessibilidade, a descrição também muda: quem usa leitor de tela precisa
    // do estado, não de um "Flash" genérico.
    private fun atualizarIconeFlash() {
        val b = _binding ?: return
        val (icone, descricao) = when (flashMode) {
            ImageCapture.FLASH_MODE_ON -> R.drawable.ic_flash_on to "Flash ligado"
            ImageCapture.FLASH_MODE_OFF -> R.drawable.ic_flash_off to "Flash desligado"
            else -> R.drawable.ic_flash_auto to "Flash automático"
        }
        b.btnFlash.setImageResource(icone)
        b.btnFlash.contentDescription = descricao
    }

    private fun alternarGrade() {
        gradeLigada = !gradeLigada
        binding.gridOverlay.visibility = if (gradeLigada) View.VISIBLE else View.GONE
        binding.btnGrid.setBackgroundResource(
            if (gradeLigada) R.drawable.bg_topbar_chip_on else R.drawable.bg_topbar_chip)
        binding.btnGrid.imageTintList = ContextCompat.getColorStateList(
            requireContext(),
            if (gradeLigada) R.color.text_on_gold else R.color.text_primary)
    }

    private fun alternarTemporizador() {
        temporizador = when (temporizador) {
            0 -> 3
            3 -> 10
            else -> 0
        }
        binding.btnTimer.text = if (temporizador == 0) "OFF" else "${temporizador}s"
        binding.btnTimer.setBackgroundResource(
            if (temporizador == 0) R.drawable.bg_topbar_chip else R.drawable.bg_topbar_chip_on)
        binding.btnTimer.setTextColor(ContextCompat.getColor(
            requireContext(),
            if (temporizador == 0) R.color.text_primary else R.color.text_on_gold))
    }

    // Disparo único do obturador, respeitando o temporizador.
    //
    // Vídeo ignora o temporizador de propósito (ver aplicarModo), e um toque
    // durante a contagem CANCELA em vez de enfileirar um segundo disparo.
    private fun dispararComTemporizador() {
        if (isVideoMode) {
            toggleRecording()
            return
        }
        if (contagemEmCurso) {
            cancelarContagem()
            return
        }
        if (temporizador == 0) {
            takePhoto()
            return
        }
        contagemEmCurso = true
        binding.tvCountdown.visibility = View.VISIBLE
        contarRegressivo(temporizador)
    }

    private fun contarRegressivo(restante: Int) {
        val b = _binding ?: return
        if (restante <= 0) {
            cancelarContagem()
            takePhoto()
            return
        }
        b.tvCountdown.text = restante.toString()
        timerHandler.postDelayed({ contarRegressivo(restante - 1) }, 1000L)
    }

    private fun cancelarContagem() {
        if (!contagemEmCurso) return
        contagemEmCurso = false
        timerHandler.removeCallbacksAndMessages(null)
        _binding?.tvCountdown?.visibility = View.GONE
        // removeCallbacksAndMessages derruba também o cronômetro de gravação,
        // que compartilha este handler -- então ele volta se estiver gravando.
        if (isRecording) timerHandler.post(timerRunnable)
    }

    // ── Foto ───────────────────────────────────────────────────────────────
    private fun takePhoto() {
        val capture = imageCapture ?: return
        val name    = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val cv      = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, "IMG_$name")
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                put(MediaStore.Images.Media.RELATIVE_PATH, "DCIM/VisuAll")
        }
        capture.takePicture(
            ImageCapture.OutputFileOptions.Builder(
                requireContext().contentResolver,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv).build(),
            ContextCompat.getMainExecutor(requireContext()),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(out: ImageCapture.OutputFileResults) {
                    Toast.makeText(requireContext(), "Foto salva!", Toast.LENGTH_SHORT).show()
                    out.savedUri?.let { uri -> updateThumbnail(uri) }
                }
                override fun onError(e: ImageCaptureException) {
                    Toast.makeText(requireContext(), "Erro: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        )
    }

    // ── Vídeo ──────────────────────────────────────────────────────────────
    private fun toggleRecording() {
        if (isRecording) {
            recording?.stop(); recording = null; isRecording = false
            timerHandler.removeCallbacks(timerRunnable)
            _binding?.tvTimer?.visibility = View.GONE
            _binding?.btnShutter?.setImageResource(R.drawable.ic_shutter_video)
        } else {
            val vc   = videoCapture ?: return
            val name = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val cv   = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, "VID_$name")
                put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                    put(MediaStore.Video.Media.RELATIVE_PATH, "DCIM/VisuAll")
            }
            recording = vc.output.prepareRecording(requireContext(),
                MediaStoreOutputOptions.Builder(requireContext().contentResolver,
                    MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
                    .setContentValues(cv).build())
                .start(ContextCompat.getMainExecutor(requireContext())) { event ->
                    if (event is VideoRecordEvent.Finalize && !event.hasError()) {
                        activity?.runOnUiThread {
                            Toast.makeText(requireContext(), "Vídeo salvo!", Toast.LENGTH_SHORT).show()
                            loadLastThumbnail()
                        }
                    }
                }
            isRecording = true; timerSeconds = 0
            _binding?.tvTimer?.visibility = View.VISIBLE
            _binding?.btnShutter?.setImageResource(R.drawable.ic_shutter_stop)
            timerHandler.post(timerRunnable)
        }
    }

    // ── Thumbnail ──────────────────────────────────────────────────────────
    private fun updateThumbnail(uri: Uri) {
        try {
            val bmp = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                requireContext().contentResolver
                    .loadThumbnail(uri, android.util.Size(200, 200), null)
            } else {
                val stream = requireContext().contentResolver.openInputStream(uri)
                BitmapFactory.decodeStream(stream)
            }
            _binding?.btnGallery?.setImageBitmap(bmp)
        } catch (e: Exception) { e.printStackTrace() }
    }

    // A miniatura da galeria é enfeite: se não der pra ler, o botão só fica
    // com o ícone padrão. O que NÃO pode acontecer é derrubar o app -- e era
    // isso que acontecia, porque este método roda no onViewCreated (antes de
    // qualquer permissão de mídia ter sido concedida; o app nem chega a pedir
    // READ_MEDIA_IMAGES) e a query no MediaStore sem permissão lança
    // SecurityException em Android 9 e em vários aparelhos mais novos.
    private fun loadLastThumbnail() {
        val context = context ?: return
        try {
            context.contentResolver.query(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                arrayOf(MediaStore.Images.Media._ID),
                null, null,
                "${MediaStore.Images.Media.DATE_ADDED} DESC"
            )?.use { c ->
                if (c.moveToFirst()) {
                    val id  = c.getLong(c.getColumnIndexOrThrow(MediaStore.Images.Media._ID))
                    val uri = Uri.withAppendedPath(
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id.toString())
                    updateThumbnail(uri)
                }
            }
        } catch (e: Exception) {
            Log.w("CameraFragment", "Nao consegui ler a ultima foto da galeria", e)
        }
    }

    // ── Galeria ────────────────────────────────────────────────────────────
    private fun openGallery() {
        try {
            startActivity(Intent(Intent.ACTION_VIEW).apply {
                type  = "image/*"
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            })
        } catch (e: Exception) {
            startActivity(Intent(Intent.ACTION_PICK,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI))
        }
    }

    private fun releaseCamera() {
        try {
            recording?.stop()
            recording = null
            isRecording = false
            cameraProvider?.unbindAll()
            camera = null
            imageCapture = null
            videoCapture = null
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    override fun onPause() {
        super.onPause()
        releaseCamera()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        landscapeHud = null
        timerHandler.removeCallbacks(timerRunnable)
        releaseCamera()
        zoomStateObserver?.let { observer ->
            zoomStateLiveData?.removeObserver(observer)
        }
        zoomStateObserver = null
        zoomStateLiveData = null
        cameraExecutor.shutdown()
        _binding = null
    }
}

