package com.visuall.app.camera

import android.Manifest
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.BitmapFactory
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
import android.widget.Toast
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.core.ZoomState
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.MediaStoreOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.core.content.ContextCompat
import androidx.constraintlayout.widget.ConstraintSet
import androidx.fragment.app.Fragment
import androidx.lifecycle.LiveData
import androidx.lifecycle.Observer
import androidx.navigation.fragment.findNavController
import com.visuall.app.R
import com.visuall.app.databinding.FragmentCameraBinding
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
    private var ignoreZoomCallback = false

    private var lensFacing  = CameraSelector.LENS_FACING_BACK
    private var flashMode   = ImageCapture.FLASH_MODE_AUTO
    private var isVideoMode = false
    private var isRecording = false
    private var landscapeHud: View? = null
    private var bindRetryPosted = false
    private var cameraStartRequested = false

    // Aspect ratio: true = 4:3 (padrão), false = 16:9
    private var is4to3 = true

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
        val selector = cameraSelectorForAvailableLens(lensFacing)
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
            setupZoom()
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
            cs.setDimensionRatio(R.id.preview_view, ratio)
            cs.applyTo(currentBinding.root)
        } catch (e: RuntimeException) {
            Log.e("CameraFragment", "Nao consegui ajustar a proporcao da preview " +
                "(alguma view do fragment_camera.xml esta sem android:id?)", e)
        }

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

    private fun setPortraitHudVisible(visible: Boolean) {
        val visibility = if (visible) View.VISIBLE else View.GONE
        binding.btnFlash.visibility = visibility
        binding.btnAspectRatio.visibility = visibility
        binding.tvTimer.visibility = if (visible && isRecording) View.VISIBLE else View.GONE
        binding.zoomContainer.visibility = visibility
        binding.modesRow.visibility = visibility
        binding.controlsRow.visibility = visibility
    }

    private fun ensureLandscapeHud() {
        if (landscapeHud != null) return
        val hud = layoutInflater.inflate(R.layout.hud_camera_land, binding.root, false)
        hud.id = R.id.hud_camera_land_root
        binding.root.addView(
            hud,
            ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        )
        hud.findViewById<View>(R.id.btn_libras_land).setOnClickListener {
            releaseCamera()
            findNavController().navigate(R.id.action_camera_to_libras)
        }
        hud.findViewById<View>(R.id.btn_flip_land).setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK)
                CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
            bindCamera()
        }
        hud.findViewById<android.widget.TextView>(R.id.btn_aspect_ratio_land).apply {
            text = if (is4to3) "4:3" else "16:9"
            setOnClickListener {
                is4to3 = !is4to3
                text = if (is4to3) "4:3" else "16:9"
                bindCamera()
            }
        }
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

    // ── Zoom ───────────────────────────────────────────────────────────────
    private fun setupZoom() {
        val cam = camera ?: return

        zoomStateObserver?.let { observer ->
            zoomStateLiveData?.removeObserver(observer)
        }

        val liveData = cam.cameraInfo.zoomState
        val observer = Observer<ZoomState> { zoomState ->
            if (_binding == null) return@Observer
            binding.tvZoomMin.text = formatZoomLabel(zoomState.minZoomRatio)
            binding.tvZoomMax.text = formatZoomLabel(zoomState.maxZoomRatio)

            if (!binding.seekZoom.isPressed) {
                val progress = (zoomState.linearZoom * 100f).roundToInt().coerceIn(0, 100)
                if (binding.seekZoom.progress != progress) {
                    ignoreZoomCallback = true
                    binding.seekZoom.progress = progress
                    ignoreZoomCallback = false
                }
            }
        }
        zoomStateLiveData = liveData
        zoomStateObserver = observer
        liveData.observe(viewLifecycleOwner, observer)

        _binding?.seekZoom?.setOnProgressChangedListener { progress, _ ->
            if (ignoreZoomCallback) return@setOnProgressChangedListener
            cam.cameraControl.setLinearZoom((progress / 100f).coerceIn(0f, 1f))
        }
    }

    // ── Botões ─────────────────────────────────────────────────────────────
    private fun formatZoomLabel(ratio: Float): String {
        val inteiro = ratio.roundToInt()
        return if (abs(ratio - inteiro) < 0.05f) {
            "${inteiro}x"
        } else {
            "%.1fx".format(Locale.US, ratio)
        }
    }

    private fun setupButtons() {
        binding.modeRetrato.setOnClickListener { selectMode("RETRATO") }
        binding.modeFoto.setOnClickListener    { selectMode("FOTO") }
        binding.modeVideo.setOnClickListener   { selectMode("VIDEO") }
        binding.btnLibrasMode.setOnClickListener {
            releaseCamera()
            findNavController().navigate(R.id.action_camera_to_libras)
        }

        // Botão aspect ratio: alterna entre 4:3 e 16:9
        binding.btnAspectRatio.setOnClickListener {
            is4to3 = !is4to3
            bindCamera()
        }

        binding.btnShutter.setOnTouchListener { v, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    binding.btnShutter.setImageResource(R.drawable.ic_shutter_pressed)
                    true
                }
                MotionEvent.ACTION_UP -> {
                    binding.btnShutter.setImageResource(
                        if (isVideoMode) R.drawable.ic_shutter_video else R.drawable.ic_shutter)
                    if (isVideoMode) toggleRecording() else takePhoto()
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
            binding.btnFlash.setImageResource(when (flashMode) {
                ImageCapture.FLASH_MODE_ON  -> R.drawable.ic_flash_on
                ImageCapture.FLASH_MODE_OFF -> R.drawable.ic_flash_off
                else                         -> R.drawable.ic_flash_auto
            })
            imageCapture?.flashMode = flashMode
        }

        binding.btnGallery.setOnClickListener { openGallery() }
    }

    private fun selectMode(mode: String) {
        val orange = ContextCompat.getColor(requireContext(), R.color.orange_accent)
        val dim    = ContextCompat.getColor(requireContext(), R.color.text_dim)
        binding.modeRetrato.setTextColor(if (mode == "RETRATO") orange else dim)
        binding.modeFoto.setTextColor(   if (mode == "FOTO")    orange else dim)
        binding.modeVideo.setTextColor(  if (mode == "VIDEO")   orange else dim)
        isVideoMode = mode == "VIDEO"
        binding.btnShutter.setImageResource(
            if (isVideoMode) R.drawable.ic_shutter_video else R.drawable.ic_shutter)
        bindCamera()
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

