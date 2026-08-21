package com.visuall.app.libras

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.content.res.ColorStateList
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.RecognizerIntent
import android.speech.tts.TextToSpeech
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.visuall.app.R
import com.visuall.app.databinding.FragmentLibrasBinding
import com.visuall.app.ui.ScanFrameView
import org.json.JSONArray
import org.json.JSONObject
import java.text.Normalizer
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class LibrasFragment : Fragment(), TextToSpeech.OnInitListener {

    private var _binding: FragmentLibrasBinding? = null
    private val binding get() = _binding!!

    private lateinit var cameraExecutor: ExecutorService
    private var librasAnalyzer: LibrasAnalyzer? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var tts: TextToSpeech? = null

    private var lensFacing = CameraSelector.LENS_FACING_FRONT

    private val historicoEntries = arrayListOf<HistoryEntry>()
    private var ultimaMensagemLibras = ""
    private var indiceLibrasAtual = -1
    private var indiceRespostaAtual = -1

    // Controle de histórico: rastreia a frase completa anterior
    private var fraseAnterior = ""
    private var linhasAtivas = true
    private var modoAtual = LibrasAnalyzer.Modo.ALFABETO
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
        startCamera()
        setupButtons()
        updateModeButtons()
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

        val preview = Preview.Builder().build().also { p ->
            p.setSurfaceProvider(binding.previewView.surfaceProvider)
        }

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

        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

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
                onFeedback = { mensagem, nivel -> onFeedback(mensagem, nivel) }
            ).also { it.setModo(modoAtual) }

            if (!isAdded || _binding == null || cameraExecutor.isShutdown) {
                newAnalyzer.close()
                return@execute
            }

            librasAnalyzer = newAnalyzer
            analysis.setAnalyzer(cameraExecutor, newAnalyzer)
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

        binding.btnSuggestion1.setOnClickListener { applySuggestionFrom(binding.btnSuggestion1) }
        binding.btnSuggestion2.setOnClickListener { applySuggestionFrom(binding.btnSuggestion2) }
        binding.btnSuggestion3.setOnClickListener { applySuggestionFrom(binding.btnSuggestion3) }

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
        }

        binding.btnModeBody.setOnClickListener {
            modoAtual = LibrasAnalyzer.Modo.CORPO
            librasAnalyzer?.setModo(modoAtual)
            updateModeButtons()
            Toast.makeText(requireContext(), "Modo corpo ativo", Toast.LENGTH_SHORT).show()
        }

        binding.btnLines.setOnClickListener {
            linhasAtivas = !linhasAtivas
            binding.scanFrame.isVisible = linhasAtivas
            binding.btnLines.text = if (linhasAtivas) "LINHAS ON" else "LINHAS OFF"
        }

        binding.btnFlip.setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT)
                CameraSelector.LENS_FACING_BACK
            else CameraSelector.LENS_FACING_FRONT
            bindCamera()
        }

        binding.btnDeleteLetter.setOnClickListener {
            librasAnalyzer?.apagarUltima()
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
            } else {
                binding.chipResult.visibility = View.INVISIBLE
                binding.progressConfidence.visibility = View.INVISIBLE
                binding.progressConfidence.progress = 0
            }
        }
    }

    private fun confidenceColor(confianca: Float): Int {
        val ctx = requireContext()
        return when {
            confianca >= 0.92f -> ContextCompat.getColor(ctx, R.color.gold_light)
            confianca >= 0.84f -> ContextCompat.getColor(ctx, R.color.gold_primary)
            else -> 0xFF8E6A26.toInt()
        }
    }

    private fun onFeedback(mensagem: String, nivel: Int) {
        activity?.runOnUiThread {
            if (_binding == null) return@runOnUiThread
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
            binding.tvPhrase.text = frase
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
            binding.chipResult.visibility = View.INVISIBLE
            binding.progressConfidence.visibility = View.INVISIBLE
            binding.progressConfidence.progress = 0
            binding.progressClear.isVisible = false
        }
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
        try {
            librasAnalyzer?.close()
            librasAnalyzer = null
            cameraProvider?.unbindAll()
            cameraProvider = null
            if (!cameraExecutor.isShutdown) cameraExecutor.shutdown()
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            if (isAdded && view != null) {
                val navController = findNavController()
                val voltouParaCamera = navController.popBackStack(R.id.nav_camera, false)
                if (!voltouParaCamera && navController.currentDestination?.id == R.id.nav_libras) {
                    navController.navigate(R.id.action_libras_to_camera)
                }
            }
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) tts?.language = Locale("pt", "BR")
    }

    override fun onDestroyView() {
        super.onDestroyView()
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
