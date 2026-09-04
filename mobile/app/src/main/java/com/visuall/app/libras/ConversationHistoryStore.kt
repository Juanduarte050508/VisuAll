package com.visuall.app.libras

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

// Persistência e montagem do histórico de conversa (mensagens Libras +
// respostas por voz/texto) — extraído do LibrasFragment porque essa lógica
// é puro estado + SharedPreferences, sem nenhuma dependência de view.
internal class ConversationHistoryStore(private val context: Context) {

    private val _entries = arrayListOf<HistoryEntry>()
    val entries: List<HistoryEntry> get() = _entries

    private var ultimaMensagemLibras = ""
    private var indiceLibrasAtual = -1
    private var indiceRespostaAtual = -1

    fun load() {
        val raw = prefs().getString("entries", null) ?: return

        try {
            val json = JSONArray(raw)
            _entries.clear()
            for (index in 0 until json.length()) {
                val item = json.optJSONObject(index) ?: continue
                val texto = item.optString("word").trim()
                if (texto.isBlank()) continue
                _entries.add(
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
            limitar()
        } catch (e: Exception) {
            _entries.clear()
            salvar()
        }
    }

    fun registrarMensagemLibras(fraseNova: String) {
        val texto = fraseNova.trim()
        if (texto.isBlank()) {
            ultimaMensagemLibras = ""
            indiceLibrasAtual = -1
            return
        }
        if (texto == ultimaMensagemLibras) return

        if (indiceLibrasAtual in _entries.indices &&
            _entries[indiceLibrasAtual].source == "LIBRAS") {
            val anterior = _entries[indiceLibrasAtual]
            _entries[indiceLibrasAtual] = HistoryEntry(texto, anterior.timestamp, "LIBRAS")
        } else {
            _entries.add(HistoryEntry(texto, source = "LIBRAS"))
            indiceLibrasAtual = _entries.lastIndex
            indiceRespostaAtual = -1
        }
        ultimaMensagemLibras = texto
        limitar()
        salvar()
    }

    fun registrarMensagemResposta(textoResposta: String) {
        val texto = textoResposta.trim()
        if (texto.isBlank()) return

        if (indiceRespostaAtual in _entries.indices &&
            _entries[indiceRespostaAtual].source == "RESPOSTA") {
            val anterior = _entries[indiceRespostaAtual]
            _entries[indiceRespostaAtual] = HistoryEntry(texto, anterior.timestamp, "RESPOSTA")
        } else {
            _entries.add(HistoryEntry(texto, source = "RESPOSTA"))
            indiceRespostaAtual = _entries.lastIndex
            indiceLibrasAtual = -1
        }
        limitar()
        salvar()
    }

    fun removerRespostaAtual() {
        if (indiceRespostaAtual in _entries.indices &&
            _entries[indiceRespostaAtual].source == "RESPOSTA") {
            _entries.removeAt(indiceRespostaAtual)
        }
        indiceRespostaAtual = -1
        salvar()
    }

    fun limpar() {
        _entries.clear()
        ultimaMensagemLibras = ""
        indiceLibrasAtual = -1
        indiceRespostaAtual = -1
        salvar()
    }

    /**
     * As mensagens de uma origem so ("LIBRAS" ou "RESPOSTA").
     *
     * As duas metades da conversa sao consultadas em momentos diferentes: o que
     * a pessoa sinalizou se confere no modo de leitura, e o que foi respondido
     * se confere de dentro da propria tela de resposta. Misturar as duas numa
     * lista so obrigava a garimpar. O campo `source` ja existia em cada entrada,
     * entao separar e filtrar -- nao ha dado novo a guardar.
     */
    fun entriesDe(source: String): List<HistoryEntry> = _entries.filter { it.source == source }

    /** Limpa so uma das metades, preservando a outra. */
    fun limpar(source: String) {
        _entries.removeAll { it.source == source }
        if (source == "LIBRAS") {
            ultimaMensagemLibras = ""
            indiceLibrasAtual = -1
        } else {
            indiceRespostaAtual = -1
        }
        // Os indices apontam pra posicoes que mudaram com a remocao; recalcular
        // um a um seria fragil, e perder o "continuar editando a ultima" custa
        // menos que apagar a entrada errada depois.
        indiceLibrasAtual = -1
        indiceRespostaAtual = -1
        salvar()
    }

    private fun limitar() {
        while (_entries.size > 80) {
            _entries.removeAt(0)
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

    private fun salvar() {
        val json = JSONArray()
        _entries.forEach { entry ->
            json.put(JSONObject().apply {
                put("word", entry.word)
                put("timestamp", entry.timestamp)
                put("source", entry.source)
            })
        }
        prefs().edit().putString("entries", json.toString()).apply()
    }

    private fun prefs() = context.getSharedPreferences("visuall_conversa", Context.MODE_PRIVATE)
}
