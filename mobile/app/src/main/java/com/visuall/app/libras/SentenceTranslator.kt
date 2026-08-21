package com.visuall.app.libras

// Tradução de sinais de corpo (rótulos como PESSOA, SURDO...) em frase com
// concordância (artigo/gênero/verbo) — porto fiel do VOCABULARIO/
// traduzir_frase do Python (app.py). Função pura: sem dependência de
// Context nem de estado do analisador, dá pra testar direto em JVM.
internal object SentenceTranslator {

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

    // Rótulo cru do modelo -> palavra exibida no feedback ("CORPO: ajuda").
    fun traduzirCorpo(label: String): String {
        return when (label.uppercase()) {
            "AJUDAR" -> "ajuda"
            "COMPUTADOR" -> "computador"
            "CONVERSAR" -> "conversa"
            "PESSOA" -> "pessoa"
            "SURDO" -> "surdo"
            else -> label.lowercase()
        }
    }

    // Re-traduz a sequência INTEIRA de sinais reconhecidos toda vez (em vez
    // de só concatenar palavra por palavra), porque a concordância de um
    // token (ex.: "surdo"/"surda") depende do gênero do substantivo que veio
    // antes dele na frase.
    fun traduzirFrase(palavras: List<String>): String {
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
}
