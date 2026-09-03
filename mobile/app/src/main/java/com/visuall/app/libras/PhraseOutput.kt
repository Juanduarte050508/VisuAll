package com.visuall.app.libras

import java.text.Normalizer
import java.util.Locale

// O que a pessoa VÊ e OUVE a partir da frase montada.
//
// Decisões que estavam soltas dentro do LibrasFragment, no meio de código que
// mexe em view e chama TextToSpeech — então nenhuma delas dava pra testar,
// apesar de escolherem o texto que é falado em voz alta. São três: o "?" da
// exibição, QUAL trecho falar quando a frase cresce e COMO pronunciá-lo.
internal object PhraseOutput {

    /**
     * Acrescenta "?" ao texto exibido quando o marcador de sobrancelha está
     * ativo, sem mexer na frase armazenada.
     *
     * Porta de montar_exibicao (computer/linear/backend/app.py). O "?" é só apresentação:
     * a frase guardada continua sem ele, senão apagar uma letra deixaria o "?"
     * preso no meio do texto.
     */
    fun exibicao(base: String, interrogativo: Boolean): String {
        if (!interrogativo || base.isBlank()) return base
        val limpo = base.trimEnd()
        // Não duplica se a pessoa já escreveu "?" à mão.
        return if (limpo.endsWith("?")) base else "$limpo?"
    }

    /**
     * O trecho a ser falado quando a frase cresce. String vazia = não falar
     * nada.
     *
     * A regra: normalmente fala só o que entrou de novo (uma letra). Mas quando
     * a frase termina em espaço, a palavra acabou de ser fechada — aí falar a
     * palavra inteira é muito mais útil que falar o espaço.
     *
     * Só olha crescimento: apagar e limpar não falam. O caminho de "termina em
     * espaço" é a palavra fechada por adicionarEspaco(); o do removePrefix
     * cobre o modo corpo, onde a frase não é montada aqui — ela é retraduzida
     * de fora e chega inteira por definir(), então nem sempre é a anterior mais
     * um pedaço.
     */
    fun trechoParaFalar(frase: String, anterior: String): String {
        if (frase.length <= anterior.length) return ""
        if (frase.endsWith(" ")) {
            return frase.trim().substringAfterLast(' ')
        }
        val novo = frase.removePrefix(anterior).trim()
        return novo.ifBlank { frase.lastOrNull()?.toString().orEmpty() }
    }

    // ── Pronúncia ──────────────────────────────────────────────────────────

    private val PT_BR = Locale("pt", "BR")
    private val DIACRITICOS = "\\p{Mn}+".toRegex()

    // Nome de cada letra em português. É o que impede o motor de voz de
    // reinterpretar o texto: entregando "vê" não sobra nada pra ele normalizar.
    private val NOME_DA_LETRA = mapOf(
        'A' to "á", 'B' to "bê", 'C' to "cê", 'D' to "dê", 'E' to "é",
        'F' to "efe", 'G' to "gê", 'H' to "agá", 'I' to "i", 'J' to "jota",
        'K' to "cá", 'L' to "ele", 'M' to "eme", 'N' to "ene", 'O' to "ó",
        'P' to "pê", 'Q' to "quê", 'R' to "erre", 'S' to "esse", 'T' to "tê",
        'U' to "u", 'V' to "vê", 'W' to "dáblio", 'X' to "xis",
        'Y' to "ípsilon", 'Z' to "zê", 'Ç' to "cê cedilha"
    )

    // Abreviações que o motor de voz expande sozinho: soletrar A-V e fechar a
    // palavra saía como "avenida", R virava "rua", KM virava "quilômetro".
    // Quem soletra está escrevendo LETRAS, não abreviando nada — então toda
    // palavra desta lista é soletrada em vez de lida.
    //
    // Letra sozinha não precisa entrar aqui: o caminho de uma letra já soletra
    // sempre, o que cobre R, N, M, H e companhia.
    private val ABREVIACOES = setOf(
        "av", "al", "apto", "aprox", "art", "bl", "cel", "cia", "cm", "dep",
        "dr", "dra", "end", "esq", "etc", "ex", "jr", "kg", "km", "lt", "ltda",
        "min", "mm", "num", "obs", "pag", "pca", "prof", "profa", "qtd", "ref",
        "rod", "seg", "sr", "sra", "srta", "tel", "trav"
    )

    /**
     * O texto que vai para o TextToSpeech, a partir do trecho devolvido por
     * [trechoParaFalar].
     *
     * Três decisões, em ordem:
     *  1. Letra sozinha → o nome dela ("V" → "vê"). Sem isso o motor lê a letra
     *     como palavra ou como abreviação.
     *  2. Palavra que é abreviação conhecida → soletrada letra por letra.
     *  3. Qualquer outra palavra → lida normalmente, em minúsculas. As letras
     *     chegam em CAIXA ALTA, e é a caixa alta que faz o motor tratar o token
     *     como sigla.
     */
    fun textoParaVoz(trecho: String): String {
        val limpo = trecho.trim()
        if (limpo.isEmpty()) return ""
        if (limpo.length == 1) return soletrar(limpo)
        if (normalizar(limpo) in ABREVIACOES) return soletrar(limpo)
        return limpo.lowercase(PT_BR)
    }

    /**
     * Letra por letra, separadas por vírgula pra o motor pausar entre elas em
     * vez de emendar tudo numa palavra inventada.
     */
    private fun soletrar(texto: String): String = texto
        .uppercase(PT_BR)
        .mapNotNull { c -> NOME_DA_LETRA[c] ?: if (c.isLetterOrDigit()) c.toString() else null }
        .joinToString(", ")

    /** Minúsculas e sem acento, pra "AV" e "Av." casarem com "av" da lista. */
    private fun normalizar(texto: String): String {
        val soLetras = texto.filter { it.isLetter() }
        return Normalizer.normalize(soLetras.lowercase(PT_BR), Normalizer.Form.NFD)
            .replace(DIACRITICOS, "")
    }
}
