package com.visuall.app.libras

// O que a pessoa VÊ e OUVE a partir da frase montada.
//
// Duas decisões que estavam soltas dentro do LibrasFragment, no meio de código
// que mexe em view e chama TextToSpeech — então nenhuma das duas dava pra
// testar, apesar de uma delas escolher o texto que é falado em voz alta.
internal object PhraseOutput {

    /**
     * Acrescenta "?" ao texto exibido quando o marcador de sobrancelha está
     * ativo, sem mexer na frase armazenada.
     *
     * Porta de montar_exibicao (linear/backend/app.py). O "?" é só apresentação:
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
     * Só olha crescimento: apagar e limpar não falam. E vale notar que a frase
     * nova não é necessariamente a antiga mais um pedaço — aplicar uma sugestão
     * reescreve a última palavra. O caminho de "termina em espaço" cobre esse
     * caso, porque toda sugestão deixa um espaço no fim (ver SentenceBuilder).
     */
    fun trechoParaFalar(frase: String, anterior: String): String {
        if (frase.length <= anterior.length) return ""
        if (frase.endsWith(" ")) {
            return frase.trim().substringAfterLast(' ')
        }
        val novo = frase.removePrefix(anterior).trim()
        return novo.ifBlank { frase.lastOrNull()?.toString().orEmpty() }
    }
}
