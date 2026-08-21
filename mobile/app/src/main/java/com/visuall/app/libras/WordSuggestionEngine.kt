package com.visuall.app.libras

import java.text.Normalizer
import java.util.Locale

// Sugestão de palavras enquanto a pessoa soletra — extraído do
// LibrasFragment porque é lógica pura (texto entra, lista de palavras sai),
// sem nenhuma dependência de View/binding. Isso também deixa a regra de
// pontuação testável: antes dela estar aqui, a única forma de conferir se
// "ban" sugere "banheiro" era abrir o app e soletrar na mão.
internal object WordSuggestionEngine {

    // Compilado uma vez: normalizar() roda pra cada palavra do vocabulário
    // (~45) toda vez que a frase muda, então recompilar o regex ali dentro
    // seria refazer o mesmo trabalho repetidamente.
    private val DIACRITICS_REGEX = "\\p{Mn}+".toRegex()
    private val PT_BR = Locale("pt", "BR")

    val PALAVRAS = listOf(
        "ajuda", "ajudar", "agua", "amigo", "amanha", "aprender", "aqui",
        "banheiro", "bom", "boa", "casa", "comida", "computador", "conversa",
        "conversar", "desculpa", "dor", "escola", "estou", "familia", "feliz",
        "hoje", "jovi", "libras", "mae", "medico", "nao", "obrigado", "obrigada",
        "oi", "onde", "pai", "pessoa", "por favor", "preciso", "professor",
        "quero", "responder", "sim", "surdo", "tudo", "voce", "voltar"
    )

    // Palavras que costumam vir DEPOIS de outra ("bom" → "dia"). Só entram
    // quando a frase termina em espaço, ou seja, a palavra anterior está
    // fechada e a pessoa vai começar outra.
    val CONTEXTUAIS = mapOf(
        "bom" to listOf("dia"),
        "boa" to listOf("tarde", "noite"),
        "por" to listOf("favor"),
        "eu" to listOf("preciso", "quero", "estou"),
        "voce" to listOf("pode", "quer", "entendeu"),
        "preciso" to listOf("ajuda", "agua", "medico"),
        "quero" to listOf("comida", "agua", "conversar")
    )

    // Minúsculas + sem acento, pra "MÃE" digitado casar com "mae" da lista.
    fun normalizar(texto: String): String {
        return Normalizer.normalize(texto.trim().lowercase(PT_BR), Normalizer.Form.NFD)
            .replace(DIACRITICS_REGEX, "")
    }

    fun sugerir(
        frase: String,
        limite: Int = 3,
        palavras: List<String> = PALAVRAS,
        contextuaisPorPalavra: Map<String, List<String>> = CONTEXTUAIS
    ): List<String> {
        val textoNormal = frase.lowercase(PT_BR)
        val prefixo = textoNormal.substringAfterLast(' ').trim()
        val prefixoBusca = normalizar(prefixo)
        val ultimaCompleta = textoNormal.trim().split(" ").lastOrNull().orEmpty()
        val contextuais = if (frase.endsWith(" ")) {
            contextuaisPorPalavra[normalizar(ultimaCompleta)].orEmpty()
        } else {
            emptyList()
        }

        return (contextuais + palavras)
            .distinct()
            .mapNotNull { palavra ->
                val busca = normalizar(palavra)
                val score = when {
                    // Acabou de fechar uma palavra: mostra o que costuma vir
                    // depois, sem precisar de prefixo nenhum.
                    prefixoBusca.isBlank() && palavra in contextuais -> 100
                    // Com 1 letra só quase tudo casa — sugerir aí seria ruído.
                    prefixoBusca.length < 2 -> -1
                    // Começa com o que foi digitado: quanto menos sobra pra
                    // digitar, melhor a sugestão.
                    busca.startsWith(prefixoBusca) -> 80 - (busca.length - prefixoBusca.length)
                    // Contém no meio: vale menos, e quanto mais pro fim da
                    // palavra o trecho aparece, menos ainda.
                    busca.contains(prefixoBusca) -> 35 - busca.indexOf(prefixoBusca)
                    else -> -1
                }
                // busca == prefixoBusca: a palavra já está inteira escrita,
                // sugerir ela de novo não ajuda em nada.
                if (score < 0 || busca == prefixoBusca) null else palavra to score
            }
            .sortedWith(compareByDescending<Pair<String, Int>> { it.second }.thenBy { it.first.length })
            .map { it.first }
            .take(limite)
    }
}
