package com.visuall.app.libras

// A frase que está sendo montada letra por letra, e as regras de como ela muda.
//
// Vivia solta no meio do LibrasAnalyzer, junto do processamento de câmera, então
// nada disso era alcançável num teste: exigia aparelho, câmera e modelo
// carregado pra exercitar uma regra sobre concatenação de strings. É a parte
// que o usuário vê diretamente — se ela erra, aparece letra a mais ou de menos
// na tela.
//
// A classe só mexe no texto. Quem chama continua responsável por avisar a UI e
// por destravar o LetterCommitGate, usando o retorno de cada método. Ver
// SentenceBuilderTest.
internal class SentenceBuilder {

    var frase: String = ""
        private set

    // Letra reconhecida que seria uma repetição da última e por isso está
    // esperando confirmação explícita (botão REPETIR). "" = nada pendente.
    var letraRepetidaPendente: String = ""
        private set

    enum class Resultado {
        /** Entrou na frase. */
        ADICIONADA,

        /** É repetição da anterior e precisa de confirmação. */
        AGUARDANDO_CONFIRMACAO
    }

    /**
     * Uma letra passou por todos os portões e pode entrar.
     *
     * Repetir a mesma letra é o caso perigoso: a mão fica parada no sinal e o
     * app digitaria "AAAA". Por isso a segunda letra igual só entra sozinha se
     * for uma das que costumam aparecer dobradas de verdade em português;
     * qualquer outra fica pendente esperando o usuário confirmar.
     */
    fun aceitarLetra(letra: String): Resultado {
        val repetindo = frase.lastOrNull()?.toString() == letra
        if (repetindo && !podeRepetirAutomaticamente(letra)) {
            letraRepetidaPendente = letra
            return Resultado.AGUARDANDO_CONFIRMACAO
        }
        frase += letra
        letraRepetidaPendente = ""
        return Resultado.ADICIONADA
    }

    // Só letras que dobram naturalmente em português (LETRAS_REPETICAO_AUTO), e
    // no máximo duas seguidas: "ss" passa, "sss" não. Sem esse segundo limite,
    // a mão parada num S ainda encheria a frase.
    private fun podeRepetirAutomaticamente(letra: String): Boolean {
        if (letra !in LibrasAnalyzer.LETRAS_REPETICAO_AUTO) return false
        return frase.length < 2 || frase[frase.length - 2].toString() != letra
    }

    /** Confirma a repetição pendente. Devolve false se não havia nenhuma. */
    fun confirmarRepeticao(): Boolean {
        if (letraRepetidaPendente.isBlank()) return false
        frase += letraRepetidaPendente
        letraRepetidaPendente = ""
        return true
    }

    fun adicionarEspaco() {
        frase += " "
        letraRepetidaPendente = ""
    }

    /**
     * Troca a última palavra pela sugestão escolhida, deixando um espaço no fim
     * pra a próxima palavra já começar limpa. Devolve false se a sugestão for
     * vazia.
     */
    fun aplicarSugestao(palavra: String): Boolean {
        val sugestao = palavra.trim()
        if (sugestao.isBlank()) return false
        // substringBeforeLast com valor padrão "" é o que faz a frase de uma
        // palavra só ser substituída por inteiro, em vez de a sugestão ser
        // grudada nela.
        val prefixo = frase.substringBeforeLast(" ", missingDelimiterValue = "")
        frase = if (prefixo.isBlank()) "$sugestao " else "$prefixo $sugestao "
        letraRepetidaPendente = ""
        return true
    }

    /** Apaga o último caractere. Devolve false se a frase já estava vazia. */
    fun apagarUltima(): Boolean {
        if (frase.isEmpty()) return false
        frase = frase.dropLast(1)
        letraRepetidaPendente = ""
        return true
    }

    fun limpar() {
        frase = ""
        letraRepetidaPendente = ""
    }

    /**
     * Descarta só a repetição pendente, preservando a frase. Usado quando a mão
     * sai do quadro: a sequência anterior perde relação com a próxima, mas o
     * que já foi escrito continua valendo.
     */
    fun limparPendente() {
        letraRepetidaPendente = ""
    }

    /**
     * Substitui a frase inteira. Usado no modo corpo, onde apagar remove um
     * SINAL (token) e a frase é retraduzida de fora, não montada aqui.
     */
    fun definir(nova: String) {
        frase = nova
        letraRepetidaPendente = ""
    }
}
