package com.visuall.app.libras

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

// Fixa a regra que transforma a saída de um modelo em letra ou em "não sei".
//
// Era a lógica mais importante do app sem nenhum teste: ela decide se a tela
// afirma "H" ou fica em silêncio. Não era testável antes porque estava colada
// nas sessões ONNX dentro do LetraEngine — precisava de aparelho e modelo
// treinado pra exercitar uma comparação de dois números.
//
// Os limiares são lidos de LibrasAnalyzer, não copiados: se alguém mexer nos
// valores, estes testes acompanham em vez de virarem mentira silenciosa.
class LetterDecisionTest {

    private val labels = listOf("A", "B", "C")

    private fun estatico(probs: FloatArray) = LetterDecision.deProbabilidades(
        probs = probs,
        labels = labels,
        confiancaMinima = LibrasAnalyzer.CONFIANCA_MINIMA,
        margemMinima = LibrasAnalyzer.MARGEM_ESTATICA_MINIMA,
        modo = "estatico"
    )

    @Test
    fun `vencedor confiante e com folga e aceito`() {
        val p = estatico(floatArrayOf(0.02f, 0.96f, 0.02f))
        assertEquals("B", p.letra)
        assertEquals(0.96f, p.confianca, 1e-6f)
        assertEquals(0.94f, p.margem, 1e-6f)
    }

    @Test
    fun `confianca alta mas empate com o segundo e rejeitado`() {
        // O caso real que motivou a margem: o modelo dividido entre duas
        // letras parecidas (C e mão aberta). A confiança absoluta passa, mas
        // escolher seria moeda ao ar -- e o sintoma na tela é a letra trocando
        // sozinha entre as duas.
        val p = estatico(floatArrayOf(0.02f, 0.49f, 0.49f))
        assertEquals("-", p.letra)
    }

    @Test
    fun `folga enorme nao compensa confianca baixa`() {
        // Margem e confiança são portões independentes: 0.5 contra 0.0 tem
        // folga total, mas o modelo ainda está dizendo "acho que talvez".
        val p = estatico(floatArrayOf(0.5f, 0.0f, 0.0f))
        assertEquals("-", p.letra)
        assertTrue(p.margem >= LibrasAnalyzer.MARGEM_ESTATICA_MINIMA)
    }

    @Test
    fun `no limite exato dos dois portoes aceita`() {
        // Comparações são >=, então o valor exato do limiar passa. Fixado
        // porque trocar por > mudaria o comportamento sem quebrar mais nada.
        val conf = LibrasAnalyzer.CONFIANCA_MINIMA
        val segundo = conf - LibrasAnalyzer.MARGEM_ESTATICA_MINIMA
        assertEquals("B", estatico(floatArrayOf(0f, conf, segundo)).letra)
    }

    @Test
    fun `saida sem rotulo correspondente nao inventa letra`() {
        // labels.txt é gravado junto do modelo a cada treino. Se os dois saírem
        // de sincronia e o modelo tiver mais saídas que rótulos, a saída extra
        // não tem nome. Chutar o nome errado é pior que não responder: viraria
        // "a letra treinada aparece como outra letra".
        val p = LetterDecision.deProbabilidades(
            probs = floatArrayOf(0.01f, 0.01f, 0.01f, 0.97f),
            labels = labels,
            confiancaMinima = LibrasAnalyzer.CONFIANCA_MINIMA,
            margemMinima = LibrasAnalyzer.MARGEM_ESTATICA_MINIMA,
            modo = "estatico"
        )
        assertEquals("-", p.letra)
    }

    @Test
    fun `vetor vazio nao estoura`() {
        val p = estatico(floatArrayOf())
        assertEquals("-", p.letra)
        assertEquals(0f, p.confianca, 1e-6f)
    }

    @Test
    fun `margem e medida contra o segundo melhor e nao contra o pior`() {
        // Com [0.5, 0.45, 0.0], medir contra o pior daria margem 0.5 e
        // aprovaria. Contra o segundo dá 0.05 e rejeita, que é o certo.
        val p = LetterDecision.deProbabilidades(
            probs = floatArrayOf(0.5f, 0.45f, 0.0f),
            labels = labels,
            confiancaMinima = 0.4f,
            margemMinima = 0.3f,
            modo = "estatico"
        )
        assertEquals("-", p.letra)
        assertEquals(0.05f, p.margem, 1e-6f)
    }

    @Test
    fun `limiar dinamico e mais exigente que o estatico`() {
        // Pinado porque o CHANGELOG registra essa decisão: o modelo dinâmico
        // não tem classe "nada" e sai J com facilidade, então a barra dele é
        // mais alta. Se alguém igualar os dois, isto avisa.
        assertTrue(LibrasAnalyzer.CONFIANCA_DINAMICA > LibrasAnalyzer.CONFIANCA_MINIMA)
        assertTrue(LibrasAnalyzer.MARGEM_DINAMICA_MINIMA > LibrasAnalyzer.MARGEM_ESTATICA_MINIMA)
    }

    // ── Modelos individuais (um classificador binário por letra) ────────────

    private fun individual(vararg pontuacoes: Pair<String, Float>) =
        LetterDecision.deModelosIndividuais(
            pontuacoes = pontuacoes.toList(),
            confiancaMinima = LibrasAnalyzer.CONFIANCA_INDIVIDUAL,
            margemMinima = LibrasAnalyzer.MARGEM_DINAMICA_MINIMA,
            confiancaSemRival = LibrasAnalyzer.CONFIANCA_INDIVIDUAL_SEM_RIVAL,
            modo = "dinamico_individual"
        )

    @Test
    fun `com um modelo so a margem nao vale como filtro`() {
        // A regressão que este arquivo existe pra travar. Com um único modelo
        // treinado o segundo lugar é 0, então margem == confiança e o portão de
        // margem aprova qualquer coisa acima de CONFIANCA_INDIVIDUAL. E esse é
        // o PRIMEIRO cenário que acontece na prática: treinar uma letra só pra
        // medir se gravar resolve gera exatamente um modelo individual.
        val quaseLa = LibrasAnalyzer.CONFIANCA_INDIVIDUAL
        assertTrue(
            "o teste só faz sentido se houver folga entre os dois limiares",
            LibrasAnalyzer.CONFIANCA_INDIVIDUAL_SEM_RIVAL > quaseLa
        )
        assertEquals("-", individual("H" to quaseLa).letra)
    }

    @Test
    fun `com um modelo so aceita quando passa do limiar sem rival`() {
        val p = individual("H" to LibrasAnalyzer.CONFIANCA_INDIVIDUAL_SEM_RIVAL)
        assertEquals("H", p.letra)
        // Margem 0, não a confiança: nenhuma folga foi medida contra ninguém, e
        // relatar a confiança como se fosse margem era o que escondia o furo.
        assertEquals(0f, p.margem, 1e-6f)
    }

    @Test
    fun `dois modelos com vencedor destacado e aceito`() {
        val p = individual("H" to 0.99f, "K" to 0.10f)
        assertEquals("H", p.letra)
        assertEquals(0.89f, p.margem, 1e-6f)
    }

    @Test
    fun `dois modelos igualmente confiantes sao rejeitados`() {
        // Dois binários dizendo "sou eu" com 0.99 é o sintoma de modelos que
        // não aprenderam a rejeitar -- responder aí é chutar.
        assertEquals("-", individual("H" to 0.99f, "K" to 0.98f).letra)
    }

    @Test
    fun `terceiro colocado nao afeta a margem`() {
        val p = individual("H" to 0.99f, "K" to 0.98f, "Z" to 0.01f)
        assertEquals("-", p.letra)
        assertEquals(0.01f, p.margem, 1e-6f)
    }

    @Test
    fun `nenhum modelo carregado devolve indefinido`() {
        val p = LetterDecision.deModelosIndividuais(
            pontuacoes = emptyList(),
            confiancaMinima = LibrasAnalyzer.CONFIANCA_INDIVIDUAL,
            margemMinima = LibrasAnalyzer.MARGEM_DINAMICA_MINIMA,
            confiancaSemRival = LibrasAnalyzer.CONFIANCA_INDIVIDUAL_SEM_RIVAL,
            modo = "dinamico_individual"
        )
        assertEquals("-", p.letra)
    }

    @Test
    fun `limiar individual e mais alto que o do modelo geral`() {
        // Decisão registrada no CHANGELOG: binário nunca viu "mão mexendo sem
        // sinalizar" como negativo, então é propenso a confiança alta em nada.
        assertTrue(LibrasAnalyzer.CONFIANCA_INDIVIDUAL > LibrasAnalyzer.CONFIANCA_DINAMICA)
    }

    // ── Média de frames (referência de calibração) ──────────────────────────

    @Test
    fun `media de frames`() {
        val media = LetterDecision.media(
            listOf(floatArrayOf(0f, 2f, 4f), floatArrayOf(2f, 4f, 8f)),
            features = 3
        )
        assertEquals(1f, media[0], 1e-6f)
        assertEquals(3f, media[1], 1e-6f)
        assertEquals(6f, media[2], 1e-6f)
    }

    @Test
    fun `media de lista vazia devolve zeros do tamanho pedido`() {
        val media = LetterDecision.media(emptyList(), features = LibrasAnalyzer.FEATURES_ESTATICO)
        assertEquals(LibrasAnalyzer.FEATURES_ESTATICO, media.size)
        assertTrue(media.all { it == 0f })
    }
}
