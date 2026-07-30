package com.visuall.app.libras

import org.junit.Assert.assertEquals
import org.junit.Test

// Fixa o que é exibido e o que é falado a partir da frase.
//
// O trecho falado importa mais do que parece: é a saída que a pessoa surda usa
// pra conferir se o app entendeu, e errar aqui significa o celular falar uma
// frase inteira quando devia falar uma letra.
class PhraseOutputTest {

    // ── Exibição do "?" (porta de montar_exibicao do app.py) ────────────────

    @Test
    fun `sem marcador a frase sai como esta`() {
        assertEquals("OLA", PhraseOutput.exibicao("OLA", interrogativo = false))
    }

    @Test
    fun `com marcador acrescenta interrogacao`() {
        assertEquals("OLA?", PhraseOutput.exibicao("OLA", interrogativo = true))
    }

    @Test
    fun `interrogacao vem depois do texto e nao depois do espaco`() {
        assertEquals("TUDO BEM?", PhraseOutput.exibicao("TUDO BEM ", interrogativo = true))
    }

    @Test
    fun `nao duplica interrogacao ja existente`() {
        assertEquals("OLA?", PhraseOutput.exibicao("OLA?", interrogativo = true))
    }

    @Test
    fun `frase vazia nao vira so uma interrogacao`() {
        // Sobrancelha levantada sem nada escrito não deve mostrar "?" solto.
        assertEquals("", PhraseOutput.exibicao("", interrogativo = true))
        assertEquals("   ", PhraseOutput.exibicao("   ", interrogativo = true))
    }

    // ── Trecho falado ───────────────────────────────────────────────────────

    @Test
    fun `fala a letra que acabou de entrar`() {
        assertEquals("B", PhraseOutput.trechoParaFalar("AB", "A"))
    }

    @Test
    fun `fala a primeira letra da frase`() {
        assertEquals("A", PhraseOutput.trechoParaFalar("A", ""))
    }

    @Test
    fun `ao fechar a palavra fala a palavra inteira`() {
        // O espaço no fim significa "palavra terminada". Falar o espaço não
        // serviria de nada; falar "CASA" é a confirmação útil.
        assertEquals("CASA", PhraseOutput.trechoParaFalar("CASA ", "CASA"))
    }

    @Test
    fun `ao fechar a palavra ignora as palavras anteriores`() {
        assertEquals("CASA", PhraseOutput.trechoParaFalar("MINHA CASA ", "MINHA CASA"))
    }

    @Test
    fun `sugestao aplicada fala a palavra sugerida e nao a frase toda`() {
        // O caso que mais podia dar errado: a frase nova NÃO é a antiga mais um
        // pedaço, porque a sugestão reescreveu a última palavra. Um
        // removePrefix simples devolveria a frase inteira e o app falaria tudo.
        // Toda sugestão deixa espaço no fim (ver SentenceBuilder), e é isso que
        // salva este caminho.
        assertEquals("CASA", PhraseOutput.trechoParaFalar("MINHA CASA ", "MINHA CAS"))
    }

    @Test
    fun `apagar nao fala nada`() {
        assertEquals("", PhraseOutput.trechoParaFalar("A", "AB"))
    }

    @Test
    fun `limpar nao fala nada`() {
        assertEquals("", PhraseOutput.trechoParaFalar("", "OLA"))
    }

    @Test
    fun `frase inalterada nao fala nada`() {
        assertEquals("", PhraseOutput.trechoParaFalar("OLA", "OLA"))
    }

    @Test
    fun `letra repetida confirmada fala a letra`() {
        assertEquals("S", PhraseOutput.trechoParaFalar("PASS", "PAS"))
    }

    @Test
    fun `espaco em frase vazia nao tenta falar`() {
        // " " cresceu em relação a "", mas não há palavra pra anunciar.
        assertEquals("", PhraseOutput.trechoParaFalar(" ", ""))
    }
}
