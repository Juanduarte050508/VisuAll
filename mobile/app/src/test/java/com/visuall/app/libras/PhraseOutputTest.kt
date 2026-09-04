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
    fun `frase reescrita que fecha palavra fala so a ultima e nao a frase toda`() {
        // O caso que mais podia dar errado: a frase nova NÃO é a antiga mais um
        // pedaço, porque o fim dela foi reescrito. Um removePrefix simples
        // devolveria a frase inteira e o app falaria tudo. É o teste de
        // espaço-no-fim que salva este caminho.
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

    // ── Pronúncia (textoParaVoz) ───────────────────────────────────────────
    //
    // O motor de voz normaliza o texto que recebe e expande abreviação sozinho.
    // Quem soletra em Libras está escrevendo LETRAS: nada do que sai daqui pode
    // dar margem pra ele adivinhar uma palavra que a pessoa não escreveu.

    @Test
    fun `letra sozinha e falada pelo nome`() {
        assertEquals("vê", PhraseOutput.textoParaVoz("V"))
        assertEquals("agá", PhraseOutput.textoParaVoz("H"))
    }

    @Test
    fun `letra que e abreviacao conhecida nao vira a palavra expandida`() {
        // Sem isso o motor lia "R" como "rua" e "N" como "número".
        assertEquals("erre", PhraseOutput.textoParaVoz("R"))
        assertEquals("ene", PhraseOutput.textoParaVoz("N"))
    }

    @Test
    fun `AV e soletrado e nao lido como avenida`() {
        // O caso que motivou tudo isto.
        assertEquals("á, vê", PhraseOutput.textoParaVoz("AV"))
    }

    @Test
    fun `outras abreviacoes comuns tambem sao soletradas`() {
        assertEquals("cá, eme", PhraseOutput.textoParaVoz("KM"))
        assertEquals("dê, erre", PhraseOutput.textoParaVoz("DR"))
        assertEquals("tê, é, ele", PhraseOutput.textoParaVoz("TEL"))
    }

    @Test
    fun `palavra de verdade continua sendo falada como palavra`() {
        assertEquals("casa", PhraseOutput.textoParaVoz("CASA"))
        assertEquals("banheiro", PhraseOutput.textoParaVoz("BANHEIRO"))
    }

    @Test
    fun `palavra vai em minusculas porque caixa alta vira sigla`() {
        // "OBRIGADO" em caixa alta é o que faz o motor tratar o token como
        // sigla e soletrar sozinho, do jeito errado.
        assertEquals("obrigado", PhraseOutput.textoParaVoz("OBRIGADO"))
    }

    @Test
    fun `trecho vazio nao vira fala`() {
        assertEquals("", PhraseOutput.textoParaVoz(""))
        assertEquals("", PhraseOutput.textoParaVoz("   "))
    }
}
