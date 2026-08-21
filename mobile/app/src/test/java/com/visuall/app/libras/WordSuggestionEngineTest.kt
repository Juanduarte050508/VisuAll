package com.visuall.app.libras

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class WordSuggestionEngineTest {

    @Test
    fun `normalizar tira acento e caixa`() {
        assertEquals("mae", WordSuggestionEngine.normalizar("MÃE"))
        assertEquals("familia", WordSuggestionEngine.normalizar("  Família "))
    }

    @Test
    fun `prefixo de uma letra nao sugere nada`() {
        // Com 1 letra quase todo o vocabulário casa -- sugerir aí é ruído.
        assertTrue(WordSuggestionEngine.sugerir("a").isEmpty())
    }

    @Test
    fun `prefixo de duas letras ja sugere`() {
        val sugestoes = WordSuggestionEngine.sugerir("ba")
        assertTrue("esperava banheiro em $sugestoes", "banheiro" in sugestoes)
    }

    @Test
    fun `palavra que comeca com o prefixo ganha de palavra que so contem`() {
        // "conversa"/"conversar" começam com "conv"; nenhuma outra contém.
        val sugestoes = WordSuggestionEngine.sugerir("conv")
        assertEquals("conversa", sugestoes.first())
    }

    @Test
    fun `palavra ja completa nao e sugerida de novo`() {
        val sugestoes = WordSuggestionEngine.sugerir("banheiro")
        assertFalse("banheiro" in sugestoes)
    }

    @Test
    fun `contextual aparece so depois do espaco`() {
        // Sem espaço: ainda está escrevendo "bom", então "dia" não entra.
        assertFalse("dia" in WordSuggestionEngine.sugerir("bom"))
        // Com espaço: fechou "bom", agora "dia" é o próximo provável.
        assertTrue("dia" in WordSuggestionEngine.sugerir("bom "))
    }

    @Test
    fun `contextual considera so a ultima palavra da frase`() {
        val sugestoes = WordSuggestionEngine.sugerir("eu quero ")
        assertTrue("esperava contextuais de 'quero' em $sugestoes", "comida" in sugestoes)
    }

    @Test
    fun `nunca devolve mais que o limite`() {
        assertTrue(WordSuggestionEngine.sugerir("a ").size <= 3)
        assertEquals(1, WordSuggestionEngine.sugerir("co", limite = 1).size)
    }

    @Test
    fun `busca ignora acento digitado`() {
        // "mãe" está no vocabulário como "mae"; digitar com acento deve casar.
        val sugestoes = WordSuggestionEngine.sugerir("mã")
        assertTrue("esperava mae em $sugestoes", "mae" in sugestoes)
    }

    @Test
    fun `frase vazia nao quebra`() {
        assertTrue(WordSuggestionEngine.sugerir("").isEmpty())
        assertTrue(WordSuggestionEngine.sugerir("   ").isEmpty())
    }
}
