package com.visuall.app.libras

import org.junit.Assert.assertEquals
import org.junit.Test

class SentenceTranslatorTest {

    @Test
    fun `substantivo sozinho recebe artigo e maiuscula inicial`() {
        assertEquals("A pessoa", SentenceTranslator.traduzirFrase(listOf("PESSOA")))
    }

    @Test
    fun `adjetivo concorda em genero com o substantivo anterior`() {
        assertEquals("A pessoa surda", SentenceTranslator.traduzirFrase(listOf("PESSOA", "SURDO")))
    }

    @Test
    fun `verbo depois de substantivo usa a forma conjugada`() {
        assertEquals("A pessoa conversa", SentenceTranslator.traduzirFrase(listOf("PESSOA", "CONVERSAR")))
    }

    @Test
    fun `segundo verbo em sequencia vira infinitivo`() {
        assertEquals(
            "A pessoa conversa a ajudar",
            SentenceTranslator.traduzirFrase(listOf("PESSOA", "CONVERSAR", "AJUDAR"))
        )
    }

    @Test
    fun `token NEUTRO e ignorado`() {
        assertEquals("", SentenceTranslator.traduzirFrase(listOf("NEUTRO")))
        assertEquals("A pessoa", SentenceTranslator.traduzirFrase(listOf("NEUTRO", "PESSOA")))
    }

    @Test
    fun `lista vazia retorna string vazia`() {
        assertEquals("", SentenceTranslator.traduzirFrase(emptyList()))
    }

    @Test
    fun `palavra fora do vocabulario e usada como esta, so capitalizada`() {
        assertEquals("Oi", SentenceTranslator.traduzirFrase(listOf("OI")))
    }

    @Test
    fun `traduzirCorpo mapeia rotulos conhecidos independente de caixa`() {
        assertEquals("ajuda", SentenceTranslator.traduzirCorpo("AJUDAR"))
        assertEquals("pessoa", SentenceTranslator.traduzirCorpo("pessoa"))
    }

    @Test
    fun `traduzirCorpo usa minusculas para rotulos desconhecidos`() {
        assertEquals("desconhecido", SentenceTranslator.traduzirCorpo("DESCONHECIDO"))
    }
}
