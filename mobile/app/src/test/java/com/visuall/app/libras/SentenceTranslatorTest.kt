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
    fun `verbo depois de adjetivo tambem vira infinitivo se ja houve verbo`() {
        // A frase da apresentacao. Saia "...surda conversa" porque a regra
        // olhava so o token anterior (SURDO, um adjetivo) em vez de perguntar
        // se algum verbo ja tinha entrado na frase.
        assertEquals(
            "O computador ajuda a pessoa surda a conversar",
            SentenceTranslator.traduzirFrase(
                listOf("COMPUTADOR", "AJUDAR", "PESSOA", "SURDO", "CONVERSAR")
            )
        )
    }

    @Test
    fun `o primeiro verbo continua conjugado mesmo longe do sujeito`() {
        assertEquals(
            "A pessoa surda conversa",
            SentenceTranslator.traduzirFrase(listOf("PESSOA", "SURDO", "CONVERSAR"))
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
