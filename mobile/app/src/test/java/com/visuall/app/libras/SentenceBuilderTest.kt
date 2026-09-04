package com.visuall.app.libras

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

// Fixa como a frase muda letra a letra.
//
// É a parte que o usuário vê literalmente: se erra, aparece letra a mais, letra
// a menos, ou a palavra sugerida grudada na anterior. Estava dentro do
// LibrasAnalyzer, junto do processamento de câmera, então testar uma regra sobre
// concatenação de string exigia aparelho, câmera e modelo carregado.
class SentenceBuilderTest {

    private fun builder(vararg letras: String) = SentenceBuilder().apply {
        letras.forEach { aceitarLetra(it) }
    }

    // ── Letras diferentes ───────────────────────────────────────────────────

    @Test
    fun `letras diferentes entram direto`() {
        val s = builder("O", "L", "A")
        assertEquals("OLA", s.frase)
        assertEquals("", s.letraRepetidaPendente)
    }

    // ── Repetição ───────────────────────────────────────────────────────────

    @Test
    fun `letra repetida espera confirmacao`() {
        // Sem essa regra, a mão parada no sinal digitaria "AAAA".
        val s = builder("A")
        assertEquals(SentenceBuilder.Resultado.AGUARDANDO_CONFIRMACAO, s.aceitarLetra("A"))
        assertEquals("A", s.frase, )
        assertEquals("A", s.letraRepetidaPendente)
    }

    @Test
    fun `confirmar repeticao escreve a letra`() {
        val s = builder("A")
        s.aceitarLetra("A")
        assertTrue(s.confirmarRepeticao())
        assertEquals("AA", s.frase)
        assertEquals("", s.letraRepetidaPendente)
    }

    @Test
    fun `confirmar sem nada pendente nao mexe na frase`() {
        val s = builder("A", "B")
        assertFalse(s.confirmarRepeticao())
        assertEquals("AB", s.frase)
    }

    @Test
    fun `letras que dobram naturalmente entram sozinhas`() {
        // S e R dobram de verdade em português ("passo", "carro"), então exigir
        // confirmação nelas atrapalharia mais do que ajudaria.
        LibrasAnalyzer.LETRAS_REPETICAO_AUTO.forEach { letra ->
            val s = builder(letra)
            assertEquals(
                "esperado $letra dobrar sozinha",
                SentenceBuilder.Resultado.ADICIONADA,
                s.aceitarLetra(letra)
            )
            assertEquals(letra + letra, s.frase)
        }
    }

    @Test
    fun `a terceira letra igual seguida sempre pede confirmacao`() {
        // O limite que impede a mão parada num S de encher a frase: "ss" passa,
        // "sss" não.
        val letra = LibrasAnalyzer.LETRAS_REPETICAO_AUTO.first()
        val s = builder(letra)
        s.aceitarLetra(letra)
        assertEquals(letra + letra, s.frase)
        assertEquals(
            SentenceBuilder.Resultado.AGUARDANDO_CONFIRMACAO,
            s.aceitarLetra(letra)
        )
        assertEquals(letra + letra, s.frase)
    }

    @Test
    fun `letra igual separada por outra nao conta como repeticao`() {
        // "ANA": o segundo A não é repetição do primeiro.
        val s = builder("A", "N")
        assertEquals(SentenceBuilder.Resultado.ADICIONADA, s.aceitarLetra("A"))
        assertEquals("ANA", s.frase)
    }

    @Test
    fun `letra igual separada por espaco entra direto`() {
        val s = builder("A")
        s.adicionarEspaco()
        assertEquals(SentenceBuilder.Resultado.ADICIONADA, s.aceitarLetra("A"))
        assertEquals("A A", s.frase)
    }

    @Test
    fun `pendencia e descartada quando outra letra entra`() {
        // A pessoa não confirmou e fez outra letra: a repetição perdeu a vez.
        val s = builder("A")
        s.aceitarLetra("A")
        assertEquals("A", s.letraRepetidaPendente)
        s.aceitarLetra("B")
        assertEquals("", s.letraRepetidaPendente)
        assertEquals("AB", s.frase)
    }

    // ── Espaço ──────────────────────────────────────────────────────────────

    @Test
    fun `espaco pode ser adicionado em frase vazia`() {
        val s = SentenceBuilder()
        s.adicionarEspaco()
        assertEquals(" ", s.frase)
    }

    // ── Apagar e limpar ─────────────────────────────────────────────────────

    @Test
    fun `apagar remove o ultimo caractere`() {
        val s = builder("A", "B", "C")
        assertTrue(s.apagarUltima())
        assertEquals("AB", s.frase)
    }

    @Test
    fun `apagar em frase vazia nao faz nada`() {
        val s = SentenceBuilder()
        assertFalse(s.apagarUltima())
        assertEquals("", s.frase)
    }

    @Test
    fun `apagar descarta a repeticao pendente`() {
        // Senão o botão REPETIR escreveria uma letra que a pessoa acabou de
        // apagar.
        val s = builder("A")
        s.aceitarLetra("A")
        assertEquals("A", s.letraRepetidaPendente)
        s.apagarUltima()
        assertEquals("", s.letraRepetidaPendente)
        assertFalse(s.confirmarRepeticao())
    }

    @Test
    fun `apagar depois de apagar permite reescrever a mesma letra`() {
        val s = builder("A", "B")
        s.apagarUltima()
        assertEquals(SentenceBuilder.Resultado.ADICIONADA, s.aceitarLetra("B"))
        assertEquals("AB", s.frase)
    }

    @Test
    fun `limpar zera frase e pendencia`() {
        val s = builder("A")
        s.aceitarLetra("A")
        s.limpar()
        assertEquals("", s.frase)
        assertEquals("", s.letraRepetidaPendente)
    }

    @Test
    fun `limparPendente preserva a frase`() {
        // Chamado quando a mão sai do quadro: a sequência anterior perde
        // relação com a próxima, mas o que já foi escrito continua valendo.
        val s = builder("O", "L", "A")
        s.aceitarLetra("A")
        assertEquals("A", s.letraRepetidaPendente)
        s.limparPendente()
        assertEquals("OLA", s.frase)
        assertEquals("", s.letraRepetidaPendente)
    }

    @Test
    fun `definir troca a frase inteira`() {
        // Caminho do modo corpo, onde a frase é retraduzida de fora a cada
        // sinal em vez de montada letra a letra.
        val s = builder("A", "B")
        s.definir("EU AJUDAR VOCE")
        assertEquals("EU AJUDAR VOCE", s.frase)
    }
}
