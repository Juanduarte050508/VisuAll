package com.visuall.app.oculos

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Os motivos usados aqui sao TEXTOS REAIS, copiados do que apareceu na tela do
 * celular e no logcat durante os testes. Inventar mensagens plausiveis nao
 * provaria nada: o valor deste arquivo e casar com o que o Android escreve de
 * verdade, que e o que ninguem consegue adivinhar de cabeca.
 */
class MensagemDeErroTest {

    /** Foi o que apareceu em vermelho na tela antes do network_security_config. */
    @Test
    fun `bloqueio de http em claro`() {
        val fala = MensagemDeErro.emPortugues(
            "Cleartext HTTP traffic to 192.168.15.10 not permitted")
        assertTrue("devia falar em reinstalar: $fala", fala.contains("reinstalar"))
    }

    @Test
    fun `ninguem escutando na porta`() {
        val fala = MensagemDeErro.emPortugues(
            "failed to connect to /192.168.4.1 (port 80) from /192.168.4.2 " +
                "(port 41234) after 3000ms: isConnected failed: ECONNREFUSED " +
                "(Connection refused)")
        assertTrue("devia mandar conferir se os oculos estao ligados: $fala",
            fala.contains("ligados"))
    }

    /**
     * O motivo vem com maiusculas ("Wi-Fi") e a comparacao e feita em
     * minusculas: casar os dois na mao e um erro que passa despercebido,
     * porque a mensagem generica tambem "parece certa" na tela.
     */
    @Test
    fun `rede dos oculos ainda nao encontrada`() {
        val fala = MensagemDeErro.emPortugues(RedeDosOculos.SEM_REDE)
        assertTrue("devia mandar conectar no Wi-Fi dos oculos: $fala",
            fala.contains("Wi-Fi dos oculos"))
    }

    @Test
    fun `celular na rede errada`() {
        val fala = MensagemDeErro.emPortugues("connect failed: ENETUNREACH (Network is unreachable)")
        assertTrue("devia mandar conferir o Wi-Fi: $fala", fala.contains("Wi-Fi"))
    }

    @Test
    fun `stream que parou de responder`() {
        assertEquals(
            MensagemDeErro.emPortugues("Read timed out"),
            MensagemDeErro.emPortugues("Software caused connection abort"))
    }

    /** O engano mais comum: digitar o endereco sem o /stream no fim. */
    @Test
    fun `endereco que devolve pagina em vez de video`() {
        val fala = MensagemDeErro.emPortugues(
            "resposta nao e MJPEG (Content-Type: text/html; charset=utf-8)")
        assertTrue("devia citar o /stream: $fala", fala.contains("/stream"))
    }

    @Test
    fun `motivo desconhecido vira frase generica, nao texto vazio`() {
        val fala = MensagemDeErro.emPortugues("java.lang.IllegalStateException")
        assertTrue(fala.isNotBlank())
        assertTrue("devia falar dos oculos: $fala", fala.contains("oculos"))
    }

    /**
     * A regra que motivou o arquivo inteiro. Um motivo novo, que nenhum `when`
     * daqui reconhece, ainda assim nao pode chegar cru na tela -- foi
     * exatamente assim que "Cleartext HTTP traffic to..." apareceu em vermelho
     * por cima da camera.
     */
    @Test
    fun `nunca devolve o motivo cru`() {
        val crus = listOf(
            "Cleartext HTTP traffic to 192.168.15.10 not permitted",
            "failed to connect to /192.168.4.1 (port 80) after 3000ms",
            "Unable to resolve host \"oculos\": No address associated with hostname",
            "o stream respondeu HTTP 404",
            "SSLHandshakeException: chain validation failed",
            "algo que nunca vimos antes")
        for (cru in crus) {
            val fala = MensagemDeErro.emPortugues(cru)
            assertFalse("vazou o motivo cru em: $fala", fala.contains(cru))
            assertFalse("mensagem em ingles chegando na tela: $fala",
                fala.contains("failed") || fala.contains("not permitted") ||
                    fala.contains("Exception"))
        }
    }
}
