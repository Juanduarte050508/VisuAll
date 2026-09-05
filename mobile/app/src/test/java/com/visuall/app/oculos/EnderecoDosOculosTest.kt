package com.visuall.app.oculos

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Esta decisao escolhe entre dois comportamentos bem diferentes -- entrar
 * sozinho na rede da placa, ou usar o Wi-Fi em que o celular ja esta -- e erra
 * em silencio: nos dois casos o app "tenta conectar" e a tela fica preta.
 */
class EnderecoDosOculosTest {

    @Test
    fun `o endereco da placa e reconhecido`() {
        assertTrue(EnderecoDosOculos.ehAPlaca(EnderecoDosOculos.URL))
        assertTrue(EnderecoDosOculos.ehAPlaca("http://192.168.4.1/stream"))
        assertTrue("a pagina de teste tambem e a placa",
            EnderecoDosOculos.ehAPlaca("http://192.168.4.1/"))
        assertTrue("com porta explicita continua sendo a placa",
            EnderecoDosOculos.ehAPlaca("http://192.168.4.1:80/stream"))
        assertTrue("espaco sobrando ao colar o endereco",
            EnderecoDosOculos.ehAPlaca("  http://192.168.4.1/stream  "))
    }

    @Test
    fun `o mock no PC nao e a placa`() {
        assertFalse(EnderecoDosOculos.ehAPlaca("http://192.168.15.10:8080/stream"))
        assertFalse(EnderecoDosOculos.ehAPlaca("http://26.77.90.170:8080/stream"))
    }

    /**
     * O engano que o teste existe pra pegar: 192.168.4.1 dentro do endereco,
     * mas nao como destino. Uma comparacao ingenua com `contains` diria que sim
     * nos tres, e o app iria procurar a rede dos oculos pra falar com outra
     * maquina.
     */
    @Test
    fun `parecido com o endereco da placa nao basta`() {
        assertFalse(EnderecoDosOculos.ehAPlaca("http://192.168.4.10/stream"))
        assertFalse(EnderecoDosOculos.ehAPlaca("http://192.168.41.1/stream"))
        assertFalse("o IP aparece so no caminho",
            EnderecoDosOculos.ehAPlaca("http://192.168.15.10/192.168.4.1/stream"))
    }

    @Test
    fun `endereco quebrado nao trava e nao inventa`() {
        // Digitado a mao numa caixa de texto: vai vir torto as vezes.
        assertFalse(EnderecoDosOculos.ehAPlaca(""))
        assertFalse(EnderecoDosOculos.ehAPlaca("192.168.4.1"))      // sem http://
        assertFalse(EnderecoDosOculos.ehAPlaca("http://"))
        assertFalse(EnderecoDosOculos.ehAPlaca("nao e endereco nenhum"))
    }

    /**
     * Do outro lado destes valores esta o firmware. Se alguem mudar um sem
     * mudar o outro, o app procura uma rede que nao existe -- e o sintoma e a
     * tela preta de sempre, sem nenhuma pista.
     */
    @Test
    fun `os valores batem com o firmware`() {
        assertEquals("REDE_NOME no oculos_camera.ino", "VisuAll-Oculos", EnderecoDosOculos.REDE)
        assertEquals("REDE_SENHA no oculos_camera.ino", "visuall2026", EnderecoDosOculos.SENHA)
        assertEquals("IP_PLACA no oculos_camera.ino", "192.168.4.1", EnderecoDosOculos.IP)
        assertTrue("a senha precisa ter 8+ caracteres, exigencia do WPA2",
            EnderecoDosOculos.SENHA.length >= 8)
    }
}
