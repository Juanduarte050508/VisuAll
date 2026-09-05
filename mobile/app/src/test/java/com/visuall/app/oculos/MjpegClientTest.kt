package com.visuall.app.oculos

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.IOException
import java.io.InputStream

/**
 * Fixa o laco de reconexao. E a parte que decide se o app aguenta o Wi-Fi dos
 * oculos cair -- e cair e o caso normal, nao a excecao.
 */
class MjpegClientTest {

    private val boundary = "b"

    private fun parte(corpo: ByteArray): ByteArray =
        ("--$boundary\r\nContent-Type: image/jpeg\r\nContent-Length: ${corpo.size}\r\n\r\n")
            .toByteArray() + corpo + "\r\n".toByteArray()

    private class ConexaoFalsa(
        override val contentType: String?,
        dados: ByteArray
    ) : MjpegClient.Conexao {
        override val entrada: InputStream = ByteArrayInputStream(dados)
        var fechada = false
        override fun close() { fechada = true }
    }

    /** Entrega as conexoes na ordem e para o cliente quando a lista acaba. */
    private fun roteiro(vararg conexoes: () -> MjpegClient.Conexao): Pair<MjpegClient, MutableList<Long>> {
        val esperas = mutableListOf<Long>()
        var i = 0
        lateinit var cliente: MjpegClient
        cliente = MjpegClient(
            url = "http://oculos/stream",
            abrirConexao = {
                if (i >= conexoes.size) {
                    cliente.parar()
                    throw IOException("fim do roteiro")
                }
                conexoes[i++]()
            },
            esperar = { esperas.add(it) }
        )
        return cliente to esperas
    }

    @Test
    fun `entrega os quadros de uma conexao boa`() {
        val (cliente, _) = roteiro({
            ConexaoFalsa("multipart/x-mixed-replace; boundary=$boundary",
                parte(byteArrayOf(1, 2)) + parte(byteArrayOf(3)))
        })
        val recebidos = mutableListOf<ByteArray>()
        cliente.rodar(aoQuadro = { recebidos.add(it) })

        assertEquals(2, recebidos.size)
        assertEquals(listOf<Byte>(1, 2), recebidos[0].toList())
        assertEquals(listOf<Byte>(3), recebidos[1].toList())
    }

    @Test
    fun `reconecta quando o stream cai no meio e continua entregando`() {
        // O caso que motiva a classe inteira: a placa some por um instante.
        val (cliente, _) = roteiro(
            { ConexaoFalsa("multipart/x-mixed-replace; boundary=$boundary", parte(byteArrayOf(1))) },
            { ConexaoFalsa("multipart/x-mixed-replace; boundary=$boundary", parte(byteArrayOf(2))) }
        )
        val recebidos = mutableListOf<Byte>()
        cliente.rodar(aoQuadro = { recebidos.add(it[0]) })

        assertEquals("os quadros das duas conexoes precisam chegar",
            listOf<Byte>(1, 2), recebidos)
    }

    @Test
    fun `espera dobra entre tentativas e para no teto`() {
        // Sem isto, uma placa desligada vira laco de reconexao a mil por
        // segundo -- gasta bateria do celular e polui o log.
        val esperas = mutableListOf<Long>()
        var tentativas = 0
        lateinit var cliente: MjpegClient
        cliente = MjpegClient(
            url = "http://oculos/stream",
            abrirConexao = {
                tentativas++
                if (tentativas >= 12) cliente.parar()
                throw IOException("recusada")
            },
            esperar = { esperas.add(it) }
        )
        cliente.rodar {}

        assertEquals("comeca no valor inicial", MjpegClient.ESPERA_INICIAL_MS, esperas[0])
        assertEquals("dobra", MjpegClient.ESPERA_INICIAL_MS * 2, esperas[1])
        assertEquals("dobra de novo", MjpegClient.ESPERA_INICIAL_MS * 4, esperas[2])
        assertTrue("nunca passa do teto", esperas.all { it <= MjpegClient.ESPERA_MAXIMA_MS })
        assertEquals("chega no teto e fica", MjpegClient.ESPERA_MAXIMA_MS, esperas.last())
    }

    @Test
    fun `um quadro recebido zera a espera`() {
        // Uma queda depois de funcionar merece nova tentativa rapida; nao faz
        // sentido herdar a espera longa de uma falha anterior.
        val esperas = mutableListOf<Long>()
        var i = 0
        lateinit var cliente: MjpegClient
        cliente = MjpegClient(
            url = "http://oculos/stream",
            abrirConexao = {
                i++
                when (i) {
                    1, 2 -> throw IOException("recusada")        // espera cresce
                    3 -> ConexaoFalsa("multipart/x-mixed-replace; boundary=$boundary",
                        parte(byteArrayOf(9)))                   // funcionou
                    else -> { cliente.parar(); throw IOException("fim") }
                }
            },
            esperar = { esperas.add(it) }
        )
        cliente.rodar {}

        // 2 falhas + a queda depois do quadro. A 4a tentativa chama parar(), e
        // ai o laco sai sem dormir -- garantido pelo teste do `parar`.
        assertEquals(3, esperas.size)
        assertEquals(MjpegClient.ESPERA_INICIAL_MS, esperas[0])
        assertEquals(MjpegClient.ESPERA_INICIAL_MS * 2, esperas[1])
        assertEquals("depois de um quadro de verdade a espera volta ao inicio",
            MjpegClient.ESPERA_INICIAL_MS, esperas[2])
    }

    @Test
    fun `conexao que abre e cai sem quadro nenhum nao zera a espera`() {
        // Servidor que aceita e derruba na hora e o pior caso pro backoff: se
        // conectar ja zerasse, isto viraria laco apertado.
        val esperas = mutableListOf<Long>()
        var i = 0
        lateinit var cliente: MjpegClient
        cliente = MjpegClient(
            url = "http://oculos/stream",
            abrirConexao = {
                i++
                if (i > 3) { cliente.parar(); throw IOException("fim") }
                ConexaoFalsa("multipart/x-mixed-replace; boundary=$boundary", ByteArray(0))
            },
            esperar = { esperas.add(it) }
        )
        cliente.rodar {}

        assertEquals(MjpegClient.ESPERA_INICIAL_MS, esperas[0])
        assertEquals("sem quadro, a espera tem que crescer",
            MjpegClient.ESPERA_INICIAL_MS * 2, esperas[1])
    }

    @Test
    fun `resposta que nao e MJPEG vira erro que diz o content-type`() {
        // Endereco errado costuma responder 200 com HTML. A mensagem precisa
        // apontar pra isso, senao vira caca ao fantasma no dia da montagem.
        val estados = mutableListOf<MjpegClient.Estado>()
        var i = 0
        lateinit var cliente: MjpegClient
        cliente = MjpegClient(
            url = "http://oculos/stream",
            abrirConexao = {
                i++
                if (i > 1) { cliente.parar(); throw IOException("fim") }
                ConexaoFalsa("text/html; charset=utf-8", "<html>".toByteArray())
            },
            esperar = {}
        )
        cliente.rodar(aoEstado = { estados.add(it) }) {}

        val erro = estados.filterIsInstance<MjpegClient.Estado.Erro>().first()
        assertTrue("a mensagem precisa citar o content-type: ${erro.motivo}",
            erro.motivo.contains("text/html"))
    }

    @Test
    fun `avisa conectando antes e recebendo depois`() {
        val estados = mutableListOf<String>()
        val (cliente, _) = roteiro({
            ConexaoFalsa("multipart/x-mixed-replace; boundary=$boundary", parte(byteArrayOf(1)))
        })
        cliente.rodar(aoEstado = { estados.add(it::class.java.simpleName) }) {}

        assertEquals("Conectando", estados.first())
        assertTrue("precisa avisar que passou a receber", estados.contains("Recebendo"))
    }

    @Test
    fun `parar interrompe o laco sem esperar mais nada`() {
        val esperas = mutableListOf<Long>()
        lateinit var cliente: MjpegClient
        cliente = MjpegClient(
            url = "http://oculos/stream",
            abrirConexao = {
                ConexaoFalsa("multipart/x-mixed-replace; boundary=$boundary",
                    parte(byteArrayOf(1)) + parte(byteArrayOf(2)))
            },
            esperar = { esperas.add(it) }
        )
        var n = 0
        cliente.rodar { n++; cliente.parar() }

        assertEquals("para no primeiro quadro", 1, n)
        assertTrue("nao pode dormir depois de parar", esperas.isEmpty())
    }

    @Test
    fun `fecha a conexao mesmo quando o stream cai`() {
        // Sem isto cada reconexao vaza um socket, e o vazamento so aparece
        // depois de horas de uso.
        var criada: ConexaoFalsa? = null
        var i = 0
        lateinit var cliente: MjpegClient
        cliente = MjpegClient(
            url = "http://oculos/stream",
            abrirConexao = {
                i++
                if (i > 1) { cliente.parar(); throw IOException("fim") }
                ConexaoFalsa("multipart/x-mixed-replace; boundary=$boundary",
                    parte(byteArrayOf(1))).also { criada = it }
            },
            esperar = {}
        )
        cliente.rodar {}

        assertTrue("a conexao precisa ter sido fechada", criada!!.fechada)
    }
}
