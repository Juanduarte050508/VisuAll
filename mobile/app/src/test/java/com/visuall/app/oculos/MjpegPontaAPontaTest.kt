package com.visuall.app.oculos

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.net.InetAddress
import java.net.ServerSocket
import java.net.SocketException

/**
 * Ponta a ponta pelo caminho de verdade: socket real, HTTP real, o `abrirHttp`
 * de producao (nao a conexao falsa dos outros testes), o parser, e as
 * dimensoes lidas do proprio JPEG no fim.
 *
 * Os outros testes provam cada peca isolada. Este prova que elas se encaixam --
 * que e onde erro de integracao costuma morar: cabecalho lido errado, boundary
 * extraido do Content-Type real, bytes que passam no parser mas nao formam
 * imagem.
 *
 * O servidor sobe dentro do proprio teste e serve os bytes gravados do
 * oculos/mock_esp32_cam.py, entao nao depende de nada rodando por fora.
 *
 * ServerSocket na mao em vez de com.sun.net.httpserver, e as dimensoes lidas na
 * mao em vez de javax.imageio: teste de unidade de Android compila contra o
 * android.jar, que nao tem as classes de desktop do JDK.
 */
class MjpegPontaAPontaTest {

    private val boundary = "123456789000000000000987654321"

    /** Servidor de um atendimento so: responde a primeira requisicao e fecha. */
    private class ServidorDeTeste(
        private val contentType: String,
        private val corpo: ByteArray
    ) : AutoCloseable {
        private val socket = ServerSocket(0, 1, InetAddress.getByName("127.0.0.1"))
        val porta: Int get() = socket.localPort
        private val thread = Thread({
            try {
                socket.accept().use { cliente ->
                    // Consome a requisicao ate a linha em branco.
                    val entrada = cliente.getInputStream()
                    val pedido = StringBuilder()
                    while (!pedido.endsWith("\r\n\r\n")) {
                        val b = entrada.read()
                        if (b < 0) return@use
                        pedido.append(b.toChar())
                    }
                    val saida = cliente.getOutputStream()
                    saida.write(
                        ("HTTP/1.1 200 OK\r\n" +
                            "Content-Type: $contentType\r\n" +
                            "Connection: close\r\n\r\n").toByteArray())
                    saida.write(corpo)
                    saida.flush()
                }
            } catch (_: SocketException) {
                // socket fechado no fim do teste; esperado
            }
        }, "servidor-de-teste").apply { isDaemon = true; start() }

        override fun close() {
            socket.close()
            thread.join(500)
        }
    }

    /**
     * Largura e altura lidas do cabecalho SOF do JPEG.
     *
     * Serve de prova de que os bytes formam mesmo uma imagem: um quadro cortado
     * pela metade passa reto por uma checagem de tamanho, mas nao tem SOF
     * valido.
     */
    private fun dimensoesJpeg(jpeg: ByteArray): Pair<Int, Int> {
        var i = 2      // pula o SOI (FF D8)
        while (i + 9 < jpeg.size) {
            if (jpeg[i] != 0xFF.toByte()) { i++; continue }
            val marcador = jpeg[i + 1].toInt() and 0xFF
            val ehSof = marcador in 0xC0..0xCF &&
                marcador != 0xC4 && marcador != 0xC8 && marcador != 0xCC
            if (ehSof) {
                val altura = ((jpeg[i + 5].toInt() and 0xFF) shl 8) or (jpeg[i + 6].toInt() and 0xFF)
                val largura = ((jpeg[i + 7].toInt() and 0xFF) shl 8) or (jpeg[i + 8].toInt() and 0xFF)
                return largura to altura
            }
            val tamanho = ((jpeg[i + 2].toInt() and 0xFF) shl 8) or (jpeg[i + 3].toInt() and 0xFF)
            i += 2 + tamanho
        }
        throw AssertionError("nao achei o cabecalho SOF: estes bytes nao sao um JPEG inteiro")
    }

    @Test(timeout = 15_000)
    fun `do socket ate a imagem`() {
        val fixture = javaClass.getResourceAsStream("/mjpeg_mock.bin")!!.readBytes()
        ServidorDeTeste("multipart/x-mixed-replace; boundary=$boundary", fixture).use { servidor ->
            val cliente = MjpegClient("http://127.0.0.1:${servidor.porta}/stream", esperar = {})
            val estados = mutableListOf<String>()
            val imagens = mutableListOf<Pair<Int, Int>>()

            cliente.rodar(aoEstado = { estados.add(it::class.java.simpleName) }) { jpeg ->
                imagens.add(dimensoesJpeg(jpeg))
                // A fixture tem 3 quadros; depois deles o servidor fecha e o
                // cliente tentaria reconectar pra sempre.
                if (imagens.size >= 3) cliente.parar()
            }

            assertEquals("os 3 quadros da fixture", 3, imagens.size)
            assertTrue("todos em 320x240, a resolucao que o firmware vai usar: $imagens",
                imagens.all { it == 320 to 240 })
            assertEquals("Conectando", estados.first())
            assertTrue("precisa ter avisado que passou a receber", estados.contains("Recebendo"))
        }
    }

    /**
     * A conexao sai por onde mandarem, e nao pela rede que o sistema escolher.
     *
     * E esse o mecanismo inteiro da etapa do Wi-Fi sem internet: a
     * RedeDosOculos passa a rede da placa neste mesmo parametro. Se abrirHttp
     * voltasse a usar o caminho padrao, nada quebraria em teste nenhum -- so
     * na rua, com o pedido saindo pelos dados moveis.
     */
    @Test(timeout = 15_000)
    fun `abre a conexao por onde mandarem, nao pelo caminho padrao`() {
        val fixture = javaClass.getResourceAsStream("/mjpeg_mock.bin")!!.readBytes()
        ServidorDeTeste("multipart/x-mixed-replace; boundary=$boundary", fixture).use { servidor ->
            val pedidos = mutableListOf<String>()
            val endereco = "http://127.0.0.1:${servidor.porta}/stream"

            MjpegClient.abrirHttp(endereco) { url ->
                pedidos.add(url.toString())
                url.openConnection()
            }.use { conexao ->
                assertEquals("tinha que ter passado por quem eu dei", 1, pedidos.size)
                assertEquals(endereco, pedidos.first())
                assertTrue("e a conexao aberta ali e a que foi usada: ${conexao.contentType}",
                    conexao.contentType!!.contains("multipart"))
            }
        }
    }

    @Test(timeout = 15_000)
    fun `endereco que responde HTML vira erro que diz o que veio`() {
        // O engano mais comum na montagem: apontar pro endereco errado, ou
        // esquecer o /stream. Responde 200 com uma pagina, e sem esta checagem
        // o app ficaria "conectado" sem nunca mostrar imagem.
        ServidorDeTeste("text/html; charset=utf-8", "<html>qualquer</html>".toByteArray()).use { s ->
            val cliente = MjpegClient("http://127.0.0.1:${s.porta}/", esperar = {})
            val erros = mutableListOf<String>()
            cliente.rodar(aoEstado = { estado ->
                if (estado is MjpegClient.Estado.Erro) {
                    erros.add(estado.motivo)
                    cliente.parar()
                }
            }) {}

            assertTrue("precisa reclamar citando o content-type: $erros",
                erros.first().contains("text/html"))
        }
    }

    @Test(timeout = 15_000)
    fun `placa ainda fora do ar vira nova tentativa, nao desistencia`() {
        // A placa demora a subir o Wi-Fi depois de ligar. O app tem que ficar
        // tentando em vez de desistir na primeira recusa.
        var tentativas = 0
        lateinit var cliente: MjpegClient
        // porta 1 em 127.0.0.1 recusa a conexao na hora, sem esperar timeout.
        cliente = MjpegClient("http://127.0.0.1:1/stream", esperar = {})
        val erros = mutableListOf<String>()

        cliente.rodar(aoEstado = { estado ->
            if (estado is MjpegClient.Estado.Erro) {
                erros.add(estado.motivo)
                if (++tentativas >= 3) cliente.parar()
            }
        }) {}

        assertEquals("tem que ter tentado 3 vezes antes de eu mandar parar", 3, erros.size)
    }
}
