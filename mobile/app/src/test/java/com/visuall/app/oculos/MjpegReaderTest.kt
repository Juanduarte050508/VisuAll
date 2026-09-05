package com.visuall.app.oculos

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.EOFException
import java.io.IOException

class MjpegReaderTest {

    private val boundary = "123456789000000000000987654321"

    private fun parte(corpo: ByteArray, comTamanho: Boolean = true): ByteArray {
        val cabecalho = StringBuilder("--$boundary\r\nContent-Type: image/jpeg\r\n")
        if (comTamanho) cabecalho.append("Content-Length: ${corpo.size}\r\n")
        cabecalho.append("\r\n")
        return cabecalho.toString().toByteArray() + corpo + "\r\n".toByteArray()
    }

    private fun leitor(dados: ByteArray) = MjpegReader(ByteArrayInputStream(dados), boundary)

    // ---- contra os bytes REAIS do mock ------------------------------------
    //
    // A fixture foi gravada do proprio oculos/mock_esp32_cam.py, que imita o
    // firmware. Testar so contra bytes que eu mesmo montei provaria apenas que
    // o leitor concorda comigo sobre o formato -- nao que ele concorda com o
    // que vai chegar dos oculos.

    /**
     * O CameraWebServer de fabrica manda a fronteira DEPOIS de cada quadro, e
     * nao antes. E o contrario do que diz o formato multipart -- navegador
     * tolera --, e o nosso firmware nao faz isso: manda antes, como o mock.
     *
     * Mas o mundo esta cheio de firmware derivado do exemplo de fabrica, e o
     * primeiro reflexo de quem trava na bancada e gravar o exemplo pra ver se
     * a placa presta. Se o app engasgasse nesse formato, esse teste inocente
     * apontaria pro lugar errado.
     *
     * Perde o primeiro quadro, e isso e esperado: nao ha fronteira nenhuma
     * antes dele pra o leitor se ancorar. Um quadro em quinze por segundo.
     */
    @Test
    fun `aguenta o formato de fabrica, com a fronteira depois do quadro`() {
        val fixture = javaClass.getResourceAsStream("/mjpeg_mock.bin")!!.readBytes()

        // Tira os JPEGs da fixture (que esta no formato do mock)...
        val originais = mutableListOf<ByteArray>()
        val deOrigem = leitor(fixture)
        while (true) originais.add(deOrigem.proximoQuadro() ?: break)
        assertEquals("a fixture tem 3 quadros", 3, originais.size)

        // ...e remonta na ordem do firmware de fabrica.
        val saida = java.io.ByteArrayOutputStream()
        for (jpeg in originais) {
            saida.write(
                ("Content-Type: image/jpeg\r\n" +
                    "Content-Length: ${jpeg.size}\r\n\r\n").toByteArray())
            saida.write(jpeg)
            saida.write(("\r\n--$boundary\r\n").toByteArray())
        }

        val r = leitor(saida.toByteArray())
        val lidos = mutableListOf<ByteArray>()
        while (true) lidos.add(r.proximoQuadro() ?: break)

        assertEquals("perde so o primeiro, por nao ter fronteira antes dele", 2, lidos.size)
        assertArrayEquals("e o que sobra tem de sair intacto", originais[1], lidos[0])
        assertArrayEquals(originais[2], lidos[1])
    }

    @Test
    fun `le os quadros gravados do mock`() {
        val dados = javaClass.getResourceAsStream("/mjpeg_mock.bin")!!.readBytes()
        val r = MjpegReader(ByteArrayInputStream(dados), boundary)

        var n = 0
        while (true) {
            val quadro = r.proximoQuadro() ?: break
            // Todo JPEG comeca com FF D8 FF e termina com FF D9.
            assertEquals("quadro $n nao comeca como JPEG", 0xFF.toByte(), quadro[0])
            assertEquals("quadro $n nao comeca como JPEG", 0xD8.toByte(), quadro[1])
            assertEquals("quadro $n nao termina como JPEG",
                0xD9.toByte(), quadro[quadro.size - 1])
            assertTrue("quadro $n pequeno demais pra ser uma imagem", quadro.size > 1000)
            n++
        }
        assertEquals("a fixture tem 3 partes", 3, n)
    }

    @Test
    fun `extrai o boundary do content-type que o mock manda`() {
        assertEquals(
            boundary,
            MjpegReader.boundaryDe("multipart/x-mixed-replace; boundary=$boundary")
        )
    }

    // ---- formato ----------------------------------------------------------

    @Test
    fun `le dois quadros seguidos na ordem`() {
        val a = byteArrayOf(1, 2, 3, 4)
        val b = byteArrayOf(9, 8)
        val r = leitor(parte(a) + parte(b))
        assertArrayEquals(a, r.proximoQuadro())
        assertArrayEquals(b, r.proximoQuadro())
        assertNull(r.proximoQuadro())
    }

    @Test
    fun `ignora preambulo antes da primeira fronteira`() {
        // O padrao multipart permite texto antes da primeira fronteira, e
        // servidor nenhum promete que nao vai mandar.
        val a = byteArrayOf(7, 7, 7)
        val r = leitor("qualquer coisa aqui\r\n".toByteArray() + parte(a))
        assertArrayEquals(a, r.proximoQuadro())
    }

    @Test
    fun `fronteira final encerra o stream`() {
        val a = byteArrayOf(1)
        val r = leitor(parte(a) + "--$boundary--\r\n".toByteArray())
        assertNotNull(r.proximoQuadro())
        assertNull("depois de --boundary-- nao ha mais quadro", r.proximoQuadro())
    }

    @Test
    fun `stream vazio devolve null em vez de travar`() {
        assertNull(leitor(ByteArray(0)).proximoQuadro())
    }

    // ---- falhas que precisam ser barulhentas -------------------------------

    @Test(expected = EOFException::class)
    fun `quadro cortado no meio falha em vez de entregar imagem parcial`() {
        // Wi-Fi caindo no meio de um quadro e o caso comum aqui. Entregar meio
        // JPEG faria o decodificador devolver null la na frente, longe da causa.
        val completo = parte(ByteArray(500) { 5 })
        leitor(completo.copyOfRange(0, completo.size - 200)).proximoQuadro()
    }

    @Test
    fun `parte sem content-length diz o que faltou`() {
        val erro = runCatching { leitor(parte(byteArrayOf(1, 2), comTamanho = false)).proximoQuadro() }
            .exceptionOrNull()
        assertTrue("esperava IOException, veio $erro", erro is IOException)
        assertTrue("a mensagem precisa citar Content-Length: ${erro?.message}",
            erro!!.message!!.contains("Content-Length"))
    }

    @Test
    fun `content-length absurdo nao vira alocacao gigante`() {
        val dados = ("--$boundary\r\nContent-Type: image/jpeg\r\n" +
            "Content-Length: 900000000\r\n\r\n").toByteArray()
        val erro = runCatching { leitor(dados).proximoQuadro() }.exceptionOrNull()
        assertTrue("esperava IOException, veio $erro", erro is IOException)
        assertTrue("a mensagem precisa citar o teto: ${erro?.message}",
            erro!!.message!!.contains("teto"))
    }

    // ---- boundaryDe -------------------------------------------------------

    /**
     * O cabecalho EXATO que o nosso firmware manda, copiado de
     * oculos/firmware/oculos_camera/oculos_camera.ino.
     *
     * Sem espaco depois do ponto-e-virgula, ao contrario do mock. As duas
     * formas sao validas, e nenhum teste cobria esta: se o parser exigisse o
     * espaco, tudo passaria aqui, tudo funcionaria com o mock, e a falha so
     * apareceria no dia em que a placa fosse ligada pela primeira vez -- o
     * pior dia possivel pra descobrir.
     */
    @Test
    fun `o cabecalho que a placa manda, sem espaco depois do ponto-e-virgula`() {
        assertEquals(boundary,
            MjpegReader.boundaryDe("multipart/x-mixed-replace;boundary=$boundary"))
    }

    @Test
    fun `boundary entre aspas`() {
        assertEquals("abc", MjpegReader.boundaryDe("multipart/x-mixed-replace; boundary=\"abc\""))
    }

    @Test
    fun `boundary com espacos e maiusculas`() {
        assertEquals("abc", MjpegReader.boundaryDe("Multipart/X-Mixed-Replace;  BOUNDARY=abc "))
    }

    @Test
    fun `resposta que nao e multipart devolve null`() {
        // Sintoma real: o endereco aponta pra uma pagina HTML (erro do roteador,
        // porta errada) em vez do stream. Vale distinguir de "conectou mas nao
        // veio quadro" -- sao problemas diferentes pra quem esta montando.
        assertNull(MjpegReader.boundaryDe("text/html; charset=utf-8"))
        assertNull(MjpegReader.boundaryDe(null))
    }

    @Test
    fun `multipart sem boundary devolve null`() {
        assertNull(MjpegReader.boundaryDe("multipart/x-mixed-replace"))
        assertNull(MjpegReader.boundaryDe("multipart/x-mixed-replace; boundary="))
    }
}
