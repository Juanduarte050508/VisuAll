package com.visuall.app.oculos

import java.io.EOFException
import java.io.IOException
import java.io.InputStream

/**
 * Le um stream MJPEG e devolve um JPEG por vez.
 *
 * O ESP32 (e o mock em oculos/mock_esp32_cam.py, que imita o mesmo firmware)
 * responde com `multipart/x-mixed-replace`: uma conexao HTTP que nunca fecha e
 * vai empurrando partes separadas por uma linha de fronteira. Cada parte e:
 *
 *     --<boundary>\r\n
 *     Content-Type: image/jpeg\r\n
 *     Content-Length: 5304\r\n
 *     \r\n
 *     <5304 bytes de JPEG>
 *     \r\n
 *
 * Esta classe so entende o formato. Quem abre a conexao, decodifica o JPEG e
 * entrega pro reconhecimento e o [NetworkStreamSource] -- separado de proposito,
 * porque o formato da pra testar na JVM com bytes de mentira, e conexao e
 * Bitmap nao dao.
 */
internal class MjpegReader(
    entrada: InputStream,
    private val boundary: String,
    /**
     * Teto de sanidade pro Content-Length. Um quadro 320x240 do ESP32 tem uns
     * 5-15 KB; se vier um numero absurdo, e stream corrompido ou nao e MJPEG --
     * melhor falhar do que tentar alocar o que o cabecalho pedir.
     */
    private val tamanhoMaximo: Int = 2 * 1024 * 1024
) {
    private val entrada = entrada.buffered()
    private var achouPrimeiraFronteira = false

    /**
     * Devolve o proximo JPEG, ou null quando o stream acabou.
     *
     * Bloqueia ate ter um quadro completo: e pra ser chamado numa thread de
     * fundo, nunca na thread da interface.
     */
    fun proximoQuadro(): ByteArray? {
        if (!avancaAteFronteira()) return null

        var tamanho = -1
        while (true) {
            val linha = leLinha() ?: return null
            // Linha em branco encerra os cabecalhos desta parte.
            if (linha.isEmpty()) break
            val i = linha.indexOf(':')
            if (i > 0 && linha.substring(0, i).trim().equals("Content-Length", true)) {
                tamanho = linha.substring(i + 1).trim().toIntOrNull() ?: -1
            }
        }

        // Tanto o firmware CameraWebServer quanto o mock mandam Content-Length.
        // Sem ele so restaria varrer os bytes atras da proxima fronteira, e um
        // JPEG pode conter a fronteira por acaso -- prefiro falhar dizendo o
        // que houve a decodificar um quadro cortado em silencio.
        if (tamanho <= 0) {
            throw IOException("parte MJPEG sem Content-Length utilizavel (veio $tamanho)")
        }
        if (tamanho > tamanhoMaximo) {
            throw IOException("Content-Length de $tamanho bytes acima do teto de $tamanhoMaximo")
        }

        val jpeg = ByteArray(tamanho)
        var lidos = 0
        while (lidos < tamanho) {
            val n = entrada.read(jpeg, lidos, tamanho - lidos)
            if (n < 0) throw EOFException("stream cortado no meio de um quadro")
            lidos += n
        }
        return jpeg
    }

    /**
     * Consome tudo ate passar de uma linha de fronteira.
     *
     * Ate a primeira fronteira pode vir preambulo (o padrao permite), e o
     * proprio servidor manda um \r\n depois de cada JPEG -- por isso isto
     * ignora qualquer coisa que nao seja a fronteira em vez de exigir que ela
     * seja a proxima linha.
     */
    private fun avancaAteFronteira(): Boolean {
        val marca = "--$boundary"
        while (true) {
            val linha = leLinha() ?: return false
            if (linha.startsWith(marca)) {
                achouPrimeiraFronteira = true
                // "--boundary--" e a fronteira final: o servidor encerrou.
                return !linha.startsWith("$marca--")
            }
            // Antes da primeira fronteira, lixo e esperado. Depois dela, uma
            // linha estranha ainda pode ser o \r\n de fecho do quadro anterior,
            // entao seguir em frente e o comportamento certo nos dois casos.
        }
    }

    /**
     * Uma linha de cabecalho, sem o \r\n. Null no fim do stream.
     *
     * Byte a byte de proposito: depois dos cabecalhos vem binario, e um leitor
     * de texto com buffer proprio engoliria parte do JPEG junto.
     */
    private fun leLinha(): String? {
        val buffer = StringBuilder()
        while (true) {
            val b = entrada.read()
            if (b < 0) return if (buffer.isEmpty()) null else buffer.toString()
            if (b == '\n'.code) {
                if (buffer.isNotEmpty() && buffer.last() == '\r') buffer.setLength(buffer.length - 1)
                return buffer.toString()
            }
            buffer.append(b.toChar())
        }
    }

    companion object {
        /**
         * Tira o boundary do cabecalho Content-Type da resposta, por exemplo
         * `multipart/x-mixed-replace; boundary=123456789000000000000987654321`.
         *
         * Devolve null quando a resposta nao e multipart -- que na pratica quer
         * dizer que o endereco nao aponta pro stream (uma pagina HTML de erro
         * do roteador, por exemplo). Vale distinguir isso de "conectou mas nao
         * veio quadro": sao problemas diferentes pra quem esta montando.
         */
        fun boundaryDe(contentType: String?): String? {
            if (contentType == null) return null
            if (!contentType.contains("multipart/", ignoreCase = true)) return null
            for (parte in contentType.split(';')) {
                val p = parte.trim()
                if (p.startsWith("boundary=", ignoreCase = true)) {
                    return p.substring("boundary=".length).trim().trim('"')
                        .ifEmpty { null }
                }
            }
            return null
        }
    }
}
