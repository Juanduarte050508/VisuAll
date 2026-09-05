package com.visuall.app.oculos

import java.io.Closeable
import java.io.IOException
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLConnection

/**
 * Mantem uma conexao com o stream dos oculos e entrega os quadros, reconectando
 * sozinho quando cai.
 *
 * Nao sabe o que e um Bitmap nem o que e Android: entrega bytes de JPEG. Isso e
 * de proposito -- assim o laco de reconexao, que e a parte com regra de
 * verdade, roda em teste de JVM. Quem transforma em imagem e entrega pro
 * reconhecimento e o [NetworkStreamSource].
 *
 * Reconectar nao e refinamento: o Wi-Fi de uma placa embarcada numa armacao de
 * oculos cai por bateria, distancia e calor. Sem isto, a primeira queda
 * encerraria a sessao e a pessoa teria de reabrir o app.
 */
internal class MjpegClient(
    private val url: String,
    // Lambda, e nao ::abrirHttp: com o parametro de onde a conexao sai, a
    // referencia ao metodo deixou de ser de um tipo so.
    private val abrirConexao: (String) -> Conexao = { abrirHttp(it) },
    /** Injetavel pra o teste nao dormir de verdade. */
    private val esperar: (Long) -> Unit = { Thread.sleep(it) }
) {
    /** O minimo que o laco precisa de uma conexao; o teste implementa com bytes de mentira. */
    interface Conexao : Closeable {
        val contentType: String?
        val entrada: InputStream
    }

    sealed interface Estado {
        /** Tentando abrir a conexao. */
        object Conectando : Estado
        /** Quadros chegando. */
        object Recebendo : Estado
        /** Caiu ou nem abriu; o laco vai tentar de novo sozinho. */
        data class Erro(val motivo: String) : Estado
    }

    @Volatile
    private var rodando = false

    /** Faz [rodar] devolver. Pode ser chamado de qualquer thread. */
    fun parar() {
        rodando = false
    }

    /**
     * Fica recebendo ate alguem chamar [parar]. BLOQUEIA -- chame numa thread
     * de fundo.
     *
     * [aoQuadro] roda na mesma thread, um quadro por vez: se ele demorar mais
     * que o intervalo entre quadros, a fila do TCP segura o resto e o atraso
     * aparece na tela. Quem chama decide se descarta quadro atrasado.
     */
    fun rodar(aoEstado: (Estado) -> Unit = {}, aoQuadro: (ByteArray) -> Unit) {
        rodando = true
        var espera = ESPERA_INICIAL_MS

        while (rodando) {
            try {
                aoEstado(Estado.Conectando)
                abrirConexao(url).use { conexao ->
                    val boundary = MjpegReader.boundaryDe(conexao.contentType)
                    // Sintoma real de endereco errado: responde 200 com uma
                    // pagina HTML. Dizer "nao e MJPEG" e mostrar o Content-Type
                    // aponta pro erro; "nao veio quadro" mandaria procurar no
                    // lugar errado.
                        ?: throw IOException(
                            "resposta nao e MJPEG (Content-Type: ${conexao.contentType})")

                    val leitor = MjpegReader(conexao.entrada, boundary)
                    aoEstado(Estado.Recebendo)

                    var recebeuAlgo = false
                    while (rodando) {
                        val quadro = leitor.proximoQuadro() ?: break
                        if (!recebeuAlgo) {
                            recebeuAlgo = true
                            // So zera a espera depois de um quadro DE VERDADE.
                            // Zerar ao conectar faria um servidor que aceita e
                            // derruba na hora virar laco apertado de reconexao.
                            espera = ESPERA_INICIAL_MS
                        }
                        aoQuadro(quadro)
                    }
                }
            } catch (erro: Exception) {
                if (!rodando) break
                aoEstado(Estado.Erro(erro.message ?: erro::class.java.simpleName))
            }

            if (!rodando) break
            esperar(espera)
            espera = (espera * 2).coerceAtMost(ESPERA_MAXIMA_MS)
        }
    }

    companion object {
        const val ESPERA_INICIAL_MS = 500L
        const val ESPERA_MAXIMA_MS = 5_000L

        /**
         * Timeout de leitura. Precisa ser bem maior que o intervalo entre
         * quadros (15 quadros/s = 67ms) pra nao matar um stream que so
         * engasgou, e curto o bastante pra perceber que o link morreu em vez de
         * ficar pendurado pra sempre.
         */
        const val TIMEOUT_LEITURA_MS = 5_000
        const val TIMEOUT_CONEXAO_MS = 3_000

        private class ConexaoHttp(private val http: HttpURLConnection) : Conexao {
            override val contentType: String? get() = http.contentType
            override val entrada: InputStream get() = http.inputStream
            override fun close() = http.disconnect()
        }

        /**
         * @param conectar de onde a conexao sai. O padrao usa a rede que o
         *   sistema escolher; a [RedeDosOculos] passa a rede dos oculos aqui,
         *   que e o que impede o pedido de sair pelos dados moveis quando o
         *   Wi-Fi da placa nao tem internet. Fica como parametro, e nao como
         *   um `if` la dentro, pra que o resto -- timeouts, codigo HTTP,
         *   cabecalhos -- exista uma vez so pros dois caminhos.
         */
        fun abrirHttp(
            url: String,
            conectar: (URL) -> URLConnection = { it.openConnection() }
        ): Conexao {
            val http = (conectar(URL(url)) as HttpURLConnection).apply {
                connectTimeout = TIMEOUT_CONEXAO_MS
                readTimeout = TIMEOUT_LEITURA_MS
                // O stream nao acaba nunca: manter viva uma conexao dessas no
                // pool depois de fechada nao ajuda em nada.
                setRequestProperty("Connection", "close")
            }
            val codigo = http.responseCode
            if (codigo != HttpURLConnection.HTTP_OK) {
                http.disconnect()
                throw IOException("o stream respondeu HTTP $codigo")
            }
            return ConexaoHttp(http)
        }
    }
}
