package com.visuall.app.oculos

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.util.Log
import java.io.IOException
import java.net.URL

/**
 * Faz o stream sair pelo Wi-Fi dos oculos, mesmo que ele nao tenha internet.
 *
 * O problema que isto resolve nao aparece enquanto se testa em casa. Os oculos
 * criam a propria rede e nao tem internet nenhuma pra oferecer -- nao ha de
 * onde tirar. O Android percebe isso, conclui que a rede nao presta e volta
 * sozinho pros dados moveis. O Wi-Fi continua conectado na tela, o celular
 * continua dizendo que esta na rede dos oculos, e o pedido do app sai pela
 * operadora: 192.168.4.1 nao existe na internet, e a conexao morre sem nunca
 * ter chegado perto da placa.
 *
 * A saida e pedir a rede explicitamente, dizendo que internet nao e requisito,
 * e mandar o stream POR ELA em vez de "pela rede do celular".
 *
 * Por que nao `bindProcessToNetwork`: aquilo joga o app INTEIRO na rede sem
 * internet -- a sintese de voz, qualquer coisa que precise sair. Aqui so a
 * conexao do stream muda de caminho, que e a unica que precisa.
 */
internal class RedeDosOculos(context: Context) {

    private val gerente = context.applicationContext
        .getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager

    private val trava = Object()

    /** A rede escolhida, ou null enquanto o sistema nao entregou nenhuma. */
    @Volatile
    private var rede: Network? = null

    private var inscricao: ConnectivityManager.NetworkCallback? = null

    fun ligar() {
        if (inscricao != null) return

        val pedido = NetworkRequest.Builder()
            .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
            // A linha que faz a diferenca. Com internet como requisito, o
            // pedido nunca casa com os oculos. Sem ela como requisito, casa
            // com qualquer Wi-Fi -- com ou sem internet -- que e o que
            // queremos: o mesmo codigo serve pro mock em casa e pra placa.
            .removeCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
            .build()

        val cb = object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) {
                Log.i(TAG, "usando a rede $network para o stream")
                synchronized(trava) {
                    rede = network
                    trava.notifyAll()
                }
            }

            override fun onLost(network: Network) {
                if (rede == network) {
                    Log.i(TAG, "perdi a rede $network")
                    rede = null
                }
            }

            override fun onUnavailable() {
                Log.w(TAG, "o sistema nao encontrou nenhum Wi-Fi")
            }
        }

        try {
            // requestNetwork, e nao registerNetworkCallback: o primeiro diz ao
            // sistema que este app QUER a rede no ar. So observar nao impede o
            // Android de derrubar um Wi-Fi que ele julgou inutil.
            gerente.requestNetwork(pedido, cb)
            inscricao = cb
            Log.i(TAG, "pedido registrado: $pedido")
        } catch (erro: SecurityException) {
            // Falta CHANGE_NETWORK_STATE no manifesto. Sem a rede pedida o
            // stream ainda funciona onde houver internet (o caso do mock), so
            // nao funciona com a placa -- entao registrar e seguir.
            Log.e(TAG, "sem permissao pra pedir a rede", erro)
        }
    }

    fun desligar() {
        inscricao?.let {
            // Sem isto o pedido continua de pe depois de sair dos oculos, e o
            // sistema segue segurando um Wi-Fi sem internet por nossa causa.
            runCatching { gerente.unregisterNetworkCallback(it) }
                .onFailure { erro -> Log.w(TAG, "falha ao soltar a rede", erro) }
        }
        inscricao = null
        rede = null
    }

    /**
     * Abre o stream pela rede dos oculos.
     *
     * Enquanto o sistema nao entregou a rede, levanta em vez de sair pelo
     * caminho padrao: cair no padrao e exatamente o defeito que esta classe
     * existe pra impedir, e falharia longe daqui, como "ninguem atende". O
     * laco do [MjpegClient] tenta de novo sozinho, e essa espera dura o tempo
     * de o Wi-Fi conectar.
     */
    fun abrir(url: String): MjpegClient.Conexao {
        val n = esperaRede() ?: throw IOException(SEM_REDE)
        return MjpegClient.abrirHttp(url) { endereco: URL -> n.openConnection(endereco) }
    }

    /**
     * Espera um pouco pela rede antes de desistir.
     *
     * Medido no aparelho: entre registrar o pedido e o sistema entregar a rede
     * passaram 11 milissegundos. Sem esta espera, a primeira tentativa falha
     * nesse vao e a pessoa leva um aviso de "conecte o celular ao Wi-Fi" toda
     * vez que liga os oculos -- inclusive quando deu tudo certo, porque a
     * imagem aparece logo depois. Aviso que aparece quando esta tudo bem
     * ensina a ignorar aviso.
     *
     * Roda na thread de rede, nunca na da interface.
     */
    private fun esperaRede(): Network? = synchronized(trava) {
        val limite = System.currentTimeMillis() + PRAZO_REDE_MS
        while (rede == null) {
            val resta = limite - System.currentTimeMillis()
            if (resta <= 0L) break
            trava.wait(resta)
        }
        rede
    }

    companion object {
        private const val TAG = "RedeDosOculos"

        /**
         * Quanto esperar pelo sistema entregar a rede. Generoso porque o custo
         * de errar pra baixo e um aviso falso, e pra cima e so demorar mais
         * pra avisar de um problema que e real (celular fora do Wi-Fi).
         */
        private const val PRAZO_REDE_MS = 3_000L

        /** Reconhecido pela [MensagemDeErro]; mudar aqui muda o aviso na tela. */
        const val SEM_REDE = "ainda nao achei o Wi-Fi dos oculos"
    }
}
