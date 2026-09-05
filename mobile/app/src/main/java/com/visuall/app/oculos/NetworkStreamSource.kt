package com.visuall.app.oculos

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log

/**
 * A camera dos oculos, do ponto de vista do app.
 *
 * Faz o que o CameraX faz pela camera do celular: entrega Bitmap um atras do
 * outro, ate mandarem parar. Quem recebe (o LibrasFragment) nao precisa saber
 * que veio de Wi-Fi -- entrega no mesmo `analisarQuadro` do LibrasAnalyzer que
 * a camera usa.
 *
 * Duas threads, e a divisao importa:
 *
 *   rede     -- fica no [MjpegClient], recebendo JPEG e jogando na caixa. Nunca
 *               espera pelo reconhecimento.
 *   analise  -- tira da caixa, decodifica e chama de volta.
 *
 * Se fossem a mesma thread, um reconhecimento lento seguraria a leitura do
 * socket, os quadros se acumulariam no caminho e a imagem apareceria cada vez
 * mais atrasada -- um atraso que so cresce, porque nada faz ele diminuir
 * depois. Com a [UltimoQuadro] no meio, quadro que nao deu tempo e descartado e
 * a imagem continua sendo a de agora.
 */
internal class NetworkStreamSource(
    context: Context,
    private val url: String,
    /**
     * Rede dos oculos. Null (o caso do mock no PC) usa o Wi-Fi em que o
     * celular ja esta. Ver [EnderecoDosOculos.ehAPlaca].
     */
    ssidOculos: String? = null,
    senhaOculos: String? = null,
    /** Chamado na thread de analise, um quadro por vez. Recebe a posse do Bitmap. */
    private val aoQuadro: (Bitmap) -> Unit,
    private val aoEstado: (MjpegClient.Estado) -> Unit = {}
) {
    private val caixa = UltimoQuadro<ByteArray>()

    // O stream sai POR ESTA rede, nao pela que o sistema achar melhor. Ver
    // RedeDosOculos: o Wi-Fi da placa nao tem internet, e o padrao do Android
    // e abandonar uma rede assim.
    private val rede = RedeDosOculos(context, ssidOculos, senhaOculos)
    private val cliente = MjpegClient(url, abrirConexao = rede::abrir)
    private var threadRede: Thread? = null
    private var threadAnalise: Thread? = null

    /** Quantos quadros foram descartados por o reconhecimento nao dar conta. */
    @Volatile
    var descartados = 0L
        private set

    /** Quantos JPEGs chegaram quebrados (Wi-Fi ruim costuma produzir alguns). */
    @Volatile
    var corrompidos = 0L
        private set

    fun iniciar() {
        if (threadRede != null) return
        // Antes das threads: o pedido leva um instante pra ser atendido, e o
        // laco do cliente ja tenta de novo enquanto isso.
        rede.ligar()

        threadAnalise = Thread({
            while (true) {
                val jpeg = caixa.consumir() ?: break
                // decodeByteArray devolve null em vez de lancar quando os bytes
                // nao formam uma imagem. Acontece de verdade com Wi-Fi ruim:
                // pular o quadro e certo, mas contar quantos foram e o que
                // permite dizer depois se a antena esta ruim.
                val bitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)
                if (bitmap == null) {
                    corrompidos++
                    continue
                }
                try {
                    aoQuadro(bitmap)
                } catch (erro: Throwable) {
                    // Um quadro que quebra o reconhecimento nao pode derrubar o
                    // stream inteiro: o proximo pode estar bom.
                    Log.e(TAG, "falha ao analisar quadro dos oculos", erro)
                }
            }
        }, "oculos-analise").apply { start() }

        threadRede = Thread({
            cliente.rodar(aoEstado = aoEstado) { jpeg ->
                if (caixa.publicar(jpeg)) descartados++
            }
        }, "oculos-rede").apply { start() }
    }

    fun parar() {
        cliente.parar()
        caixa.fechar()
        // join com prazo: a thread de rede pode estar parada num read com
        // timeout, e travar a interface esperando por ela seria pior que
        // deixa-la terminar sozinha.
        threadRede?.join(PRAZO_PARADA_MS)
        threadAnalise?.join(PRAZO_PARADA_MS)
        threadRede = null
        threadAnalise = null
        // Depois dos joins: enquanto as threads terminam, a rede ainda precisa
        // estar de pe.
        rede.desligar()
    }

    companion object {
        private const val TAG = "NetworkStreamSource"
        private const val PRAZO_PARADA_MS = 1_000L
    }
}
