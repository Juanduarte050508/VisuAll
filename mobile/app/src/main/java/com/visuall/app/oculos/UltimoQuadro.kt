package com.visuall.app.oculos

/**
 * Caixa de UM quadro so: quem publica nunca espera, quem consome sempre pega o
 * mais novo, e o que ficou pra tras e descartado.
 *
 * Por que nao uma fila: os oculos mandam quadro num ritmo proprio e o
 * reconhecimento demora o que demorar. Se o que sobra fosse enfileirado, num
 * momento de lentidao a fila cresceria e a pessoa veria o gesto de dois
 * segundos atras -- pior que perder quadro, porque o atraso nunca se recupera
 * sozinho. Quadro velho de camera ao vivo nao vale nada; o certo e jogar fora.
 *
 * E a mesma escolha que o CameraX faz com STRATEGY_KEEP_ONLY_LATEST, aqui pra
 * fonte de rede.
 *
 * Guarda os BYTES do JPEG, nao o Bitmap: descartar bytes e de graca, enquanto
 * descartar Bitmap exigiria reciclar na mao. Decodificar fica com quem consome.
 */
internal class UltimoQuadro<T> {

    private val trava = Object()
    private var valor: T? = null
    private var fechado = false

    /**
     * Guarda [novo], jogando fora o que ainda nao tinha sido consumido.
     * Nunca bloqueia.
     *
     * @return true se algo foi descartado -- util pra medir o quanto o
     *   reconhecimento esta atras da camera.
     */
    fun publicar(novo: T): Boolean = synchronized(trava) {
        if (fechado) return false
        val descartou = valor != null
        valor = novo
        (trava as Object).notifyAll()
        descartou
    }

    /**
     * Espera ate ter quadro e devolve o mais novo. Null quando [fechar] e
     * chamado -- que e o sinal de parada pra quem consome.
     */
    fun consumir(): T? = synchronized(trava) {
        while (valor == null && !fechado) {
            (trava as Object).wait()
        }
        val v = valor
        valor = null
        v
    }

    /** Acorda quem estiver esperando e faz [consumir] devolver null pra sempre. */
    fun fechar() = synchronized(trava) {
        fechado = true
        valor = null
        (trava as Object).notifyAll()
    }
}
