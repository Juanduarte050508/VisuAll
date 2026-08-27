package com.visuall.app.libras

import kotlin.math.abs
import kotlin.math.hypot

// Decide quando a "mão aberta" limpa a frase.
//
// Duas tentativas anteriores falharam, e vale registrar por quê:
//
//  1. Só "mão aberta". Qualquer sinal feito de palma aberta (AJUDAR) caía no
//     contador de limpar e nunca era classificado.
//  2. "Mão aberta + movimento baixo", usando o mesmo bodyMotion() da captura.
//     Não resolveu: bodyMotion() é o desvio padrão sobre 5 quadros, ou seja
//     TREMOR entre quadros vizinhos. Um gesto lento como o AJUDAR tem tremor
//     baixo (mediana medida: 0.0267) mesmo atravessando a tela inteira, então
//     a barra continuava enchendo durante o sinal.
//
//  3. "Mao aberta + deslocamento a partir de uma ancora". Resolveu o sinal que
//     ATRAVESSA a tela, mas nao o AJUDAR: nele a mao aberta e a mao de APOIO,
//     que fica praticamente no mesmo ponto enquanto a outra trabalha. O pulso
//     nao percorre os 0.06 exigidos, e a barra voltava a encher.
//
// O que finalmente separa os dois e uma pergunta que o app ja sabe responder:
// TEM UM SINAL SENDO GRAVADO AGORA? A maquina de captura do BodyGestureEngine
// so entra em CAPTURANDO com movimento acima de BODY_START_MOTION por
// BODY_START_FRAMES quadros -- exatamente o que um sinal faz e o que segurar a
// mao parada nao faz. Enquanto ela estiver gravando, isto aqui nao conta.
//
// O deslocamento continua como segunda trava, pra cobrir o intervalo entre o
// comeco do movimento e a captura de fato engatar.
// O que de fato separa os dois casos é DESLOCAMENTO, não tremor: pra limpar, a
// mão fica parada NO MESMO LUGAR por 3 segundos; num sinal, ela viaja. Este
// portão guarda onde a mão estava quando a contagem começou e desiste assim que
// ela se afasta desse ponto.
internal class ClearGestureGate(
    private val duracaoMs: Long = LibrasAnalyzer.TEMPO_PRA_LIMPAR_CORPO,
    private val esperaEntreLimpezasMs: Long = LibrasAnalyzer.ESPERA_ENTRE_LIMPEZAS_MS,
    private val deslocamentoMaximo: Float = LibrasAnalyzer.LIMPAR_DESLOCAMENTO_MAXIMO
) {
    private var abertaDesde: Long? = null
    private var ancoraX = 0f
    private var ancoraY = 0f
    private var ultimaLimpeza: Long? = null

    /** progresso de 0 a 1 pra barra na tela; [limpar] só é true no instante em que dispara. */
    data class Estado(val progresso: Float, val limpar: Boolean)

    /**
     * [x] e [y] são a posição da mão (0..1 no quadro) — usamos o pulso, que é o
     * ponto que menos se mexe quando só os dedos mudam de forma.
     */
    fun avaliar(
        maoAberta: Boolean,
        x: Float,
        y: Float,
        agora: Long,
        gestoEmAndamento: Boolean = false
    ): Estado {
        // Um sinal esta sendo gravado: seja qual for o formato da mao, a
        // pessoa esta sinalizando, nao pedindo pra limpar.
        if (!maoAberta || gestoEmAndamento) {
            abertaDesde = null
            return Estado(0f, false)
        }

        val inicio = abertaDesde
        if (inicio == null) {
            abertaDesde = agora
            ancoraX = x
            ancoraY = y
            return Estado(0f, false)
        }

        // Saiu do lugar: não é alguém segurando a mão parada, é um sinal em
        // andamento. Reancora aqui, pra o caso de a mão parar mais adiante.
        if (hypot(abs(x - ancoraX), abs(y - ancoraY)) > deslocamentoMaximo) {
            abertaDesde = agora
            ancoraX = x
            ancoraY = y
            return Estado(0f, false)
        }

        val progresso = ((agora - inicio).toFloat() / duracaoMs).coerceIn(0f, 1f)
        if ((agora - inicio) < duracaoMs) return Estado(progresso, false)

        // Espera entre limpezas: sem ela, manter a mão aberta depois de limpar
        // dispararia de novo a cada quadro.
        val ultima = ultimaLimpeza
        if (ultima != null && (agora - ultima) <= esperaEntreLimpezasMs) {
            return Estado(1f, false)
        }
        ultimaLimpeza = agora
        abertaDesde = null
        return Estado(1f, true)
    }

    fun reset() {
        abertaDesde = null
    }
}
