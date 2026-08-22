package com.visuall.app.libras

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

// Fixa a histerese que decide quando o movimento conta como gesto.
//
// O CHANGELOG registra que este portão já foi contado em FRAMES e que isso
// fazia o app perder H/J/K/X/Z em celulares lentos: "3 frames" num aparelho
// que analisa poucos quadros por segundo vira uma janela de tempo bem maior do
// que se pretendia, e o gesto (que dura 300-500ms de verdade) acabava antes de
// o portão liberar. A correção foi passar pra tempo. Nada verificava isso.
class MovementGateTest {

    private val acima = LibrasAnalyzer.LIMIAR_MOVIMENTO + 0.05f
    private val abaixo = LibrasAnalyzer.LIMIAR_MOVIMENTO - 0.05f
    private val espera = LibrasAnalyzer.MOVIMENTO_SUSTENTADO_MS
    private val graca = LibrasAnalyzer.MOVIMENTO_ENCERRAMENTO_MS

    @Test
    fun `movimento parado nunca libera`() {
        val gate = MovementGate()
        var t = 1_000L
        repeat(50) {
            assertEquals(EstadoMovimento.PARADO, gate.avaliar(abaixo, t))
            t += 16L
        }
    }

    @Test
    fun `um frame de movimento nao basta`() {
        // A tremida de um frame: é o falso positivo que o portão existe pra
        // barrar. Passa do limiar de magnitude, mas não de duração.
        val gate = MovementGate()
        assertEquals(EstadoMovimento.PARADO, gate.avaliar(acima, 1_000L))
    }

    @Test
    fun `libera quando o movimento e sustentado pelo tempo minimo`() {
        val gate = MovementGate()
        assertEquals(EstadoMovimento.PARADO, gate.avaliar(acima, 1_000L))
        assertEquals(EstadoMovimento.PARADO, gate.avaliar(acima, 1_000L + espera - 1))
        assertEquals(EstadoMovimento.SUSTENTADO, gate.avaliar(acima, 1_000L + espera))
    }

    @Test
    fun `cair abaixo do limiar zera o acumulo`() {
        // "Tremida, pausa, tremida" não deve somar até virar gesto: cada queda
        // abaixo do limiar reinicia a contagem.
        val gate = MovementGate()
        gate.avaliar(acima, 1_000L)
        gate.avaliar(abaixo, 1_000L + espera / 2)
        gate.avaliar(acima, 1_000L + espera / 2 + 1)
        // Já passou tempo total suficiente desde o primeiro movimento, mas não
        // de forma contínua.
        assertEquals(EstadoMovimento.PARADO, gate.avaliar(acima, 1_000L + espera))
    }

    @Test
    fun `continua liberado enquanto o movimento se mantem`() {
        val gate = MovementGate()
        gate.avaliar(acima, 1_000L)
        assertEquals(EstadoMovimento.SUSTENTADO, gate.avaliar(acima, 1_000L + espera))
        assertEquals(EstadoMovimento.SUSTENTADO, gate.avaliar(acima, 1_000L + espera * 3))
    }

    @Test
    fun `reset obriga a sustentar de novo`() {
        val gate = MovementGate()
        gate.avaliar(acima, 1_000L)
        assertEquals(EstadoMovimento.SUSTENTADO, gate.avaliar(acima, 1_000L + espera))
        gate.reset()
        assertEquals(EstadoMovimento.PARADO, gate.avaliar(acima, 1_000L + espera + 1))
    }

    @Test
    fun `o portao nao depende da taxa de quadros`() {
        // A propriedade que a correção comprou, e a razão de o portão ser em
        // tempo e não em contagem de frames: um celular que analisa 60 quadros
        // por segundo e um que analisa 8 precisam liberar no MESMO instante.
        // Com contagem de frames, o lento levaria 7x mais tempo de parede.
        fun instanteDeLiberacao(intervaloMs: Long): Long {
            val gate = MovementGate()
            var t = 0L
            repeat(500) {
                if (gate.avaliar(acima, t) == EstadoMovimento.SUSTENTADO) return t
                t += intervaloMs
            }
            return -1L
        }

        val rapido = instanteDeLiberacao(16L)   // ~60 quadros/s
        val lento = instanteDeLiberacao(125L)   // ~8 quadros/s
        assertTrue("nunca liberou no aparelho rapido", rapido >= 0)
        assertTrue("nunca liberou no aparelho lento", lento >= 0)
        // Cada um libera no primeiro frame após o tempo mínimo, então a
        // diferença é no máximo um intervalo de amostragem do mais lento.
        assertTrue(
            "liberou em tempos muito diferentes: rapido=$rapido lento=$lento",
            Math.abs(rapido - lento) <= 125L
        )
        assertTrue(rapido >= espera)
        assertTrue(lento >= espera)
    }

    @Test
    fun `o gesto continua valendo por um instante depois que o movimento para`() {
        // O bug de teste em campo: a letra dinâmica se perdia no FIM do
        // movimento. O portão fechava no mesmo quadro em que a mão parava, e a
        // janela que continha o gesto inteiro nunca era classificada pelos
        // ESTAB_MIN_DINAMICO_MS que a letra precisa pra entrar na frase.
        val gate = MovementGate()
        gate.avaliar(acima, 1_000L)
        assertEquals(EstadoMovimento.SUSTENTADO, gate.avaliar(acima, 1_000L + espera))

        val fim = 1_000L + espera
        assertEquals(EstadoMovimento.ENCERRANDO, gate.avaliar(abaixo, fim + 1))
        assertEquals(EstadoMovimento.ENCERRANDO, gate.avaliar(abaixo, fim + graca - 1))
        assertEquals(EstadoMovimento.PARADO, gate.avaliar(abaixo, fim + graca + 1))
    }

    @Test
    fun `a graca dura mais que a estabilidade exigida pra letra dinamica`() {
        // A propriedade que faz a correção valer: não adianta reclassificar a
        // janela do gesto se a janela fecha antes de a letra poder ser aceita.
        assertTrue(
            "graca=$graca precisa ser maior que ESTAB_MIN_DINAMICO_MS",
            graca > LibrasAnalyzer.ESTAB_MIN_DINAMICO_MS
        )
    }

    @Test
    fun `uma tremida que nunca virou gesto nao ganha periodo de graca`() {
        // A graça é o rabo de um gesto real. Ruído que nunca chegou a
        // SUSTENTADO não tem gesto nenhum pra reclassificar.
        val gate = MovementGate()
        assertEquals(EstadoMovimento.PARADO, gate.avaliar(acima, 1_000L))
        assertEquals(EstadoMovimento.PARADO, gate.avaliar(abaixo, 1_010L))
    }

    @Test
    fun `a mao voltando ao repouso nao reabre o gesto`() {
        // O movimento de trazer a mão de volta cruza o limiar de novo. Se ele
        // reiniciasse a contagem, viraria um segundo gesto fantasma; enquanto
        // a graça corre ele continua sendo o encerramento do primeiro.
        val gate = MovementGate()
        gate.avaliar(acima, 1_000L)
        gate.avaliar(acima, 1_000L + espera)
        val fim = 1_000L + espera
        // A graça começa no primeiro quadro abaixo do limiar (fim + 10), não
        // no último quadro do gesto — é de lá que o prazo conta.
        assertEquals(EstadoMovimento.ENCERRANDO, gate.avaliar(abaixo, fim + 10))
        assertEquals(EstadoMovimento.ENCERRANDO, gate.avaliar(acima, fim + 20))
        assertEquals(EstadoMovimento.ENCERRANDO, gate.avaliar(abaixo, fim + 10 + graca - 1))
        assertEquals(EstadoMovimento.PARADO, gate.avaliar(abaixo, fim + 10 + graca))
    }

    @Test
    fun `reset cancela a graca em andamento`() {
        // A mão saiu do quadro: o gesto anterior não tem mais nada a ver com o
        // que vier depois.
        val gate = MovementGate()
        gate.avaliar(acima, 1_000L)
        gate.avaliar(acima, 1_000L + espera)
        gate.reset()
        assertEquals(EstadoMovimento.PARADO, gate.avaliar(abaixo, 1_000L + espera + 10))
    }
}
