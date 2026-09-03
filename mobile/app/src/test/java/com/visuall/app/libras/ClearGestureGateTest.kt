package com.visuall.app.libras

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

// Fixa quando a mão aberta limpa a frase.
//
// Relatado no aparelho, em duas rodadas: primeiro "AJUDAR não funciona, fica
// contando os segundos pra limpar", depois "quando vou fazer o sinal de AJUDAR,
// que contém levemente a mão aberta, ele já começa a carregar a barra". Os dois
// vêm do mesmo lugar: usar a abertura da mão (e depois o tremor entre quadros)
// pra decidir algo que só o DESLOCAMENTO distingue.
class ClearGestureGateTest {

    private val duracao = LibrasAnalyzer.TEMPO_PRA_LIMPAR_CORPO
    private val espera = LibrasAnalyzer.ESPERA_ENTRE_LIMPEZAS_MS
    private val limite = LibrasAnalyzer.LIMPAR_DESLOCAMENTO_MAXIMO

    @Test
    fun `mao aberta e parada no lugar limpa depois do tempo`() {
        val g = ClearGestureGate()
        assertFalse(g.avaliar(true, 0.5f, 0.5f, 0L).limpar)
        assertFalse(g.avaliar(true, 0.5f, 0.5f, duracao - 1).limpar)
        assertTrue(g.avaliar(true, 0.5f, 0.5f, duracao).limpar)
    }

    @Test
    fun `tremor pequeno no mesmo lugar nao atrapalha`() {
        // Ninguém segura a mão perfeitamente imóvel: o portão tem que tolerar o
        // tremor natural, senão limpar vira impossível.
        val g = ClearGestureGate()
        var t = 0L
        var limpou = false
        while (t <= duracao + 200) {
            val jitter = if ((t / 33) % 2 == 0L) limite / 4 else -limite / 4
            if (g.avaliar(true, 0.5f + jitter, 0.5f, t).limpar) limpou = true
            t += 33L
        }
        assertTrue("tremor natural nao pode impedir a limpeza", limpou)
    }

    @Test
    fun `sinal que atravessa a tela com a mao aberta nunca limpa`() {
        // O AJUDAR: mão aberta o tempo todo, mas viajando. É o caso que fez a
        // barra encher durante o sinal nas duas tentativas anteriores.
        val g = ClearGestureGate()
        var t = 0L
        var x = 0.2f
        repeat(400) {
            val e = g.avaliar(true, x, 0.5f, t)
            assertFalse("nao pode limpar durante um sinal", e.limpar)
            x += 0.004f
            if (x > 0.8f) x = 0.2f
            t += 33L
        }
    }

    @Test
    fun `sinal lento tambem nao enche a barra`() {
        // O motivo de trocar tremor por deslocamento: um gesto devagar tem
        // tremor baixíssimo entre quadros vizinhos, mas percorre distância.
        val g = ClearGestureGate()
        var t = 0L
        var x = 0.3f
        var progressoMaximo = 0f
        repeat(300) {
            val e = g.avaliar(true, x, 0.5f, t)
            progressoMaximo = maxOf(progressoMaximo, e.progresso)
            assertFalse(e.limpar)
            x += 0.0015f   // devagar, mas sempre na mesma direção
            t += 33L
        }
        assertTrue("a barra nao devia chegar perto do fim", progressoMaximo < 1f)
    }

    @Test
    fun `mao que viaja e depois para limpa a partir de onde parou`() {
        // A pessoa termina um sinal e então segura a mão: a contagem vale do
        // momento em que ela de fato parou, não de antes.
        val g = ClearGestureGate()
        var t = 0L
        // Vai e volta com passo maior que o limite: cada quadro reancora, entao
        // a contagem so pode valer a partir do instante em que a mao PARA.
        val passo = limite * 1.2f
        repeat(30) {
            g.avaliar(true, if (it % 2 == 0) 0.5f else 0.5f + passo, 0.5f, t)
            t += 33L
        }
        val x = 0.5f
        // A mao para aqui: e deste instante que os 5s valem.
        val parouEm = t
        g.avaliar(true, x, 0.5f, parouEm)
        assertFalse(g.avaliar(true, x, 0.5f, parouEm + duracao - 100).limpar)
        assertTrue(g.avaliar(true, x, 0.5f, parouEm + duracao + 100).limpar)
    }

    @Test
    fun `mao fechada nunca limpa`() {
        val g = ClearGestureGate()
        var t = 0L
        repeat(200) {
            assertFalse(g.avaliar(false, 0.5f, 0.5f, t).limpar)
            t += 33L
        }
    }

    @Test
    fun `nao limpa duas vezes seguidas segurando a mao`() {
        val g = ClearGestureGate()
        g.avaliar(true, 0.5f, 0.5f, 0L)
        assertTrue(g.avaliar(true, 0.5f, 0.5f, duracao).limpar)
        var t = duracao + 33L
        var limpezas = 0
        while (t < duracao + espera) {
            if (g.avaliar(true, 0.5f, 0.5f, t).limpar) limpezas++
            t += 33L
        }
        assertEquals("so podia ter limpado uma vez", 0, limpezas)
    }

    @Test
    fun `pode limpar de novo depois da espera`() {
        val g = ClearGestureGate()
        g.avaliar(true, 0.5f, 0.5f, 0L)
        assertTrue(g.avaliar(true, 0.5f, 0.5f, duracao).limpar)
        val base = duracao + espera + 100L
        g.avaliar(true, 0.5f, 0.5f, base)
        assertTrue(g.avaliar(true, 0.5f, 0.5f, base + duracao).limpar)
    }

    @Test
    fun `reset obriga a segurar de novo`() {
        val g = ClearGestureGate()
        g.avaliar(true, 0.5f, 0.5f, 0L)
        g.reset()
        assertFalse(g.avaliar(true, 0.5f, 0.5f, duracao).limpar)
    }

    @Test
    fun `sinal sendo gravado nunca enche a barra`() {
        // O AJUDAR de verdade: a mao aberta e a de APOIO e fica no MESMO lugar,
        // entao o deslocamento nao ajuda em nada aqui. O que segura a barra e
        // saber que a captura esta gravando um sinal.
        val g = ClearGestureGate()
        var t = 0L
        repeat(300) {
            val e = g.avaliar(true, 0.5f, 0.5f, t, gestoEmAndamento = true)
            assertFalse("nao pode limpar durante um sinal", e.limpar)
            assertEquals("a barra nem devia sair do zero", 0f, e.progresso, 0f)
            t += 33L
        }
    }

    @Test
    fun `captura no meio da contagem zera a barra`() {
        val g = ClearGestureGate()
        g.avaliar(true, 0.5f, 0.5f, 0L)
        assertTrue(g.avaliar(true, 0.5f, 0.5f, duracao - 500).progresso > 0f)

        // Comecou um sinal: a contagem morre aqui, nao so pausa.
        assertEquals(0f, g.avaliar(true, 0.5f, 0.5f, duracao - 400, true).progresso, 0f)

        // Sinal acabou e a mao continua aberta e parada: precisa dos 3s cheios.
        val fim = duracao - 400
        g.avaliar(true, 0.5f, 0.5f, fim)
        assertFalse(g.avaliar(true, 0.5f, 0.5f, fim + duracao - 100).limpar)
        assertTrue(g.avaliar(true, 0.5f, 0.5f, fim + duracao).limpar)
    }

    @Test
    fun `funciona com o relogio comecando em zero`() {
        val g = ClearGestureGate()
        assertFalse(g.avaliar(true, 0.5f, 0.5f, 0L).limpar)
        assertTrue(g.avaliar(true, 0.5f, 0.5f, duracao).limpar)
    }
}
