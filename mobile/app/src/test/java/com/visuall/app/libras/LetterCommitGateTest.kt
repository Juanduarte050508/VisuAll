package com.visuall.app.libras

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Testa a regra que decide se uma letra reconhecida vira letra na frase.
 *
 * Antes destes testes, a única forma de verificar qualquer uma destas
 * afirmações era instalar o app e soletrar na frente da câmera — e é
 * exatamente essa regra que vem sendo ajustada por tentativa e erro (ver as
 * entradas de threshold no CHANGELOG). Aqui o tempo é um número controlado
 * pelo teste, então dá pra afirmar "com 200ms não entra, com 900ms entra"
 * sem depender de mão, celular nem modelo.
 *
 * Os tempos usados vêm das constantes reais (ESTAB_MIN_*, COOLDOWN_*), então
 * mexer nelas mantém os testes válidos: eles verificam o COMPORTAMENTO
 * (precisa segurar; não repete sozinha; respeita o intervalo), não os
 * números em si.
 */
class LetterCommitGateTest {

    private val estatico = "estatico"
    private val dinamico = "dinamico"
    private val estabEstatico = LibrasAnalyzer.ESTAB_MIN_ESTATICO_MS
    private val estabDinamico = LibrasAnalyzer.ESTAB_MIN_DINAMICO_MS
    private val cooldownEstatico = LibrasAnalyzer.COOLDOWN_ESTATICO

    // Base de tempo arbitrária, só pra os testes lerem como tempo de relógio.
    // Não precisa mais ser alta: o cooldown inicial agora é "nenhuma letra
    // aceita ainda" em vez de "última aceita no instante 0" (ver o teste
    // `funciona com o relogio comecando em zero`).
    private val t0 = 1_000_000L

    /** Segura a mesma letra por [duracao] ms; devolve se ela foi aceita. */
    private fun LetterCommitGate.segurar(
        letra: String,
        modo: String,
        duracao: Long,
        inicio: Long,
        passo: Long = 33L
    ): Boolean {
        var aceita = false
        var agora = inicio
        while (agora <= inicio + duracao) {
            if (avaliar(letra, modo, agora)) {
                aceita = true
                registrarComite(letra, agora)
                break
            }
            agora += passo
        }
        return aceita
    }

    @Test
    fun `funciona com o relogio comecando em zero`() {
        // Os dois portões deste pacote guardavam "desde quando" num Long com
        // 0L significando "não começou". Isso torna o instante 0 indistinguível
        // de "não começou", e o portão nunca abre. Não aparece em produção
        // porque System.currentTimeMillis() nunca dá 0 — apareceu quando um
        // teste do MovementGate passou a contar a partir de t=0, e a mesma
        // estrutura estava aqui. Agora os dois usam null.
        val gate = LetterCommitGate()
        assertTrue(gate.segurar("A", estatico, estabEstatico * 2, inicio = 0L))
    }

    @Test
    fun `letra so entra depois de segurar o tempo minimo`() {
        val gate = LetterCommitGate()
        // Bem abaixo do mínimo: não pode entrar.
        assertFalse(gate.segurar("A", estatico, estabEstatico / 3, t0))

        // Segurando o suficiente: entra.
        val gate2 = LetterCommitGate()
        assertTrue(gate2.segurar("A", estatico, estabEstatico * 2, t0))
    }

    @Test
    fun `um unico quadro nunca aceita a letra`() {
        // O caso do "reconheceu sem eu estar fazendo": um pico de um quadro
        // só não pode virar letra.
        val gate = LetterCommitGate()
        assertFalse(gate.avaliar("A", estatico, t0))
    }

    @Test
    fun `trocar de letra reinicia a contagem`() {
        val gate = LetterCommitGate()
        var agora = t0
        // Quase estabiliza em "A"...
        repeat(20) { gate.avaliar("A", estatico, agora); agora += 33 }
        // ...e troca pra "B" no último instante.
        gate.avaliar("B", estatico, agora)
        // O tempo acumulado em "A" não pode valer pro "B".
        assertFalse(gate.avaliar("B", estatico, agora + estabEstatico / 2))
    }

    @Test
    fun `mao parada na mesma letra nao digita ela repetidamente`() {
        // Sem esta trava, segurar a mão parada num sinal escreveria "AAAAA".
        val gate = LetterCommitGate()
        assertTrue(gate.segurar("A", estatico, estabEstatico * 2, t0))

        val depois = t0 + estabEstatico * 2
        var agora = depois
        repeat(200) {
            assertFalse(
                "letra repetiu sozinha em t=$agora",
                gate.avaliar("A", estatico, agora)
            )
            agora += 33
        }
    }

    @Test
    fun `letra diferente entra, mas so depois do cooldown`() {
        val gate = LetterCommitGate()
        assertTrue(gate.segurar("A", estatico, estabEstatico * 2, t0))
        val tA = t0 + estabEstatico * 2

        // "B" estabiliza, mas ainda dentro do cooldown do "A".
        assertFalse(gate.segurar("B", estatico, cooldownEstatico / 4, tA))

        // Passado o cooldown, "B" entra.
        val gate2 = LetterCommitGate()
        assertTrue(gate2.segurar("A", estatico, estabEstatico * 2, t0))
        val depoisDoCooldown = t0 + estabEstatico * 2 + cooldownEstatico + 1
        assertTrue(gate2.segurar("B", estatico, estabEstatico * 2, depoisDoCooldown))
    }

    @Test
    fun `letra com movimento estabiliza mais rapido que letra parada`() {
        // Gesto dinâmico dura ~300-500ms; exigir o tempo da estática faria a
        // janela do gesto acabar antes de a letra ser aceita.
        val duracaoCurta = (estabDinamico + estabEstatico) / 2

        val gateDinamico = LetterCommitGate()
        assertTrue(
            "dinâmica deveria entrar com $duracaoCurta ms",
            gateDinamico.segurar("J", dinamico, duracaoCurta, t0)
        )

        val gateEstatico = LetterCommitGate()
        assertFalse(
            "estática NÃO deveria entrar com $duracaoCurta ms",
            gateEstatico.segurar("A", estatico, duracaoCurta, t0)
        )
    }

    @Test
    fun `modos individual e parcial contam como dinamico`() {
        // Os modelos individuais/parciais reportam "dinamico_individual" e
        // "dinamico_parcial"; se a checagem fosse igualdade exata em vez de
        // prefixo, eles cairiam nos tempos da estática e ficariam lentos
        // demais pra pegar o gesto.
        val duracaoCurta = (estabDinamico + estabEstatico) / 2
        listOf("dinamico", "dinamico_individual", "dinamico_parcial").forEach { modo ->
            val gate = LetterCommitGate()
            assertTrue("modo $modo deveria usar o tempo da dinâmica",
                gate.segurar("K", modo, duracaoCurta, t0))
        }
    }

    @Test
    fun `sem letra reconhecida zera a estabilidade`() {
        val gate = LetterCommitGate()
        var agora = t0
        // Acumula quase o tempo necessário...
        repeat(20) { gate.avaliar("A", estatico, agora); agora += 33 }
        // ...e a mão sai/perde o sinal por um quadro.
        gate.avaliar("-", estatico, agora)
        // O acumulado não vale mais.
        assertFalse(gate.avaliar("A", estatico, agora + estabEstatico / 2))
    }

    @Test
    fun `reset limpa a estabilidade e libera a mesma letra`() {
        val gate = LetterCommitGate()
        assertTrue(gate.segurar("A", estatico, estabEstatico * 2, t0))

        // Mão saiu do quadro: reset. Depois do cooldown, "A" pode de novo —
        // refazer o sinal é a forma natural de repetir a letra.
        gate.reset()
        val depois = t0 + estabEstatico * 2 + cooldownEstatico + 1
        assertTrue(gate.segurar("A", estatico, estabEstatico * 2, depois))
    }

    @Test
    fun `liberarRepeticao permite a mesma letra sem zerar o cooldown`() {
        val gate = LetterCommitGate()
        assertTrue(gate.segurar("A", estatico, estabEstatico * 2, t0))
        val tA = t0 + estabEstatico * 2

        gate.liberarRepeticao()

        // Ainda dentro do cooldown: mesmo liberada, não entra agora.
        assertFalse(gate.segurar("A", estatico, cooldownEstatico / 4, tA))

        // Passado o cooldown, entra.
        val depois = tA + cooldownEstatico + 1
        assertTrue(gate.segurar("A", estatico, estabEstatico * 2, depois))
    }

    @Test
    fun `letraEstabilizando reflete a letra sendo observada`() {
        val gate = LetterCommitGate()
        gate.avaliar("C", estatico, t0)
        assert(gate.letraEstabilizando == "C")
        gate.avaliar("-", estatico, t0 + 33)
        assert(gate.letraEstabilizando == "")
    }
}
