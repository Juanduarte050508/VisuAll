package com.visuall.app.libras

// Em que fase do gesto o movimento está — ver MovementGate.
internal enum class EstadoMovimento {
    // Mão parada: quem manda é o classificador estático.
    PARADO,
    // Movimento intencional em curso: o classificador dinâmico vale, sobre a
    // janela de quadros que está entrando agora.
    SUSTENTADO,
    // O movimento acabou de terminar, mas a janela do gesto ainda vale por um
    // instante — ver ENCERRAMENTO_MS.
    ENCERRANDO
}

// Decide QUANDO confiar no classificador dinâmico (H, J, K, X, Z).
//
// O problema que isto resolve: LIMIAR_MOVIMENTO sozinho não distingue uma
// tremida de um frame de um traço real de letra — os dois cruzam o mesmo valor
// de magnitude. A diferença está na DURAÇÃO: um gesto de verdade sustenta o
// movimento por um tempo mínimo, ruído de câmera não.
//
// Estava embutido no LetraEngine chamando System.currentTimeMillis() direto, o
// que tornava o comportamento impossível de testar sem esperar em tempo real.
// Aqui o instante entra como parâmetro, igual ao LetterCommitGate. Ver
// MovementGateTest.
//
// A SAÍDA do gesto também é um portão, e por muito tempo não era. Quando o
// movimento parava, o portão fechava no mesmo quadro e o modo voltava na hora
// pro estático — só que a letra dinâmica ainda precisava de
// ESTAB_MIN_DINAMICO_MS de estabilidade pra entrar na frase. Ou seja: a janela
// que continha o gesto inteiro, justamente a mais informativa, nunca chegava a
// ser classificada tempo suficiente, e a letra se perdia bem no fim do
// movimento. Agora o portão continua aberto por ENCERRAMENTO_MS depois que o
// movimento cai (estado ENCERRANDO), tempo em que o LetraEngine reclassifica a
// janela CONGELADA no instante em que o gesto acabou — sem deixar os quadros
// de mão em repouso entrarem nela.
internal class MovementGate(
    private val limiarMovimento: Float = LibrasAnalyzer.LIMIAR_MOVIMENTO,
    private val sustentadoMs: Long = LibrasAnalyzer.MOVIMENTO_SUSTENTADO_MS,
    private val encerramentoMs: Long = LibrasAnalyzer.MOVIMENTO_ENCERRAMENTO_MS
) {
    // null = não está sustentando movimento agora.
    //
    // Nullable em vez de "0L significa parado", que é como estava no
    // LetraEngine: com o sentinela 0L, um instante que por acaso vale 0 é
    // indistinguível de "não começou", e o portão nunca liberava. Em produção
    // System.currentTimeMillis() não dá 0, então isso nunca apareceu — o teste
    // de independência de taxa de quadros, que começa a contar em t=0, pegou.
    private var sustentadoDesde: Long? = null

    // Houve um gesto de verdade que ainda não foi encerrado — é o que dá
    // direito ao período de graça. Sem isso, uma tremida curta (que nunca
    // chegou a SUSTENTADO) também ganharia o ENCERRANDO.
    private var gestoAberto = false
    private var encerrandoDesde: Long? = null

    // Chamado uma vez por frame com a magnitude de movimento medida.
    fun avaliar(movimento: Float, agora: Long): EstadoMovimento {
        if (movimento > limiarMovimento) {
            // Só marca o início na primeira vez; frames seguintes acumulam
            // tempo em cima do mesmo instante.
            val inicio = sustentadoDesde ?: agora.also { sustentadoDesde = it }
            if ((agora - inicio) >= sustentadoMs) {
                gestoAberto = true
                encerrandoDesde = null
                return EstadoMovimento.SUSTENTADO
            }
            // Acima do limiar mas ainda acumulando. Se um gesto acabou de
            // terminar, este movimento é o rabicho dele (a mão voltando ao
            // repouso), não um gesto novo: a graça continua correndo.
            return manterEncerramento(agora)
        }
        // Cair abaixo do limiar zera o acúmulo. É o que faz "tremida, pausa,
        // tremida" não somar até virar gesto.
        sustentadoDesde = null
        return manterEncerramento(agora)
    }

    private fun manterEncerramento(agora: Long): EstadoMovimento {
        if (!gestoAberto) return EstadoMovimento.PARADO
        val desde = encerrandoDesde ?: agora.also { encerrandoDesde = it }
        if ((agora - desde) >= encerramentoMs) {
            gestoAberto = false
            encerrandoDesde = null
            return EstadoMovimento.PARADO
        }
        return EstadoMovimento.ENCERRANDO
    }

    fun reset() {
        sustentadoDesde = null
        encerrandoDesde = null
        gestoAberto = false
    }
}
