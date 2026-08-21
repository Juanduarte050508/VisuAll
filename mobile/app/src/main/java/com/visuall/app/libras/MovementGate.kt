package com.visuall.app.libras

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
internal class MovementGate(
    private val limiarMovimento: Float = LibrasAnalyzer.LIMIAR_MOVIMENTO,
    private val sustentadoMs: Long = LibrasAnalyzer.MOVIMENTO_SUSTENTADO_MS
) {
    // null = não está sustentando movimento agora.
    //
    // Nullable em vez de "0L significa parado", que é como estava no
    // LetraEngine: com o sentinela 0L, um instante que por acaso vale 0 é
    // indistinguível de "não começou", e o portão nunca liberava. Em produção
    // System.currentTimeMillis() não dá 0, então isso nunca apareceu — o teste
    // de independência de taxa de quadros, que começa a contar em t=0, pegou.
    private var sustentadoDesde: Long? = null

    // Chamado uma vez por frame com a magnitude de movimento medida. Devolve
    // true quando o movimento já foi sustentado o suficiente pra ser tratado
    // como gesto intencional.
    fun avaliar(movimento: Float, agora: Long): Boolean {
        if (movimento <= limiarMovimento) {
            // Cair abaixo do limiar zera o acúmulo. É o que faz "tremida,
            // pausa, tremida" não somar até virar gesto.
            sustentadoDesde = null
            return false
        }
        // Só marca o início na primeira vez; frames seguintes acumulam tempo
        // em cima do mesmo instante.
        val inicio = sustentadoDesde ?: agora.also { sustentadoDesde = it }
        return (agora - inicio) >= sustentadoMs
    }

    fun reset() {
        sustentadoDesde = null
    }
}
