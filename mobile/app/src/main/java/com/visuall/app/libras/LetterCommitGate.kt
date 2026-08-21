package com.visuall.app.libras

// Decide QUANDO uma letra reconhecida vira letra na frase.
//
// Esta é a regra que mais define a sensação de uso do app: se está frouxa,
// aparecem letras que a pessoa não fez; se está apertada, a pessoa faz o
// sinal certo e nada acontece. Era o único trecho realmente crítico que
// vivia no meio do processamento de câmera do LibrasAnalyzer, onde não dava
// pra testar sem celular, câmera e modelo carregado — e é justamente onde
// mais se mexeu por tentativa e erro (ver as entradas de threshold no
// CHANGELOG). Isolado aqui, cada portão vira um teste
// (LetterCommitGateTest).
//
// São quatro portões, todos precisam passar:
//  1. ESTABILIDADE — a mesma letra tem que se manter por um tempo mínimo.
//     Conta em milissegundos, não em quadros: em celular mais lento, "3
//     quadros" viravam um tempo de parede bem maior e o app perdia gestos
//     rápidos.
//  2. Não pode ser "-" (nenhuma letra reconhecida).
//  3. Não pode ser a mesma letra que acabou de entrar (evita a mão parada
//     digitar "AAAA").
//  4. COOLDOWN — tem que ter passado um tempo desde a última letra aceita.
//
// Letras com movimento (modo "dinamico*") usam tempos menores, porque o
// gesto é passageiro: esperar demais faz a janela do gesto terminar antes de
// a letra ser aceita.
internal class LetterCommitGate {

    // Última letra vista (mesmo que ainda não aceita) e desde quando ela
    // vem se repetindo. null = ainda não estabilizou.
    //
    // Nullable em vez de "0L significa que não começou": com o sentinela, um
    // instante que por acaso valha 0 é indistinguível de "não começou" e o
    // portão nunca abre. Em produção currentTimeMillis() não dá 0, então nunca
    // apareceu aqui — mas apareceu no MovementGate, que tem a mesma estrutura,
    // assim que um teste começou a contar em t=0.
    private var ultimaPredicao = ""
    private var estabilidadeDesde: Long? = null

    // Última letra que de fato entrou na frase, e quando. null = nenhuma ainda,
    // e aí não há cooldown pra respeitar.
    private var ultimaLetraAdicionada = ""
    private var ultimoTempoAdicao: Long? = null

    val letraEstabilizando: String get() = ultimaPredicao

    /**
     * Chamado a cada quadro em que houve classificação. Atualiza o
     * acompanhamento de estabilidade e responde se esta letra pode entrar na
     * frase agora.
     *
     * Quem chama continua responsável pelo que fazer com a letra (repetição,
     * frase); ao aceitar, precisa chamar [registrarComite].
     */
    fun avaliar(letra: String, modo: String, agora: Long): Boolean {
        if (letra != "-") {
            // Trocou de letra: recomeça a contagem. Se é a mesma e ainda não
            // havia contagem em andamento, começa agora.
            if (letra != ultimaPredicao) estabilidadeDesde = agora
            else if (estabilidadeDesde == null) estabilidadeDesde = agora
            ultimaPredicao = letra
        } else {
            estabilidadeDesde = null
            ultimaPredicao = ""
        }

        val dinamico = modo.startsWith("dinamico")
        val estabMinMs = if (dinamico) {
            LibrasAnalyzer.ESTAB_MIN_DINAMICO_MS
        } else {
            LibrasAnalyzer.ESTAB_MIN_ESTATICO_MS
        }
        val cooldown = if (dinamico) {
            LibrasAnalyzer.COOLDOWN_DINAMICO
        } else {
            LibrasAnalyzer.COOLDOWN_ESTATICO
        }
        val desde = estabilidadeDesde
        val estabilidadeOk = desde != null && (agora - desde) >= estabMinMs
        // Sem letra aceita antes, não há cooldown a cumprir.
        val cooldownOk = ultimoTempoAdicao?.let { (agora - it) > cooldown } ?: true

        return estabilidadeOk &&
            letra != "-" &&
            letra != ultimaLetraAdicionada &&
            cooldownOk
    }

    /** A letra entrou na frase: zera a estabilidade e arma o cooldown. */
    fun registrarComite(letra: String, agora: Long) {
        ultimaLetraAdicionada = letra
        ultimoTempoAdicao = agora
        estabilidadeDesde = null
    }

    /**
     * Libera a mesma letra a ser aceita de novo imediatamente. Usado quando a
     * pessoa mexeu na frase por fora (apagou, limpou, confirmou uma
     * repetição): sem isso ela teria que fazer outra letra no meio pra poder
     * repetir a mesma.
     *
     * Repare que o cooldown NÃO é zerado junto — só a trava de letra
     * repetida. Zerar o tempo também deixaria duas letras entrarem quase no
     * mesmo instante.
     */
    fun liberarRepeticao() {
        ultimaLetraAdicionada = ""
    }

    /**
     * Zera tudo. Usado quando a mão sai do quadro ou o modo muda — aí a
     * sequência anterior não tem mais relação com a próxima.
     *
     * O cooldown (ultimoTempoAdicao) é preservado de propósito: ele existe
     * pra evitar duas letras coladas no tempo, e isso continua valendo mesmo
     * se a mão sumiu por um instante.
     */
    fun reset() {
        ultimaPredicao = ""
        estabilidadeDesde = null
        ultimaLetraAdicionada = ""
    }
}
