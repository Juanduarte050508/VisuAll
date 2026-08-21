package com.visuall.app.libras

// A regra que transforma a saída de um modelo em "é esta letra" ou "não sei".
//
// Estava escrita três vezes dentro do LetraEngine — uma pro modelo estático,
// uma pro dinâmico e uma pros individuais — com diferenças pequenas entre as
// cópias (uma checava índice com getOrNull, outra com <, e a dos individuais
// tratava o vice-campeão de um jeito próprio). Diferença pequena em código
// duplicado é justamente o que ninguém revisa e o que nenhum teste pegava,
// porque nada disso era alcançável sem carregar um modelo ONNX de verdade num
// aparelho.
//
// Aqui não há sessão ONNX, Context nem asset: entra número, sai decisão. É o
// que deixa a decisão testável em JVM. Ver LetterDecisionTest.
internal object LetterDecision {

    // Decisão sobre um modelo MULTICLASSE (softmax sobre todas as letras): o
    // vencedor precisa de confiança absoluta E de distância do segundo lugar.
    //
    // A margem existe porque confiança alta sozinha não basta: um modelo
    // dividido entre C e mão aberta pode dar 0.94 e 0.93. Absoluto passa, mas
    // a decisão é uma moeda ao ar — e o sintoma no app é a letra trocando
    // sozinha entre duas opções.
    fun deProbabilidades(
        probs: FloatArray,
        labels: List<String>,
        confiancaMinima: Float,
        margemMinima: Float,
        modo: String
    ): Prediction {
        if (probs.isEmpty()) return Prediction("-", 0f, modo, 0f)

        var idx = 0
        for (i in probs.indices) if (probs[i] > probs[idx]) idx = i
        val confianca = probs[idx]

        var segundo = 0f
        for (i in probs.indices) if (i != idx && probs[i] > segundo) segundo = probs[i]
        val margem = confianca - segundo

        // labels vem do labels.txt gravado junto do modelo. Se os dois saírem
        // de sincronia (modelo com mais saídas que rótulos), a saída extra não
        // tem nome — e chutar um nome errado é pior que não responder.
        val label = labels.getOrNull(idx)
        val letra = if (label != null && confianca >= confiancaMinima && margem >= margemMinima) {
            label
        } else {
            "-"
        }
        return Prediction(letra, confianca, modo, margem)
    }

    // Decisão sobre os modelos INDIVIDUAIS: cada letra tem seu próprio
    // classificador binário ("é H ou não é H"), então não existe softmax
    // compartilhado — a competição é feita aqui, comparando a resposta de cada
    // modelo.
    //
    // ATENÇÃO ao caso de UM único modelo treinado. Com um só, o segundo lugar
    // é 0, então a margem passa a valer o mesmo que a própria confiança e o
    // portão de margem fica inoperante: qualquer coisa acima de
    // confiancaMinima passa. É grave justamente aqui, porque um binário nunca
    // viu "mão mexendo sem sinalizar" como negativo (ver CONFIANCA_INDIVIDUAL
    // em LibrasAnalyzer) e responde alto com facilidade.
    //
    // E não é um caso raro: é o primeiro caso. Treinar UMA letra pra testar se
    // gravar resolve produz exatamente um modelo individual. Com um só,
    // exigimos confiancaSemRival — mais alta — em vez de aceitar uma folga que
    // não foi medida contra ninguém.
    fun deModelosIndividuais(
        pontuacoes: List<Pair<String, Float>>,
        confiancaMinima: Float,
        margemMinima: Float,
        confiancaSemRival: Float,
        modo: String
    ): Prediction {
        if (pontuacoes.isEmpty()) return Prediction("-", 0f, modo, 0f)

        var melhorLabel = ""
        var melhor = 0f
        var segundo = 0f
        pontuacoes.forEach { (label, valor) ->
            if (valor > melhor) {
                segundo = melhor
                melhor = valor
                melhorLabel = label
            } else if (valor > segundo) {
                segundo = valor
            }
        }

        if (pontuacoes.size == 1) {
            // Sem ninguém pra comparar, a margem é fictícia: reportamos 0 pra
            // não fingir uma folga que não foi medida.
            val (label, valor) = pontuacoes[0]
            return Prediction(
                if (valor >= confiancaSemRival) label else "-",
                valor,
                modo,
                0f
            )
        }

        val margem = melhor - segundo
        val letra = if (melhor >= confiancaMinima && margem >= margemMinima) melhorLabel else "-"
        return Prediction(letra, melhor, modo, margem)
    }

    // Média quadro a quadro, usada pra transformar uma gravação de calibração
    // em uma referência única. Lista vazia devolve zeros.
    fun media(frames: List<FloatArray>, features: Int): FloatArray {
        val resultado = FloatArray(features)
        if (frames.isEmpty()) return resultado
        frames.forEach { frame ->
            for (i in resultado.indices) resultado[i] += frame[i]
        }
        for (i in resultado.indices) resultado[i] /= frames.size.toFloat()
        return resultado
    }
}
