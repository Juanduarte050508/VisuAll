package com.visuall.app.libras

// Contas do painel de treino/calibração (quanto já foi coletado, quais
// letras ainda estão fracas, qual treinar em seguida) — extraído do
// LibrasFragment porque é aritmética pura sobre contagens de amostras, sem
// nada de View. Recebe a contagem por letra como função em vez de acessar o
// LibrasAnalyzer direto, o que deixa cada regra testável sem câmera nem
// modelo carregado.
internal object TrainingProgressCalculator {

    data class Progress(
        val percent: Int,
        val missingLetters: List<String>,
        val trainedLetters: Int,
        val totalSamples: Int
    )

    fun calcular(
        letras: List<String>,
        alvoForte: Int,
        contarAmostras: (String) -> Int
    ): Progress {
        var trainedLetters = 0
        var totalSamples = 0
        var cappedSamples = 0
        val missing = mutableListOf<Pair<String, Int>>()

        letras.forEach { letra ->
            val count = contarAmostras(letra)
            totalSamples += count
            // Limitado ao alvo: 300 amostras de "A" não podem compensar 0 de
            // "B" na barra de progresso — senão ela mostra "quase pronto" com
            // metade do alfabeto sem nenhum dado.
            cappedSamples += count.coerceAtMost(alvoForte)
            if (count >= alvoForte) {
                trainedLetters++
            } else {
                missing += letra to count
            }
        }

        val percent = if (letras.isEmpty()) {
            0
        } else {
            (cappedSamples * 100 / (letras.size * alvoForte)).coerceIn(0, 100)
        }
        // Mais fraca primeiro (menos amostras), desempate alfabético pra a
        // ordem não ficar dançando entre atualizações.
        val sortedMissing = missing
            .sortedWith(compareBy<Pair<String, Int>> { it.second }.thenBy { it.first })
            .map { it.first }
        return Progress(percent, sortedMissing, trainedLetters, totalSamples)
    }

    // Próxima letra que ainda não atingiu o alvo, varrendo circularmente a
    // partir da atual. null = todas já estão fortes.
    fun indiceProximaLetraFraca(
        letras: List<String>,
        indiceAtual: Int,
        includeCurrent: Boolean,
        alvoForte: Int,
        contarAmostras: (String) -> Int
    ): Int? {
        if (letras.isEmpty()) return null
        val offset = if (includeCurrent) 0 else 1
        return letras.indices
            .map { (indiceAtual + offset + it) % letras.size }
            .firstOrNull { index -> contarAmostras(letras[index]) < alvoForte }
    }

    fun nivel(count: Int, alvoForte: Int, alvoBasico: Int): String {
        return when {
            count >= alvoForte -> "FORTE"
            count >= alvoBasico -> "BASICO"
            count > 0 -> "INICIO"
            else -> "SEM DADOS"
        }
    }
}
