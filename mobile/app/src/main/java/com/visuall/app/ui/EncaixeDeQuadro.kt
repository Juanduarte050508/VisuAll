package com.visuall.app.ui

/**
 * Onde a imagem realmente cai dentro da View que a mostra.
 *
 * Os landmarks chegam normalizados (0..1) no espaco DA IMAGEM. Pra desenhar em
 * cima da mao, e preciso saber que pedaco da tela a imagem ocupa -- e isso
 * depende de como ela foi encaixada. Sao dois encaixes diferentes no app, e
 * usar a conta de um pro outro nao da erro nenhum: da desenho no lugar errado,
 * que foi exatamente o que aconteceu com a camera dos oculos.
 *
 * Fica separado da View pra poder ser medido em teste: geometria errada nao
 * levanta excecao, so parece torta na tela, e "parece torta" nao falha build
 * nenhum.
 */
internal object EncaixeDeQuadro {

    enum class Modo {
        /** Cobre a View inteira e corta o que sobra (o `fillCenter` da PreviewView). */
        CORTANDO,

        /** Cabe inteira, com faixas vazias nas sobras (o `fitCenter` da ImageView). */
        INTEIRA
    }

    /** Retangulo ocupado pela imagem, em pixels da View. */
    data class Area(val esquerda: Float, val topo: Float, val largura: Float, val altura: Float)

    /**
     * @param proporcaoConteudo largura/altura da imagem processada.
     */
    fun calcular(larguraView: Float, alturaView: Float, proporcaoConteudo: Float, modo: Modo): Area {
        if (larguraView <= 0f || alturaView <= 0f || proporcaoConteudo <= 0f) {
            return Area(0f, 0f, 0f, 0f)
        }
        val proporcaoView = larguraView / alturaView
        // A View e mais larga que a imagem?
        val maisLargaQueAImagem = proporcaoView > proporcaoConteudo
        // Cortando, a dimensao que "sobra" e a que manda; cabendo inteira, e a
        // que "falta". Por isso os dois modos sao a mesma conta com o teste
        // invertido -- e por isso trocar um pelo outro passa despercebido.
        val limitadoPelaLargura = if (modo == Modo.CORTANDO) maisLargaQueAImagem else !maisLargaQueAImagem

        return if (limitadoPelaLargura) {
            val largura = larguraView
            val altura = larguraView / proporcaoConteudo
            Area(0f, (alturaView - altura) / 2f, largura, altura)
        } else {
            val altura = alturaView
            val largura = alturaView * proporcaoConteudo
            Area((larguraView - largura) / 2f, 0f, largura, altura)
        }
    }
}
