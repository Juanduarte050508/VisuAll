package com.visuall.app.ui

import com.visuall.app.ui.EncaixeDeQuadro.Modo
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * Os numeros sao os do aparelho de teste: a area de camera mede 1080x2184, e a
 * camera dos oculos manda 320x240 (4:3). Nao sao numeros redondos de proposito
 * -- foi com eles que o desenho saiu no lugar errado.
 */
class EncaixeDeQuadroTest {

    private val larguraTela = 1080f
    private val alturaTela = 2184f
    private val quatroPorTres = 4f / 3f

    @Test
    fun `imagem larga numa tela alta cabe inteira com faixas em cima e embaixo`() {
        val a = EncaixeDeQuadro.calcular(larguraTela, alturaTela, quatroPorTres, Modo.INTEIRA)
        assertEquals("usa toda a largura", 1080f, a.largura, 0.5f)
        assertEquals("altura = largura / (4/3)", 810f, a.altura, 0.5f)
        assertEquals("encostada nas laterais", 0f, a.esquerda, 0.5f)
        assertEquals("centralizada na vertical", 687f, a.topo, 0.5f)
        assertTrue("nao pode passar da tela", a.altura <= alturaTela)
    }

    @Test
    fun `a mesma imagem cortada estoura a largura da tela`() {
        val a = EncaixeDeQuadro.calcular(larguraTela, alturaTela, quatroPorTres, Modo.CORTANDO)
        assertEquals("usa toda a altura", 2184f, a.altura, 0.5f)
        assertEquals("largura = altura * (4/3)", 2912f, a.largura, 0.5f)
        assertTrue("sobra pra fora dos dois lados: $a", a.esquerda < 0f)
    }

    /**
     * O defeito de verdade, em numeros.
     *
     * O CENTRO da imagem cai no mesmo pixel nos dois modos -- os dois
     * centralizam. Entao e preciso olhar uma borda: o topo da imagem exibida
     * inteira comeca 687px mais abaixo do que a conta de cortar diz. Um
     * landmark no alto da mao ia parar la em cima, na faixa preta, que foi o
     * que apareceu na tela.
     */
    @Test
    fun `trocar os modos desloca o desenho em centenas de pixels`() {
        val inteira = EncaixeDeQuadro.calcular(larguraTela, alturaTela, quatroPorTres, Modo.INTEIRA)
        val cortando = EncaixeDeQuadro.calcular(larguraTela, alturaTela, quatroPorTres, Modo.CORTANDO)

        val topoCerto = inteira.topo                 // y do landmark em ny=0
        val topoErrado = cortando.topo
        assertTrue("o topo da imagem devia estar bem mais baixo: $topoCerto vs $topoErrado",
            abs(topoCerto - topoErrado) > 600f)

        // E a altura errada estica tudo: a mesma mao ocupa quase 3x mais tela.
        assertTrue("a escala vertical devia estar muito errada",
            cortando.altura / inteira.altura > 2.5f)
    }

    @Test
    fun `imagem e tela da mesma proporcao dao o mesmo resultado nos dois modos`() {
        // Sem sobra nenhuma os dois encaixes coincidem -- e por isso um teste
        // feito so com proporcoes iguais nao pegaria a troca de modo.
        val inteira = EncaixeDeQuadro.calcular(1200f, 900f, quatroPorTres, Modo.INTEIRA)
        val cortando = EncaixeDeQuadro.calcular(1200f, 900f, quatroPorTres, Modo.CORTANDO)
        assertEquals(inteira, cortando)
    }

    /** A camera do celular em retrato: imagem em pe, 3:4. O modo antigo segue igual. */
    @Test
    fun `camera do celular em retrato continua cortando as laterais`() {
        val a = EncaixeDeQuadro.calcular(larguraTela, alturaTela, 3f / 4f, Modo.CORTANDO)
        assertEquals("cobre a tela toda na vertical", 2184f, a.altura, 0.5f)
        assertTrue("e corta um pouco nas laterais: $a", a.esquerda < 0f)
    }

    @Test
    fun `medidas invalidas nao explodem`() {
        // A View e medida como 0x0 antes do primeiro layout, e o primeiro
        // quadro pode chegar antes disso.
        val vazia = EncaixeDeQuadro.Area(0f, 0f, 0f, 0f)
        assertEquals(vazia, EncaixeDeQuadro.calcular(0f, 0f, quatroPorTres, Modo.INTEIRA))
        assertEquals(vazia, EncaixeDeQuadro.calcular(1080f, 2184f, 0f, Modo.CORTANDO))
    }
}
