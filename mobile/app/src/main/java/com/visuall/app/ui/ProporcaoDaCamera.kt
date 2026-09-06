package com.visuall.app.ui

import android.content.Context

/**
 * A proporcao de imagem que a pessoa escolheu: 4:3 ou 16:9.
 *
 * Existe porque a escolha e feita numa tela e usada em outra. O botao 4:3/16:9
 * fica na tela da camera; quem entra no modo Libras depois esperava encontrar
 * o mesmo enquadramento, e encontrava sempre tela cheia -- a preview do Libras
 * ignorava a escolha e enchia o aparelho, cortando as beiradas do quadro 4:3.
 * Num aparelho alto isso corta MUITO, e a imagem parece ampliada.
 *
 * Guardado em disco, e nao passado por argumento, porque o caminho entre as
 * duas telas nao e unico (da pra entrar no modo Libras, sair, voltar) e porque
 * a escolha deve sobreviver a fechar o app.
 */
internal object ProporcaoDaCamera {

    private const val ARQUIVO = "visuall_camera"
    private const val CHAVE = "quatro_por_tres"

    /** 4:3 e o padrao: e o quadro cheio do sensor, sem cortar nada. */
    private const val PADRAO = true

    private fun prefs(context: Context) =
        context.getSharedPreferences(ARQUIVO, Context.MODE_PRIVATE)

    fun ehQuatroPorTres(context: Context): Boolean =
        prefs(context).getBoolean(CHAVE, PADRAO)

    fun guardar(context: Context, quatroPorTres: Boolean) {
        prefs(context).edit().putBoolean(CHAVE, quatroPorTres).apply()
    }

    /**
     * A razao que a preview deve ter, no formato que o ConstraintLayout
     * entende, ou null para ocupar tudo o que houver.
     *
     * "H,3:4" quer dizer: a altura sai da largura, tres de largura para quatro
     * de altura -- o quadro 4:3 do sensor deitado de pe. Em 16:9 nao ha razao
     * nenhuma: quem decide o corte e o fillCenter da PreviewView, exatamente
     * como na tela da camera.
     */
    fun razaoDaPreview(quatroPorTres: Boolean): String? =
        if (quatroPorTres) "H,3:4" else null
}
