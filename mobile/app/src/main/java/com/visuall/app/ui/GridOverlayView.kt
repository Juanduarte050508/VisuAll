package com.visuall.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

// Grade de composição (regra dos terços) desenhada por cima da preview.
//
// É uma View e não um drawable porque as linhas precisam acompanhar o tamanho
// real da preview, que muda com a proporção escolhida (4:3 / 16:9) -- um
// shape XML teria que ser refeito a cada troca.
//
// Não intercepta toque: fica com isClickable=false para o gesto de zoom e o
// toque-para-focar continuarem chegando na PreviewView por baixo.
class GridOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        // Branco translúcido: some sobre qualquer cena sem competir com o
        // enquadramento, que é o ponto de uma grade auxiliar.
        color = Color.argb(90, 255, 255, 255)
        strokeWidth = 1f * resources.displayMetrics.density
        style = Paint.Style.STROKE
    }

    init {
        isClickable = false
        isFocusable = false
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0f || h <= 0f) return

        for (i in 1..2) {
            val x = w * i / 3f
            canvas.drawLine(x, 0f, x, h, paint)
            val y = h * i / 3f
            canvas.drawLine(0f, y, w, y, paint)
        }
    }
}
