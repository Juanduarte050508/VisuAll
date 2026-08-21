package com.visuall.app.ui

import android.animation.ArgbEvaluator
import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator

/**
 * Moldura de escaneamento do modo Libras.
 *
 * A cor da moldura reflete o nível de feedback do reconhecimento
 * (neutro / bom / alerta), para que o usuário não precise desviar o
 * olhar da mão para ler o texto de status embaixo da tela.
 */
class ScanFrameView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0
) : View(context, attrs, defStyle) {

    companion object {
        const val FEEDBACK_NEUTRO = 0
        const val FEEDBACK_BOM = 1
        const val FEEDBACK_ALERTA = 2

        private val COLOR_NEUTRO = Color.parseColor("#E8A020")
        private val COLOR_BOM = Color.parseColor("#4CD97A")
        private val COLOR_ALERTA = Color.parseColor("#FF6B5E")
    }

    private var currentLevel = FEEDBACK_NEUTRO
    private var animatedColor = COLOR_NEUTRO
    private var colorAnimator: ValueAnimator? = null

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 8f
        // BUTT (e nao ROUND): a ponta arredondada afinava o traco no fim de
        // cada braco e dava aspecto de esfumacado/cortado. Com BUTT a linha
        // termina reta e solida.
        strokeCap = Paint.Cap.BUTT
        strokeJoin = Paint.Join.MITER
    }

    private val armLen get() = width * 0.24f

    init {
        applyColor(COLOR_NEUTRO)
    }

    /** Atualiza a cor da moldura conforme o nível de feedback do reconhecimento. */
    fun setFeedbackLevel(level: Int) {
        if (level == currentLevel) return
        currentLevel = level
        val target = when (level) {
            FEEDBACK_BOM -> COLOR_BOM
            FEEDBACK_ALERTA -> COLOR_ALERTA
            else -> COLOR_NEUTRO
        }
        colorAnimator?.cancel()
        colorAnimator = ValueAnimator.ofObject(ArgbEvaluator(), animatedColor, target).apply {
            duration = 180L
            interpolator = LinearInterpolator()
            addUpdateListener {
                applyColor(it.animatedValue as Int)
                invalidate()
            }
            start()
        }
    }

    private fun applyColor(color: Int) {
        animatedColor = color
        paint.color = color
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        drawCorners(canvas, paint)
    }

    private fun drawCorners(canvas: Canvas, p: Paint) {
        val w = width.toFloat()
        val h = height.toFloat()
        val a = armLen
        val r = 58f

        canvas.drawLine(0f, a, 0f, r, p)
        canvas.drawArc(0f, 0f, r * 2, r * 2, 180f, 90f, false, p)
        canvas.drawLine(r, 0f, a, 0f, p)

        canvas.drawLine(w - a, 0f, w - r, 0f, p)
        canvas.drawArc(w - r * 2, 0f, w, r * 2, 270f, 90f, false, p)
        canvas.drawLine(w, r, w, a, p)

        canvas.drawLine(0f, h - a, 0f, h - r, p)
        canvas.drawArc(0f, h - r * 2, r * 2, h, 90f, 90f, false, p)
        canvas.drawLine(r, h, a, h, p)

        canvas.drawLine(w - a, h, w - r, h, p)
        canvas.drawArc(w - r * 2, h - r * 2, w, h, 0f, 90f, false, p)
        canvas.drawLine(w, h - r, w, h - a, p)
    }
}
