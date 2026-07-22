package com.visuall.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class ScanFrameView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0
) : View(context, attrs, defStyle) {

    private val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#66E8A020")
        style = Paint.Style.STROKE
        strokeWidth = 14f
        strokeCap = Paint.Cap.ROUND
        setShadowLayer(22f, 0f, 0f, Color.parseColor("#AAE8A020"))
    }

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#E8A020")
        style = Paint.Style.STROKE
        strokeWidth = 7f
        strokeCap = Paint.Cap.ROUND
    }

    private val armLen get() = width * 0.20f

    init {
        setLayerType(LAYER_TYPE_SOFTWARE, null)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        drawCorners(canvas, glowPaint)
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
