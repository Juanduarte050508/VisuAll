package com.visuall.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.roundToInt

class VerticalZoomSlider @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(95, 245, 241, 232)
        style = Paint.Style.FILL
    }
    private val activePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(230, 168, 23)
        style = Paint.Style.FILL
    }
    private val thumbPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(12, 12, 12)
        style = Paint.Style.FILL
    }
    private val thumbStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(230, 168, 23)
        style = Paint.Style.STROKE
        strokeWidth = dp(2f)
    }

    private var listener: ((progress: Int, fromUser: Boolean) -> Unit)? = null

    private var progressValue: Int = 0

    var progress: Int
        get() = progressValue
        set(value) {
            setProgressInternal(value, fromUser = false)
        }

    fun setOnProgressChangedListener(
        listener: ((progress: Int, fromUser: Boolean) -> Unit)?
    ) {
        this.listener = listener
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val centerX = width / 2f
        val top = paddingTop + dp(10f)
        val bottom = height - paddingBottom - dp(10f)
        val trackWidth = dp(6f)
        val radius = trackWidth / 2f
        val thumbRadius = dp(10f)
        val available = (bottom - top).coerceAtLeast(1f)
        val thumbY = bottom - available * (progressValue / 100f)

        canvas.drawRoundRect(
            RectF(centerX - trackWidth / 2f, top, centerX + trackWidth / 2f, bottom),
            radius,
            radius,
            trackPaint
        )
        canvas.drawRoundRect(
            RectF(centerX - trackWidth / 2f, thumbY, centerX + trackWidth / 2f, bottom),
            radius,
            radius,
            activePaint
        )
        canvas.drawCircle(centerX, thumbY, thumbRadius, thumbPaint)
        canvas.drawCircle(centerX, thumbY, thumbRadius, thumbStrokePaint)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (!isEnabled) return false
        parent?.requestDisallowInterceptTouchEvent(true)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN,
            MotionEvent.ACTION_MOVE,
            MotionEvent.ACTION_UP,
            MotionEvent.ACTION_CANCEL -> {
                isPressed = event.actionMasked != MotionEvent.ACTION_UP &&
                    event.actionMasked != MotionEvent.ACTION_CANCEL
                val top = paddingTop + dp(10f)
                val bottom = height - paddingBottom - dp(10f)
                val y = event.y.coerceIn(top, bottom)
                val available = (bottom - top).coerceAtLeast(1f)
                val next = (((bottom - y) / available) * 100f).roundToInt()
                setProgressInternal(next, fromUser = true)
                if (event.actionMasked == MotionEvent.ACTION_UP) performClick()
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    private fun setProgressInternal(value: Int, fromUser: Boolean) {
        val next = value.coerceIn(0, 100)
        if (progressValue == next && !fromUser) return
        progressValue = next
        invalidate()
        listener?.invoke(progressValue, fromUser)
    }

    private fun dp(value: Float): Float {
        return value * resources.displayMetrics.density
    }
}
