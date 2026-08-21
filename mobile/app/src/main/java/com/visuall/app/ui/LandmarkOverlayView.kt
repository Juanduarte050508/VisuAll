package com.visuall.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

/**
 * Desenha as linhas de reconhecimento (esqueleto da mão e do corpo) por cima
 * do preview da câmera. Recebe landmarks normalizados (0..1) no MESMO espaço
 * do preview (imagem já rotacionada e espelhada como a câmera frontal mostra),
 * então basta mapear com a mesma lógica FILL_CENTER que a PreviewView usa
 * (app:scaleType="fillCenter"): a View fica cheia e o excesso do sensor é
 * recortado nas sobras.
 *
 * A proporção do conteúdo NÃO é fixa — ela vem do analyzer a cada frame
 * (muda entre retrato e paisagem), então o desenho acompanha a imagem real.
 */
class LandmarkOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0
) : View(context, attrs, defStyle) {

    companion object {
        // Proporção inicial (retrato 3:4) até o primeiro frame chegar.
        private const val CONTENT_ASPECT_PADRAO = 3f / 4f  // largura / altura

        // Topologia da mão do MediaPipe (21 pontos).
        private val HAND_CONNECTIONS = arrayOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 4,          // polegar
            0 to 5, 5 to 6, 6 to 7, 7 to 8,          // indicador
            5 to 9, 9 to 10, 10 to 11, 11 to 12,     // médio
            9 to 13, 13 to 14, 14 to 15, 15 to 16,   // anelar
            13 to 17, 17 to 18, 18 to 19, 19 to 20,  // mínimo
            0 to 17                                  // base da palma
        )

        // Subconjunto útil da pose (tronco + braços) — 33 pontos do MediaPipe.
        private val POSE_CONNECTIONS = arrayOf(
            11 to 12,                 // ombros
            11 to 13, 13 to 15,       // braço esquerdo
            12 to 14, 14 to 16,       // braço direito
            11 to 23, 12 to 24, 23 to 24  // tronco
        )
    }

    private var hands: List<FloatArray> = emptyList()
    private var pose: FloatArray? = null
    // Proporção (largura/altura) da imagem que o analyzer processou.
    private var contentAspect = CONTENT_ASPECT_PADRAO

    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#E8A020")
        style = Paint.Style.STROKE
        strokeWidth = 6f
        strokeCap = Paint.Cap.ROUND
    }
    private val jointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#F5C842")
        style = Paint.Style.FILL
    }
    private val poseLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#5C9CE8")
        style = Paint.Style.STROKE
        strokeWidth = 6f
        strokeCap = Paint.Cap.ROUND
    }
    private val poseJointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#8FC0FF")
        style = Paint.Style.FILL
    }

    fun update(hands: List<FloatArray>, pose: FloatArray?, frameAspect: Float) {
        this.hands = hands
        this.pose = pose
        if (frameAspect > 0f) this.contentAspect = frameAspect
        postInvalidate()
    }

    fun clear() {
        this.hands = emptyList()
        this.pose = null
        postInvalidate()
    }

    // Mapeamento FILL_CENTER: cobre a View inteira, centralizado, recortando
    // o excesso — igual ao app:scaleType="fillCenter" da PreviewView.
    private var dispW = 0f
    private var dispH = 0f
    private var offX = 0f
    private var offY = 0f

    private fun recomputeMapping() {
        val vw = width.toFloat()
        val vh = height.toFloat()
        if (vw <= 0f || vh <= 0f) return
        val viewAspect = vw / vh
        if (viewAspect > contentAspect) {
            // View mais larga que o conteúdo: cobre pela largura e recorta em cima/baixo.
            dispW = vw
            dispH = vw / contentAspect
            offX = 0f
            offY = (vh - dispH) / 2f
        } else {
            // View mais estreita/alta: cobre pela altura e recorta nas laterais.
            dispH = vh
            dispW = vh * contentAspect
            offY = 0f
            offX = (vw - dispW) / 2f
        }
    }

    private fun mapX(nx: Float) = offX + nx * dispW
    private fun mapY(ny: Float) = offY + ny * dispH

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (hands.isEmpty() && pose == null) return
        recomputeMapping()

        pose?.let { p ->
            drawSkeleton(canvas, p, POSE_CONNECTIONS, poseLinePaint, poseJointPaint, 6f)
        }
        hands.forEach { hand ->
            drawSkeleton(canvas, hand, HAND_CONNECTIONS, linePaint, jointPaint, 7f)
        }
    }

    private fun drawSkeleton(
        canvas: Canvas,
        pts: FloatArray,
        connections: Array<Pair<Int, Int>>,
        line: Paint,
        joint: Paint,
        jointRadius: Float
    ) {
        val count = pts.size / 2
        connections.forEach { (a, b) ->
            if (a < count && b < count) {
                val ax = mapX(pts[a * 2]); val ay = mapY(pts[a * 2 + 1])
                val bx = mapX(pts[b * 2]); val by = mapY(pts[b * 2 + 1])
                if (isValid(pts[a * 2], pts[a * 2 + 1]) && isValid(pts[b * 2], pts[b * 2 + 1])) {
                    canvas.drawLine(ax, ay, bx, by, line)
                }
            }
        }
        for (i in 0 until count) {
            val x = pts[i * 2]; val y = pts[i * 2 + 1]
            if (isValid(x, y)) canvas.drawCircle(mapX(x), mapY(y), jointRadius, joint)
        }
    }

    // Pontos zerados (mão/parte ausente) não são desenhados.
    private fun isValid(x: Float, y: Float): Boolean =
        (x != 0f || y != 0f) && x in -0.05f..1.05f && y in -0.05f..1.05f
}
