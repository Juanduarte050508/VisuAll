package com.visuall.app.libras

import android.content.Context
import android.util.Log
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker.FaceLandmarkerOptions
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlin.math.sqrt

// Marcador de sobrancelha (frase vira pergunta) — extraído do LibrasAnalyzer.
// Roda nos DOIS modos (igual ao Holistic do Python, que sempre lê o rosto),
// mas só a cada FACE_DETECT_STRIDE frames, e se a inicialização falhar
// (aparelho fraco/sem o asset) o app segue sem o marcador em vez de quebrar
// — mesma filosofia defensiva do BodyGestureEngine.
internal class FaceMarkerEngine(private val context: Context) {

    private var faceLandmarker: FaceLandmarker? = null
    private var faceModelIndisponivel = false
    private val bufferSobrancelha = ArrayDeque<Float>()
    private var interrogativoAtivo = false
    private var faceFrameCounter = 0

    // Chamado uma vez por frame (nos dois modos). `onChange` só é disparado
    // quando o estado da sobrancelha muda, nunca a cada frame.
    fun step(mpImage: MPImage, timestamp: Long, onChange: (ativo: Boolean) -> Unit) {
        faceFrameCounter++
        if (faceFrameCounter % LibrasAnalyzer.FACE_DETECT_STRIDE != 0) return

        val faceResult = ensureLoaded()?.detectForVideo(mpImage, timestamp)
        val sobrancelhaAtiva = lerMarcador(faceResult)
        if (sobrancelhaAtiva != interrogativoAtivo) {
            interrogativoAtivo = sobrancelhaAtiva
            onChange(sobrancelhaAtiva)
        }
    }

    private fun faceOptions(delegate: Delegate) = FaceLandmarkerOptions.builder()
        .setBaseOptions(BaseOptions.builder()
            .setModelAssetPath("face_landmarker.task")
            .setDelegate(delegate)
            .build())
        .setRunningMode(RunningMode.VIDEO)
        .setNumFaces(1)
        .setMinFaceDetectionConfidence(0.5f)
        .setMinFacePresenceConfidence(0.5f)
        .setMinTrackingConfidence(0.5f)
        .build()

    private fun ensureLoaded(): FaceLandmarker? {
        faceLandmarker?.let { return it }
        if (faceModelIndisponivel) return null

        // GPU primeiro (mais rápido pro 3º modelo rodando por frame); se o
        // aparelho não aceitar, cai pra CPU antes de desistir de vez — mesma
        // ideia do HandLandmarker no LibrasAnalyzer.
        return try {
            FaceLandmarker.createFromOptions(context, faceOptions(Delegate.GPU))
                .also { faceLandmarker = it }
        } catch (error: Throwable) {
            try {
                Log.w("FaceMarkerEngine", "GPU indisponivel pro FaceLandmarker, usando CPU", error)
                FaceLandmarker.createFromOptions(context, faceOptions(Delegate.CPU))
                    .also { faceLandmarker = it }
            } catch (error2: Throwable) {
                faceLandmarker = null
                faceModelIndisponivel = true
                null
            }
        }
    }

    // Porta fiel de ler_marcador (app.py): sobrancelha levantada em relação à
    // distância entre os cantos externos dos olhos (escala invariante à
    // distância da câmera), suavizada numa janela de JANELA_SOBR frames.
    private fun lerMarcador(faceResult: FaceLandmarkerResult?): Boolean {
        val rosto = faceResult?.faceLandmarks()?.firstOrNull() ?: return false
        val maxIdx = maxOf(
            LibrasAnalyzer.IDX_BROW_L, LibrasAnalyzer.IDX_BROW_R,
            LibrasAnalyzer.IDX_EYE_TOP_L, LibrasAnalyzer.IDX_EYE_TOP_R,
            LibrasAnalyzer.IDX_EYE_OUT_L, LibrasAnalyzer.IDX_EYE_OUT_R
        )
        if (rosto.size <= maxIdx) return false

        val dx = rosto[LibrasAnalyzer.IDX_EYE_OUT_L].x() - rosto[LibrasAnalyzer.IDX_EYE_OUT_R].x()
        val dy = rosto[LibrasAnalyzer.IDX_EYE_OUT_L].y() - rosto[LibrasAnalyzer.IDX_EYE_OUT_R].y()
        val escala = sqrt(dx * dx + dy * dy)
        if (escala < 1e-6f) return false

        val gl = rosto[LibrasAnalyzer.IDX_EYE_TOP_L].y() - rosto[LibrasAnalyzer.IDX_BROW_L].y()
        val gr = rosto[LibrasAnalyzer.IDX_EYE_TOP_R].y() - rosto[LibrasAnalyzer.IDX_BROW_R].y()
        bufferSobrancelha.addLast(((gl + gr) / 2f) / escala)
        while (bufferSobrancelha.size > LibrasAnalyzer.JANELA_SOBR) bufferSobrancelha.removeFirst()

        val suavizado = bufferSobrancelha.average().toFloat()
        return suavizado >= LibrasAnalyzer.LIMIAR_SOBRANCELHA
    }

    // NOTA: antes desta refatoração, o FaceLandmarker nunca era fechado em
    // close() (só handLandmarker/poseLandmarker/bodyInterpreter eram) — um
    // vazamento real do detector nativo a cada troca de câmera (o Fragment
    // cria um LibrasAnalyzer novo por bind). Corrigido aqui.
    fun close() {
        runCatching { faceLandmarker?.close() }
            .onFailure { Log.w("FaceMarkerEngine", "Falha ao fechar FaceLandmarker", it) }
        faceLandmarker = null
    }
}
