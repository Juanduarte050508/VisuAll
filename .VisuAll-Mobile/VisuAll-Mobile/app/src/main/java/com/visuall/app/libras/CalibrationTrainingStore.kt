package com.visuall.app.libras

import android.content.Context
import java.io.File
import kotlin.math.abs

class CalibrationTrainingStore(
    private val context: Context,
    private val staticLabels: List<String>,
    private val dynamicLabels: List<String>
) {
    data class Match(val letter: String, val distance: Float)

    private val calibrationPrefs =
        context.getSharedPreferences("visuall_libras_calibration", Context.MODE_PRIVATE)
    private val trainingPrefs =
        context.getSharedPreferences("visuall_libras_training", Context.MODE_PRIVATE)
    private val references = mutableMapOf<String, FloatArray>()
    private var loaded = false

    fun ensureLoaded() {
        if (loaded) return
        references.clear()
        staticLabels.forEach { letter ->
            calibrationPrefs.getString(calibrationKey(letter), null)
                ?.let { decodeCalibration(it) }
                ?.let { references[letter] = it }
        }
        loaded = true
    }

    fun saveStaticCalibration(letter: String, reference: FloatArray) {
        ensureLoaded()
        references[letter] = reference.copyOf()
        calibrationPrefs.edit()
            .putString(calibrationKey(letter), encodeCalibration(reference))
            .apply()
    }

    fun bestStaticMatch(candidates: List<FloatArray>, maxDistance: Float): Match? {
        ensureLoaded()
        if (references.isEmpty()) return null

        var bestLetter = "-"
        var bestDistance = Float.MAX_VALUE
        references.forEach { (letter, reference) ->
            candidates.forEach { candidate ->
                val distance = landmarkDistance(candidate, reference)
                if (distance < bestDistance) {
                    bestDistance = distance
                    bestLetter = letter
                }
            }
        }
        return if (bestDistance <= maxDistance) Match(bestLetter, bestDistance) else null
    }

    fun calibrationCount(): Int {
        ensureLoaded()
        return references.size
    }

    fun totalSampleCount(): Int =
        staticLabels.sumOf { letter -> sampleCount(letter) }

    fun sampleCount(letter: String): Int =
        trainingPrefs.getInt(trainingKey(letter), 0)

    fun incrementSampleCount(letter: String, amount: Int) {
        val key = trainingKey(letter)
        trainingPrefs.edit().putInt(key, trainingPrefs.getInt(key, 0) + amount).apply()
    }

    fun staticDatasetPath(): String =
        File(context.filesDir, LibrasAnalyzer.TRAINING_FILE_NAME).absolutePath

    fun dynamicDatasetPath(): String =
        File(context.filesDir, LibrasAnalyzer.DYNAMIC_TRAINING_FILE_NAME).absolutePath

    fun clear() {
        references.clear()
        loaded = true
        calibrationPrefs.edit().clear().apply()
        trainingPrefs.edit().clear().apply()
        runCatching { File(context.filesDir, LibrasAnalyzer.TRAINING_FILE_NAME).delete() }
        runCatching { File(context.filesDir, LibrasAnalyzer.DYNAMIC_TRAINING_FILE_NAME).delete() }
    }

    fun saveStaticSamples(letter: String, frames: List<FloatArray>) {
        if (frames.isEmpty()) return
        val file = File(context.filesDir, LibrasAnalyzer.TRAINING_FILE_NAME)
        val writeHeader = !file.exists() || file.length() == 0L
        val now = System.currentTimeMillis()
        val text = buildString {
            if (writeHeader) {
                append("timestamp,label,source")
                for (i in 0 until LibrasAnalyzer.FEATURES_ESTATICO) append(",f$i")
                appendLine()
            }
            frames.forEachIndexed { index, frame ->
                append(now + index)
                append(',')
                append(letter)
                append(",phone_calibration")
                frame.forEach { value ->
                    append(',')
                    append(value)
                }
                appendLine()
            }
        }
        runCatching { file.appendText(text) }
    }

    fun saveDynamicSamples(letter: String, frames: List<FloatArray>): Int {
        if (letter !in dynamicLabels) return 0
        val windows = frames.windowed(
            size = LibrasAnalyzer.JANELA_MLP,
            step = 1,
            partialWindows = false
        )
        if (windows.isEmpty()) return 0

        val file = File(context.filesDir, LibrasAnalyzer.DYNAMIC_TRAINING_FILE_NAME)
        val writeHeader = !file.exists() || file.length() == 0L
        val now = System.currentTimeMillis()
        val text = buildString {
            if (writeHeader) {
                append("timestamp,label,source")
                for (i in 0 until LibrasAnalyzer.FEATURES_DINAMICO) append(",f$i")
                appendLine()
            }
            windows.forEachIndexed { windowIndex, window ->
                append(now + windowIndex)
                append(',')
                append(letter)
                append(",phone_dynamic")
                window.forEach { frame ->
                    frame.forEach { value ->
                        append(',')
                        append(value)
                    }
                }
                appendLine()
            }
        }
        runCatching { file.appendText(text) }
        return windows.size
    }

    private fun calibrationKey(letter: String): String = "letter_${letter.uppercase()}"

    private fun trainingKey(letter: String): String = "samples_${letter.uppercase()}"

    private fun encodeCalibration(data: FloatArray): String =
        data.joinToString(separator = ",")

    private fun decodeCalibration(raw: String): FloatArray? {
        val parts = raw.split(",")
        if (parts.size != LibrasAnalyzer.FEATURES_ESTATICO) return null
        val result = FloatArray(LibrasAnalyzer.FEATURES_ESTATICO)
        parts.forEachIndexed { index, value ->
            result[index] = value.toFloatOrNull() ?: return null
        }
        return result
    }

    private fun landmarkDistance(a: FloatArray, b: FloatArray): Float {
        val size = minOf(a.size, b.size)
        if (size == 0) return Float.MAX_VALUE
        var total = 0f
        for (i in 0 until size) {
            total += abs(a[i] - b[i])
        }
        return total / size
    }
}
