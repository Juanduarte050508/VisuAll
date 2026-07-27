package com.visuall.app.libras

import android.content.Context
import java.io.File
import kotlin.math.abs

class CalibrationTrainingStore(
    private val context: Context,
    private val staticLabels: List<String>,
    private val dynamicLabels: List<String>
) {
    private companion object {
        // Teto de linhas guardadas por arquivo — ver appendCapped().
        const val MAX_STATIC_SAMPLE_ROWS = 6_000
        const val MAX_DYNAMIC_SAMPLE_ROWS = 1_500
    }

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
        val header = staticHeader()
        val now = System.currentTimeMillis()
        val newLines = frames.mapIndexed { index, frame ->
            buildString {
                append(now + index); append(','); append(letter); append(",phone_calibration")
                frame.forEach { append(','); append(it) }
            }
        }
        appendCapped(file, header, newLines, MAX_STATIC_SAMPLE_ROWS)
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
        val header = dynamicHeader()
        val now = System.currentTimeMillis()
        val newLines = windows.mapIndexed { windowIndex, window ->
            buildString {
                append(now + windowIndex); append(','); append(letter); append(",phone_dynamic")
                window.forEach { frame -> frame.forEach { append(','); append(it) } }
            }
        }
        appendCapped(file, header, newLines, MAX_DYNAMIC_SAMPLE_ROWS)
        return windows.size
    }

    private fun staticHeader(): String = buildString {
        append("timestamp,label,source")
        for (i in 0 until LibrasAnalyzer.FEATURES_ESTATICO) append(",f$i")
    }

    private fun dynamicHeader(): String = buildString {
        append("timestamp,label,source")
        for (i in 0 until LibrasAnalyzer.FEATURES_DINAMICO) append(",f$i")
    }

    // Cada calibração acrescenta linhas, sem limite, para sempre — em uso
    // contínuo o CSV cresceria indefinidamente (linhas dinâmicas têm 420
    // valores cada). Em vez de só anexar, mantemos as MAX_*_SAMPLE_ROWS
    // linhas mais recentes: reescreve o arquivo com (dados antigos + novos)
    // cortado às últimas N linhas. É mais I/O que um append puro, mas
    // calibração é uma ação manual e rara do usuário, não por frame.
    private fun appendCapped(file: File, header: String, newLines: List<String>, maxRows: Int) {
        if (newLines.isEmpty()) return
        val existing = if (file.exists() && file.length() > 0L) file.readLines().drop(1) else emptyList()
        val kept = (existing + newLines).takeLast(maxRows)
        val text = buildString {
            appendLine(header)
            kept.forEach { appendLine(it) }
        }
        runCatching { file.writeText(text) }
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
