package com.visuall.app.libras

import android.content.Context
import android.util.Log
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
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

    // Toda escrita em disco passa por aqui. Uma thread só (não um pool) de
    // propósito: garante que as gravações acontecem em ordem e que nunca há
    // duas mexendo no mesmo arquivo ao mesmo tempo — inclusive o clear(),
    // que também é enfileirado, senão poderia apagar o arquivo no meio de uma
    // gravação ainda pendente.
    private val ioExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "visuall-calibracao-io")
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
        // Vai pra MESMA fila das gravações: se apagasse aqui direto, uma
        // gravação ainda pendente rodaria depois e recriaria o arquivo que o
        // usuário acabou de mandar zerar.
        ioExecutor.execute {
            runCatching { File(context.filesDir, LibrasAnalyzer.TRAINING_FILE_NAME).delete() }
            runCatching { File(context.filesDir, LibrasAnalyzer.DYNAMIC_TRAINING_FILE_NAME).delete() }
        }
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

    // Cada calibração acrescenta linhas, e sem teto o CSV cresceria pra
    // sempre (uma linha dinâmica tem 420 números). Guardamos as
    // MAX_*_SAMPLE_ROWS mais recentes.
    //
    // Duas coisas importam aqui, e as duas já causaram problema:
    //
    // 1. Roda FORA da thread da tela. Isto é chamado pelo botão de salvar
    //    calibração; fazendo o I/O ali, o app congelava a cada amostra salva
    //    — e piorava conforme o arquivo enchia, dando a impressão de que "foi
    //    ficando lento". Quem grava dezenas de amostras seguidas sente isso o
    //    tempo todo.
    // 2. ANEXA em vez de reescrever tudo. A versão anterior lia e regravava o
    //    arquivo inteiro a cada salvamento (~5 MB no caso dinâmico cheio) só
    //    pra cortar as linhas velhas. Agora o custo normal é o das linhas
    //    novas, e a poda só acontece quando o arquivo realmente passa do
    //    teto.
    //
    // A contagem de linhas fica guardada nas prefs pra não precisar ler o
    // arquivo só pra saber o tamanho; se sumir (instalação antiga), é
    // recalculada uma vez.
    private fun appendCapped(file: File, header: String, newLines: List<String>, maxRows: Int) {
        if (newLines.isEmpty()) return
        ioExecutor.execute {
            runCatching {
                val existiaAntes = file.exists() && file.length() > 0L
                if (!existiaAntes) {
                    file.writeText(buildString {
                        appendLine(header)
                        newLines.forEach { appendLine(it) }
                    })
                    guardarContagem(file, newLines.size)
                    return@runCatching
                }

                file.appendText(buildString { newLines.forEach { appendLine(it) } })
                val total = contagemDeLinhas(file) + newLines.size
                if (total <= maxRows) {
                    guardarContagem(file, total)
                } else {
                    podar(file, header, maxRows)
                }
            }.onFailure { erro ->
                // Falha aqui não pode derrubar o app: a pessoa está no meio de
                // uma sessão de gravação e perder uma amostra é melhor que
                // perder a sessão. Mas precisa aparecer no log, senão vira
                // "gravei 50 amostras e o treino não achou nada".
                Log.e("CalibrationStore", "Falha ao gravar amostras em ${file.name}", erro)
            }
        }
    }

    // Reescreve o arquivo mantendo só as últimas maxRows linhas. Custa uma
    // leitura completa, mas só acontece quando o teto é ultrapassado.
    private fun podar(file: File, header: String, maxRows: Int) {
        val mantidas = file.readLines().drop(1).takeLast(maxRows)
        file.writeText(buildString {
            appendLine(header)
            mantidas.forEach { appendLine(it) }
        })
        guardarContagem(file, mantidas.size)
    }

    private fun chaveContagem(file: File): String = "rows_${file.name}"

    private fun contagemDeLinhas(file: File): Int {
        val guardada = trainingPrefs.getInt(chaveContagem(file), -1)
        if (guardada >= 0) return guardada
        // Sem contagem guardada (arquivo de uma versão anterior): conta uma
        // vez e memoriza.
        val contadas = if (file.exists()) (file.readLines().size - 1).coerceAtLeast(0) else 0
        guardarContagem(file, contadas)
        return contadas
    }

    private fun guardarContagem(file: File, linhas: Int) {
        trainingPrefs.edit().putInt(chaveContagem(file), linhas).apply()
    }

    // Espera as gravações pendentes terminarem. Chamado quando o
    // reconhecimento é encerrado, pra não perder a última amostra salva.
    fun close() {
        ioExecutor.shutdown()
        runCatching {
            if (!ioExecutor.awaitTermination(3, TimeUnit.SECONDS)) {
                Log.w("CalibrationStore", "Gravacoes pendentes nao terminaram a tempo")
            }
        }
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
