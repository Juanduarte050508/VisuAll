package com.visuall.app.libras

// Resultado de uma classificação (letra estática/dinâmica, sinal de corpo ou
// calibração pessoal) — compartilhado entre os módulos de reconhecimento
// (LetraEngine, BodyGestureEngine).
internal data class Prediction(
    val letra: String,
    val confianca: Float,
    val modo: String,
    val margem: Float = 1f
)
