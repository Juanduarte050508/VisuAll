package com.visuall.app.oculos

import java.net.URI

/**
 * O que se sabe de fabrica sobre os oculos, num lugar so.
 *
 * Estes valores tem par do outro lado, em
 * `esp32/firmware/oculos_camera/oculos_camera.ino`. Mudar aqui sem mudar la
 * (ou o contrario) faz o app procurar uma rede que nao existe.
 */
internal object EnderecoDosOculos {

    /** Rede que a placa cria. Ver `REDE_NOME` no firmware. */
    const val REDE = "VisuAll-Oculos"

    /** Ver `REDE_SENHA` no firmware. */
    const val SENHA = "visuall2026"

    /** Ver `IP_PLACA` no firmware. E fixo pra este endereco nunca mudar. */
    const val IP = "192.168.4.1"

    const val URL = "http://$IP/stream"

    /**
     * O endereco aponta pra propria placa?
     *
     * Decide se o app tenta entrar na rede dos oculos sozinho ou se usa o Wi-Fi
     * em que o celular ja esta. Nao e a mesma coisa: entrar sozinho so faz
     * sentido quando o destino e a placa, e tentar isso apontando pro mock no
     * PC deixaria o app procurando pra sempre uma rede que nao existe naquele
     * lugar.
     *
     * Endereco quebrado devolve false: o certo, na duvida, e usar a rede em que
     * o celular ja esta -- que e como funcionava antes desta funcao existir.
     */
    fun ehAPlaca(url: String): Boolean = try {
        URI(url.trim()).host == IP
    } catch (_: Exception) {
        false
    }
}
