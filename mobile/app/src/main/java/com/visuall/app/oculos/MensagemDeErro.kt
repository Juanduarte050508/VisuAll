package com.visuall.app.oculos

/**
 * Traduz a falha crua da conexao para uma frase que diga o que FAZER.
 *
 * Por que existe: o motivo que chega aqui vem do sistema, em ingles e escrito
 * pra quem programa -- "Cleartext HTTP traffic to 192.168.15.10 not permitted",
 * "failed to connect to /192.168.4.1 (port 80) after 3000ms". Mostrar isso na
 * tela nao ajuda ninguem a resolver: nao diz se o problema e a rede, o
 * endereco, ou a placa que ainda nao ligou.
 *
 * Cada frase daqui aponta pra UMA acao. O motivo cru nao se perde -- vai pro
 * logcat, que e onde ele serve.
 */
internal object MensagemDeErro {

    fun emPortugues(motivo: String): String {
        val m = motivo.lowercase()
        return when {
            // O endereco esta certo mas nao ha ninguem escutando. Na placa e o
            // caso mais comum de todos: ela ainda esta subindo o Wi-Fi.
            "refused" in m || "failed to connect" in m || "econnrefused" in m ->
                "Ninguem atende nesse endereco. Os oculos estao ligados e o " +
                    "celular esta na rede deles?"

            // Nem chegou a sair do aparelho: o celular nao tem caminho pra la.
            "unreachable" in m || "enetunreach" in m || "no route" in m ->
                "O celular nao alcanca essa rede. Confira o Wi-Fi."

            "unable to resolve host" in m || "unknownhost" in m ->
                "Endereco invalido. Use o numero do IP, como 192.168.4.1."

            // Conectou e parou de vir imagem: distancia, bateria ou a placa
            // travou. Reconectar sozinho ja esta acontecendo.
            "timeout" in m || "timed out" in m || "connection abort" in m ||
                "connection reset" in m || "unexpected end" in m ->
                "A imagem parou de chegar. Chegue mais perto dos oculos."

            // Respondeu, mas nao com video. Quase sempre falta o /stream.
            "nao e mjpeg" in m || "content-type" in m ->
                "Esse endereco responde, mas nao e video. Falta o /stream no fim?"

            "http 404" in m -> "Endereco existe mas nao tem /stream nele."

            // So aparece se o network_security_config sumir do manifesto.
            "cleartext" in m ->
                "O Android bloqueou a conexao. E preciso reinstalar o app."

            else -> "Nao consegui conectar aos oculos."
        }
    }
}
