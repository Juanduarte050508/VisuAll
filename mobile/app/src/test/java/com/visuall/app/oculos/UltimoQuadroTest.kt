package com.visuall.app.oculos

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class UltimoQuadroTest {

    @Test
    fun `consumir devolve o que foi publicado`() {
        val caixa = UltimoQuadro<String>()
        caixa.publicar("a")
        assertEquals("a", caixa.consumir())
    }

    @Test
    fun `guarda so o mais novo e joga o antigo fora`() {
        // O ponto da classe: se o reconhecimento ficou pra tras, ele deve pular
        // pro quadro atual, nao processar a fila de atrasados.
        val caixa = UltimoQuadro<String>()
        assertFalse("o primeiro nao descarta nada", caixa.publicar("velho"))
        assertTrue("o segundo descarta o primeiro", caixa.publicar("novo"))
        assertTrue(caixa.publicar("mais novo"))

        assertEquals("mais novo", caixa.consumir())
    }

    @Test
    fun `depois de consumir a caixa fica vazia`() {
        val caixa = UltimoQuadro<String>()
        caixa.publicar("a")
        assertEquals("a", caixa.consumir())
        assertFalse("publicar de novo nao descarta nada", caixa.publicar("b"))
    }

    @Test(timeout = 2_000)
    fun `consumir espera ate chegar quadro`() {
        val caixa = UltimoQuadro<String>()
        val pegou = CountDownLatch(1)
        var resultado: String? = null

        val t = Thread {
            resultado = caixa.consumir()
            pegou.countDown()
        }
        t.start()

        // Se consumir nao bloqueasse, a thread ja teria terminado com null.
        assertFalse("nao podia ter voltado ainda", pegou.await(150, TimeUnit.MILLISECONDS))

        caixa.publicar("chegou")
        assertTrue("devia acordar ao publicar", pegou.await(1, TimeUnit.SECONDS))
        assertEquals("chegou", resultado)
    }

    @Test(timeout = 2_000)
    fun `fechar acorda quem espera e devolve null`() {
        // Sem isto, a thread de analise ficaria pendurada pra sempre quando o
        // usuario saisse da tela -- e o app so morreria junto com o processo.
        val caixa = UltimoQuadro<String>()
        val pegou = CountDownLatch(1)
        var resultado: String? = "ainda nao"

        Thread {
            resultado = caixa.consumir()
            pegou.countDown()
        }.start()

        Thread.sleep(100)
        caixa.fechar()

        assertTrue("fechar tem que acordar quem espera", pegou.await(1, TimeUnit.SECONDS))
        assertNull("depois de fechado, consumir devolve null", resultado)
    }

    @Test
    fun `depois de fechado nao aceita nem entrega mais nada`() {
        val caixa = UltimoQuadro<String>()
        caixa.fechar()
        assertFalse(caixa.publicar("tarde demais"))
        assertNull(caixa.consumir())
    }

    @Test(timeout = 5_000)
    fun `nao perde o ultimo quadro com publicacao concorrente`() {
        // Aproxima o caso real: uma thread empurrando quadro sem parar e outra
        // consumindo mais devagar. Nenhuma das duas pode travar, e o consumidor
        // tem que ver sempre um valor valido -- nunca null no meio do caminho.
        val caixa = UltimoQuadro<Int>()
        val total = 2_000
        val consumidos = mutableListOf<Int>()

        val consumidor = Thread {
            while (true) {
                val v = caixa.consumir() ?: break
                consumidos.add(v)
            }
        }
        consumidor.start()

        for (i in 1..total) caixa.publicar(i)
        Thread.sleep(50)
        caixa.fechar()
        consumidor.join(2_000)

        assertTrue("tem que ter consumido alguma coisa", consumidos.isNotEmpty())
        assertTrue("nunca pode consumir mais do que foi publicado",
            consumidos.size <= total)
        assertEquals("os valores tem que sair em ordem crescente (sem repetir)",
            consumidos.sorted().distinct(), consumidos)
    }
}
