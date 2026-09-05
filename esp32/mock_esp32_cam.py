"""Dubl~e do ESP32-S3-CAM: serve um stream MJPEG igual ao dos oculos.

Por que existe: o app precisa aprender a receber imagem pela REDE em vez da
camera do celular, e isso e a maior parte do trabalho. Esperar a placa chegar
pra so entao comecar deixaria o app parado. Este script fala exatamente o mesmo
protocolo que o firmware CameraWebServer do ESP32 vai falar -- mesmo formato
multipart, mesma resolucao, mesmo cabecalho -- entao o app nao sabe a diferenca.

Quando a placa chegar, muda so o endereco: o codigo do app continua igual.

Uso:
    python mock_esp32_cam.py                 # webcam, se houver; senao sintetico
    python mock_esp32_cam.py video.mp4       # um video em loop
    python mock_esp32_cam.py --sintetico     # forca o gerador, sem camera
    python mock_esp32_cam.py --porta 8081

No navegador (PC ou celular):
    http://<IP_DO_PC>:8080/         pagina de teste, pra confirmar que funciona
    http://<IP_DO_PC>:8080/stream   o stream em si, que o app vai consumir
"""

import argparse
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# Estes valores imitam o firmware. Mexer aqui sem mexer no firmware faz o app
# ser testado contra uma coisa e rodar contra outra -- que e o tipo de erro
# que so aparece no dia da montagem.
LARGURA, ALTURA = 320, 240
FPS = 15
QUALIDADE_JPEG = 88          # 0-100 do OpenCV; ~ o "quality 12" do esp_camera
BOUNDARY = "123456789000000000000987654321"   # o mesmo do CameraWebServer


def _abre_captura(fonte):
    """Abre a camera pelo caminho rapido no Windows.

    Medido nesta maquina: o backend padrao (MSMF) levou 98 SEGUNDOS pra abrir a
    webcam; o DirectShow levou 1,5. Sem isto o mock parece travado -- foi
    exatamente o que aconteceu na primeira vez que rodei aqui.

    Arquivo de video nao passa por isso: o gargalo e so na camera.
    """
    if isinstance(fonte, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(fonte, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()   # driver que nao fala DirectShow: cai no padrao
        # Avisa antes de bloquear: o padrao pode demorar quase dois minutos, e
        # sem esta linha o mock parece travado justamente na hora em que ja deu
        # errado uma vez.
        print("  o DirectShow recusou a camera; tentando o backend padrao "
              "(pode levar ~1 min)...", flush=True)
    return cv2.VideoCapture(fonte)


class FonteWebcam:
    """Webcam ou arquivo de video, sempre em loop."""

    def __init__(self, fonte):
        self.captura = _abre_captura(fonte)
        if not self.captura.isOpened():
            raise RuntimeError("nao consegui abrir %r" % (fonte,))
        self.captura.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA)
        self.captura.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA)
        # Puxa um quadro AQUI, antes de dizer que deu certo. Uma webcam ja
        # ocupada por outro programa abre sem reclamar e so devolve nada na
        # leitura -- e a essa altura o erro ja virou "a fonte nao entregou
        # quadro", que nao aponta pra causa nenhuma. Aconteceu de verdade: uma
        # segunda copia deste mock rodando ao mesmo tempo que a primeira.
        ok, _ = self.captura.read()
        if not ok:
            self.captura.release()
            raise RuntimeError(
                "abri %r mas ela nao entrega imagem -- quase sempre e outro "
                "programa segurando a webcam: outra copia deste mock, Teams, "
                "Meet, o app Camera do Windows" % (fonte,))
        if not isinstance(fonte, int):
            self.captura.set(cv2.CAP_PROP_POS_FRAMES, 0)   # devolve o quadro de teste
        self.descricao = "webcam" if fonte == 0 else str(fonte)

    def quadro(self):
        ok, imagem = self.captura.read()
        if not ok:
            # Fim do arquivo: volta pro comeco. Numa webcam isto nao acontece.
            self.captura.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, imagem = self.captura.read()
            if not ok:
                return None
        return cv2.resize(imagem, (LARGURA, ALTURA))

    def fecha(self):
        self.captura.release()


class FonteSintetica:
    """Quadros gerados na hora, pra quando nao ha webcam nem video.

    Nao serve pra testar reconhecimento -- nao tem mao nenhuma aqui. Serve pra
    testar o CANO: se o app recebe, decodifica e exibe estes quadros, ele vai
    receber os do ESP32 do mesmo jeito. O contador na tela deixa obvio se o
    stream travou ou se o app esta mostrando um quadro velho.
    """

    descricao = "gerador sintetico (sem camera)"

    def __init__(self):
        self.n = 0

    def quadro(self):
        self.n += 1
        img = np.zeros((ALTURA, LARGURA, 3), dtype=np.uint8)
        img[:] = (40, 30 + (self.n * 2) % 180, 60)
        x = int((self.n * 4) % LARGURA)
        cv2.rectangle(img, (x, 0), (min(x + 20, LARGURA), ALTURA), (255, 255, 255), -1)
        cv2.putText(img, "MOCK ESP32-CAM", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "quadro %d" % self.n, (8, ALTURA - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img

    def fecha(self):
        pass


class Difusor:
    """Uma thread le a camera; todos os clientes consomem o ultimo quadro dela.

    Duas razoes, as duas descobertas rodando isto de verdade:

    1. No Windows o DirectShow fica preso a thread que criou a captura. O
       ThreadingHTTPServer atende cada requisicao numa thread nova, entao ler a
       camera de dentro do handler devolvia quadro vazio: o stream saia com o
       cabecalho certo e ZERO bytes de imagem, que e um sintoma bem dificil de
       ligar a causa.

    2. Com o navegador do PC e o celular abertos ao mesmo tempo -- que e como a
       gente testa -- dois handlers chamariam read() na mesma captura e
       disputariam os quadros um do outro.

    De quebra, codifica o JPEG uma vez por quadro em vez de uma vez por cliente.
    """

    def __init__(self, criar_fonte):
        self._criar_fonte = criar_fonte
        self._jpeg = None
        self._trava = threading.Lock()
        self._pronto = threading.Event()
        self._parar = False
        self.descricao = "?"
        self.erro = None
        self._thread = threading.Thread(target=self._laco, name="camera", daemon=True)
        self._thread.start()

    def _laco(self):
        # A fonte e criada AQUI, nao no construtor: quem abre a camera precisa
        # ser a mesma thread que vai le-la (ver o item 1 do docstring).
        try:
            fonte = self._criar_fonte()
        except Exception as erro:            # noqa: BLE001 - vai pra tela
            self.erro = erro
            self._pronto.set()
            return
        self.descricao = fonte.descricao
        intervalo = 1.0 / FPS
        try:
            while not self._parar:
                inicio = time.time()
                imagem = fonte.quadro()
                if imagem is None:
                    break
                ok, buf = cv2.imencode(
                    ".jpg", imagem, [int(cv2.IMWRITE_JPEG_QUALITY), QUALIDADE_JPEG])
                if ok:
                    with self._trava:
                        self._jpeg = buf.tobytes()
                    self._pronto.set()
                sobra = intervalo - (time.time() - inicio)
                if sobra > 0:
                    time.sleep(sobra)
        finally:
            fonte.fecha()

    def espera_primeiro(self, prazo=120.0):
        """Bloqueia ate o primeiro quadro. Levanta se a fonte nem abriu.

        O prazo e generoso de proposito: o backend padrao do Windows levou 98
        segundos pra abrir a webcam nesta maquina. Um prazo curto desistiria de
        uma camera que ia funcionar.
        """
        self._pronto.wait(prazo)
        if self.erro:
            raise self.erro
        return self._jpeg is not None

    def ultimo(self):
        with self._trava:
            return self._jpeg

    def para(self):
        self._parar = True


class Handler(BaseHTTPRequestHandler):
    difusor = None       # preenchido no main

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._pagina()
        elif self.path == "/stream":
            self._stream()
        else:
            self.send_error(404)

    def _pagina(self):
        html = ("""<!doctype html><meta charset="utf-8">
<title>Mock ESP32-CAM</title>
<body style="background:#111;color:#eee;font-family:sans-serif;text-align:center">
<h2>Mock ESP32-S3-CAM &mdash; %dx%d</h2>
<img src="/stream" style="width:640px;image-rendering:pixelated">
<p>O app deve consumir <code>/stream</code></p>
""" % (LARGURA, ALTURA)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=%s" % BOUNDARY)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        intervalo = 1.0 / FPS
        ultimo_enviado = None
        try:
            while True:
                inicio = time.time()
                jpeg = self.difusor.ultimo()
                # Nao reenvia o mesmo quadro: se a camera ficou pra tras, segurar
                # e melhor que gastar rede repetindo imagem identica.
                if jpeg is not None and jpeg is not ultimo_enviado:
                    ultimo_enviado = jpeg
                    self.wfile.write(b"--" + BOUNDARY.encode() + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(jpeg))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                sobra = intervalo - (time.time() - inicio)
                if sobra > 0:
                    time.sleep(sobra)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass       # o cliente fechou a aba; normal


    def log_message(self, formato, *args):
        pass           # senao imprime uma linha por quadro


def enderecos_locais():
    """IPs que o celular pode tentar. Nem todos servem -- ver o README."""
    encontrados = []
    for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        ip = info[4][0]
        if ip not in encontrados and not ip.startswith("127."):
            encontrados.append(ip)
    return encontrados


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", nargs="?", help="arquivo de video pra rodar em loop")
    p.add_argument("--porta", type=int, default=8080)
    p.add_argument("--sintetico", action="store_true",
                   help="nao usa camera nenhuma, gera os quadros")
    args = p.parse_args()

    def criar_fonte():
        if args.sintetico:
            return FonteSintetica()
        alvo = args.video if args.video else 0
        try:
            return FonteWebcam(alvo)
        except RuntimeError as erro:
            if args.video:
                raise
            print("  aviso: %s -- caindo pro gerador sintetico" % erro, flush=True)
            return FonteSintetica()

    if not args.sintetico and not args.video:
        print("abrindo a webcam...", flush=True)
    Handler.difusor = Difusor(criar_fonte)
    if not Handler.difusor.espera_primeiro():
        print("nenhum quadro em %d segundos; abortando." % 120)
        print("  tente:  python mock_esp32_cam.py --sintetico")
        print("  (nao usa camera nenhuma e serve pra testar o app do mesmo jeito)")
        return 1

    servidor = ThreadingHTTPServer(("0.0.0.0", args.porta), Handler)
    print("Mock ESP32-CAM  --  fonte: %s" % Handler.difusor.descricao)
    print("  %dx%d a %d quadros/s, JPEG qualidade %d"
          % (LARGURA, ALTURA, FPS, QUALIDADE_JPEG))
    print("")
    print("  no proprio PC:  http://localhost:%d/" % args.porta)
    for ip in enderecos_locais():
        print("  no celular:     http://%s:%d/" % (ip, args.porta))
    print("")
    print("Ctrl+C pra parar.", flush=True)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nParando.")
    finally:
        Handler.difusor.para()
    return 0


if __name__ == "__main__":
    sys.exit(main())
