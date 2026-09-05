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


class FonteWebcam:
    """Webcam ou arquivo de video, sempre em loop."""

    def __init__(self, fonte):
        self.captura = cv2.VideoCapture(fonte)
        if not self.captura.isOpened():
            raise RuntimeError("nao consegui abrir %r" % (fonte,))
        self.captura.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA)
        self.captura.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA)
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
        # Fundo que muda de cor devagar, pra dar pra ver movimento.
        img[:] = (40, 30 + (self.n * 2) % 180, 60)
        # Barra que atravessa a tela: se ela para, o stream parou.
        x = int((self.n * 4) % LARGURA)
        cv2.rectangle(img, (x, 0), (min(x + 20, LARGURA), ALTURA), (255, 255, 255), -1)
        cv2.putText(img, "MOCK ESP32-CAM", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "quadro %d" % self.n, (8, ALTURA - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img


def abre_fonte(args):
    if args.sintetico:
        return FonteSintetica()
    alvo = args.video if args.video else 0
    try:
        return FonteWebcam(alvo)
    except RuntimeError as erro:
        if args.video:
            raise
        print("  aviso: %s -- caindo pro gerador sintetico" % erro)
        return FonteSintetica()


class Handler(BaseHTTPRequestHandler):
    fonte = None       # preenchido no main

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
        try:
            while True:
                inicio = time.time()
                imagem = self.fonte.quadro()
                if imagem is None:
                    break
                ok, buf = cv2.imencode(
                    ".jpg", imagem, [int(cv2.IMWRITE_JPEG_QUALITY), QUALIDADE_JPEG])
                if not ok:
                    continue
                jpeg = buf.tobytes()

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

    Handler.fonte = abre_fonte(args)
    servidor = ThreadingHTTPServer(("0.0.0.0", args.porta), Handler)

    print("Mock ESP32-CAM  --  fonte: %s" % Handler.fonte.descricao)
    print("  %dx%d a %d quadros/s, JPEG qualidade %d" %
          (LARGURA, ALTURA, FPS, QUALIDADE_JPEG))
    print("")
    print("  no proprio PC:  http://localhost:%d/" % args.porta)
    for ip in enderecos_locais():
        print("  no celular:     http://%s:%d/" % (ip, args.porta))
    print("")
    print("Ctrl+C pra parar.")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nParando.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
