"""
Gravador assistido de amostras de treino do VisuAll.

Aparece o rotulo na tela, voce aperta ESPACO, tem 3 segundos de contagem pra
se posicionar, grava sozinho e salva. FICA no mesmo rotulo: o normal e gravar
varias amostras seguidas da mesma letra/gesto. Trocar e com N/P.

Errou a gravacao? R apaga a ultima e voce grava de novo.

Salva exatamente onde o pipeline atual deste repo le:

  - letra DINAMICA (H J K X Z) -> data/raw_videos/<LETRA>/AAAAMMDD_HHMMSS.mp4
    (lido por linear/backend/data_extraction/extract_from_videos.py)

  - letra ESTATICA (A B C ...) -> data/raw_images/<LETRA>/AAAAMMDD_HHMMSS_NN.jpg
    (lido por linear/backend/data_extraction/extract_from_images.py, que
    trabalha com FOTOS e nao com video -- por isso aqui o clipe de 3s vira
    varios .jpg, um a cada STRIDE_FOTO quadros)

  - NADA (nao e sinal nenhum)  -> data/raw_negativos/NADA/AAAAMMDD_HHMMSS.mp4
    (lido por treino/extrair_negativos.py; sao os exemplos de "isto NAO e
    letra" que os modelos individuais precisam pra parar de reconhecer letra
    onde nao tem)

Nao depende de nada alem do que ja esta em requirements.txt (opencv-python).

Teclas:
  ESPACO    grava o rotulo atual (fica nele; grave quantas quiser)
  R         apaga a ULTIMA gravacao (pra refazer na hora)
  N / P     proximo / anterior rotulo
  TAB       cicla: estatica -> dinamica -> corpo -> nada
  Q ou ESC  sai
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"

# As mesmas classes que estao dentro de models/static_classes.pkl e
# models/dynamic_classes.pkl -- gravar um rotulo fora dessas listas gera
# dados que o modelo atual nao sabe usar.
LETRAS_ESTATICAS = list("ABCDEFGILMNOPQRSTUVWY")
LETRAS_DINAMICAS = ["H", "J", "K", "X", "Z"]

# Os gestos corporais saem do labels.txt que o proprio app carrega, e nao de
# uma copia aqui: se alguem adicionar/remover um gesto la, esta ferramenta
# acompanha sozinha. NEUTRO faz parte da lista de proposito -- e o "parado,
# sem sinal" do modo corpo.
_LABELS_CORPO = (RAIZ / "mobile" / "app" / "src" / "main" / "assets"
                 / "gestos" / "geral" / "labels.txt")
try:
    GESTOS_CORPO = [l.strip().upper()
                    for l in _LABELS_CORPO.read_text(encoding="utf-8").splitlines()
                    if l.strip()]
except OSError:
    GESTOS_CORPO = ["AJUDAR", "COMPUTADOR", "CONVERSAR", "NEUTRO", "PESSOA", "SURDO"]

CONTAGEM_S = 3.0
GRAVACAO_S = 3.0
STRIDE_FOTO = 5          # 1 foto a cada 5 quadros, no modo estatico
CAMERA_INDEX = 0

# (rotulos, pasta de destino, o que salvar)
# "nada" nao e uma letra: sao exemplos do que NAO e sinal nenhum (mao a toa,
# cocando a cabeca, gesticulando). Servem so como exemplo negativo no treino
# dos modelos individuais -- sem eles, um modelo "isto e um E?" nunca aprendeu
# como e uma mao que nao esta sinalizando, e responde "sim" pra qualquer coisa.
MODOS = {
    "estatica": (LETRAS_ESTATICAS, DATA / "raw_images", "fotos"),
    "dinamica": (LETRAS_DINAMICAS, DATA / "raw_videos", "video"),
    "corpo": (GESTOS_CORPO, DATA / "raw_body_videos", "video"),
    "nada": (["NADA"], DATA / "raw_negativos", "video"),
}
ORDEM_MODOS = ["estatica", "dinamica", "corpo", "nada"]

# Quanto tempo grava em cada modo. Gesto corporal precisa de mais folga que
# uma letra: o movimento e maior e comeca/termina mais devagar.
DURACAO_POR_MODO = {"corpo": 4.0}

IDLE, CONTAGEM, GRAVANDO = "idle", "contagem", "gravando"

BARRA_BAIXO = 76         # altura da faixa escura de baixo (2 linhas de texto)

VERDE = (80, 220, 100)
AMARELO = (60, 200, 250)
VERMELHO = (60, 60, 240)
BRANCO = (245, 245, 245)
CINZA = (170, 170, 170)


def conta_amostras(pasta, formato):
    """Quantos clipes/fotos aquele rotulo ja tem."""
    if not pasta.exists():
        return 0
    return len(list(pasta.glob("*.mp4" if formato == "video" else "*.jpg")))


def salva_video(frames, pasta, prefixo, duracao):
    pasta.mkdir(parents=True, exist_ok=True)
    altura, largura = frames[0].shape[:2]
    fps = max(1.0, len(frames) / duracao)
    destino = pasta / (prefixo + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(destino), fourcc, fps, (largura, altura))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return destino


def salva_fotos(frames, pasta, prefixo):
    """Devolve a lista de arquivos criados -- e o que permite desfazer (R)."""
    pasta.mkdir(parents=True, exist_ok=True)
    criados = []
    for i in range(0, len(frames), STRIDE_FOTO):
        destino = pasta / ("%s_%02d.jpg" % (prefixo, len(criados)))
        cv2.imwrite(str(destino), frames[i])
        criados.append(destino)
    return criados


def apaga(caminhos):
    """Desfaz a ultima gravacao (tecla R)."""
    apagados = 0
    for caminho in caminhos:
        try:
            caminho.unlink()
            apagados += 1
        except OSError:
            pass
    return apagados


def texto(img, msg, xy, escala=0.7, cor=BRANCO, grossura=2):
    cv2.putText(img, msg, xy, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0),
                grossura + 3, cv2.LINE_AA)
    cv2.putText(img, msg, xy, cv2.FONT_HERSHEY_SIMPLEX, escala, cor,
                grossura, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERRO: nao consegui abrir a webcam (indice %d)." % CAMERA_INDEX)
        print("Feche Zoom/Teams/OBS e tente de novo.")
        return 1

    for rotulos_m, pasta_m, _ in MODOS.values():
        for r in rotulos_m:
            (pasta_m / r).mkdir(parents=True, exist_ok=True)

    modo = "estatica"
    idx = 0
    estado = IDLE
    marca = 0.0
    buffer = []
    ultimo_aviso = ""
    ultimo_salvo = []

    janela = "VisuAll - Gravar amostras de treino"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela, 900, 640)

    print("Gravando para: %s" % DATA)
    print("ESPACO grava | N/P troca rotulo | TAB troca modo | Q sai\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERRO: perdi o sinal da webcam.")
            break

        rotulos, pasta_base, formato = MODOS[modo]
        rotulo = rotulos[idx % len(rotulos)]
        pasta = pasta_base / rotulo
        duracao = DURACAO_POR_MODO.get(modo, GRAVACAO_S)
        agora = time.monotonic()

        # --- maquina de estados ------------------------------------------
        if estado == CONTAGEM:
            if CONTAGEM_S - (agora - marca) <= 0:
                estado = GRAVANDO
                marca = agora
                buffer = []
        elif estado == GRAVANDO:
            buffer.append(frame.copy())
            if agora - marca >= duracao:
                prefixo = datetime.now().strftime("%Y%m%d_%H%M%S")
                if formato == "video":
                    destino = salva_video(buffer, pasta, prefixo, duracao)
                    ultimo_salvo = [destino]
                    ultimo_aviso = "salvo: %s (%d quadros)" % (destino.name, len(buffer))
                else:
                    ultimo_salvo = salva_fotos(buffer, pasta, prefixo)
                    ultimo_aviso = "salvas %d fotos de %s" % (len(ultimo_salvo), rotulo)
                print("  [%s] %s" % (rotulo, ultimo_aviso))
                buffer = []
                estado = IDLE
                # FICA na mesma letra de proposito: o normal e gravar varias
                # seguidas do mesmo rotulo. Trocar e com N/P, quando voce quiser.

        # --- desenho (preview espelhado; o que e salvo e o quadro cru) -----
        vista = cv2.flip(frame, 1)
        h, w = vista.shape[:2]
        cv2.rectangle(vista, (0, 0), (w, 96), (28, 28, 28), -1)
        # Barra de baixo com DUAS linhas: aviso em cima, comandos embaixo.
        # Antes as duas dividiam a mesma linha e o aviso cobria os comandos.
        cv2.rectangle(vista, (0, h - BARRA_BAIXO), (w, h), (28, 28, 28), -1)

        texto(vista, rotulo, (24, 74), escala=2.0, cor=VERDE, grossura=4)
        texto(vista, "modo: " + modo, (int(w * 0.42), 36), escala=0.6,
              cor=CINZA, grossura=1)
        texto(vista, "%d ja gravados" % conta_amostras(pasta, formato),
              (int(w * 0.42), 66), escala=0.6, cor=CINZA, grossura=1)

        if estado == CONTAGEM:
            restante = CONTAGEM_S - (agora - marca)
            texto(vista, str(int(restante) + 1), (w // 2 - 30, h // 2),
                  escala=4.0, cor=AMARELO, grossura=8)
            texto(vista, "prepare-se", (w // 2 - 90, h // 2 + 60),
                  escala=0.9, cor=AMARELO, grossura=2)
        elif estado == GRAVANDO:
            decorrido = agora - marca
            cv2.circle(vista, (w - 44, 48), 16, VERMELHO, -1)
            texto(vista, "REC %.1fs / %.0fs" % (decorrido, duracao),
                  (w - 250, 56), escala=0.7, cor=VERMELHO, grossura=2)
            cv2.rectangle(vista, (0, h - BARRA_BAIXO - 5),
                          (int(w * decorrido / duracao), h - BARRA_BAIXO),
                          VERMELHO, -1)

        # Linha de cima da barra: o que aconteceu por ultimo.
        if estado == IDLE and ultimo_aviso:
            texto(vista, ultimo_aviso, (18, h - 44), escala=0.55,
                  cor=VERDE, grossura=1)
        # Linha de baixo: os comandos, sempre visiveis e sozinhos na linha.
        texto(vista, "ESPACO grava | R apaga | N/P troca | TAB modo | Q sai",
              (18, h - 15), escala=0.5, cor=CINZA, grossura=1)

        cv2.imshow(janela, vista)

        # --- teclado -----------------------------------------------------
        tecla = cv2.waitKey(1) & 0xFF
        # Com CapsLock ligado (ou Shift) o OpenCV manda a MAIUSCULA: 'N' e 78,
        # nao 110. Sem normalizar, as teclas de letra simplesmente nao
        # respondiam -- so ESPACO e TAB, que nao tem variante maiuscula.
        if 65 <= tecla <= 90:
            tecla += 32

        if tecla in (ord("q"), 27):
            break
        if estado == IDLE:
            if tecla == 32:                 # ESPACO
                estado = CONTAGEM
                marca = agora
                ultimo_aviso = ""
            elif tecla == ord("r"):
                if ultimo_salvo:
                    n = apaga(ultimo_salvo)
                    ultimo_salvo = []
                    ultimo_aviso = "APAGADO (%d arquivo(s)). Pode gravar de novo." % n
                    print("  [%s] %s" % (rotulo, ultimo_aviso))
                else:
                    ultimo_aviso = "nada pra apagar"
            elif tecla in (ord("n"), 83, 77):      # N ou seta pra direita
                idx += 1
                ultimo_salvo = []      # so da pra desfazer a gravacao atual
                ultimo_aviso = ""
            elif tecla in (ord("p"), 81, 75):      # P ou seta pra esquerda
                idx -= 1
                ultimo_salvo = []
                ultimo_aviso = ""
            elif tecla == 9:                # TAB cicla entre os 3 modos
                modo = ORDEM_MODOS[(ORDEM_MODOS.index(modo) + 1) % len(ORDEM_MODOS)]
                idx = 0
        if cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nPronto. Agora rode treino/Treinar.bat pra gerar os modelos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
