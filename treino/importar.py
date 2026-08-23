"""
Importa videos/fotos gravados FORA do PC (celular, outra camera) e organiza
nas pastas que o pipeline deste repo le.

Destino (o mesmo que treino/gravar.py usa):
  letras dinamicas H J K X Z -> data/raw_videos/<LETRA>/*.mp4
  letras estaticas A B C ... -> data/raw_images/<LETRA>/*.jpg
                                (fotos; videos de letra estatica sao
                                 fatiados em quadros aqui mesmo, porque
                                 extract_from_images.py so le imagem)

Uso:

  # 1) pasta ja organizada em subpastas por letra
  python importar.py "D:/DCIM/VisuAll"

  # 2) celular Android conectado por cabo (precisa do adb no PATH)
  python importar.py --adb

  # 3) todos os arquivos de uma pasta sao da mesma letra
  python importar.py "D:/DCIM/Camera" --rotulo H

O que a origem precisa ter, no caso 1:

  VisuAll/
    H/  video1.mp4  video2.mp4
    K/  ...
    A/  foto1.jpg   (ou um video, que vira varias fotos)

Os arquivos sao COPIADOS (a origem nao e apagada) e renomeados com data/hora,
entao rodar duas vezes na mesma pasta duplica -- limpe a origem entre
importacoes, ou use --mover.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"

LETRAS_ESTATICAS = set("ABCDEFGILMNOPQRSTUVWY")
LETRAS_DINAMICAS = {"H", "J", "K", "X", "Z"}

# extract_from_videos.py so faz glob de *.mp4 e *.mov -- qualquer outro
# container e convertido/refeito aqui pra nao ser ignorado em silencio.
EXT_VIDEO_OK = {".mp4", ".mov"}
EXT_VIDEO_OUTROS = {".avi", ".mkv", ".webm", ".m4v", ".3gp"}
EXT_FOTO = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

STRIDE_FOTO = 5
JANELA_MINIMA = 10       # extract_from_videos.py usa JANELA = 10
PASTA_ADB = "/sdcard/DCIM/VisuAll"


def carimbo():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def conta_frames(caminho):
    cap = cv2.VideoCapture(str(caminho))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:                       # alguns .mov nao expoem FRAME_COUNT
        n = 0
        while cap.grab():
            n += 1
    cap.release()
    return n


def reescreve_video(origem, destino):
    """Copia quadro a quadro pra um .mp4 que o OpenCV do extractor abre."""
    cap = cv2.VideoCapture(str(origem))
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return False
    altura, largura = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (largura, altura))
    while ok:
        writer.write(frame)
        ok, frame = cap.read()
    writer.release()
    cap.release()
    return True


def video_para_fotos(origem, pasta_destino, prefixo):
    pasta_destino.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(origem))
    i = salvas = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % STRIDE_FOTO == 0:
            cv2.imwrite(str(pasta_destino / ("%s_%02d.jpg" % (prefixo, salvas))), frame)
            salvas += 1
        i += 1
    cap.release()
    return salvas


def importa_arquivo(caminho, rotulo, mover, relatorio):
    ext = caminho.suffix.lower()
    prefixo = carimbo()

    if rotulo in LETRAS_DINAMICAS:
        if ext in EXT_FOTO:
            relatorio["ignorados"].append(
                "%s: foto em letra dinamica (%s precisa de video)" % (caminho.name, rotulo))
            return
        destino_dir = DATA / "raw_videos" / rotulo
        destino_dir.mkdir(parents=True, exist_ok=True)
        destino = destino_dir / (prefixo + ".mp4")

        if ext in EXT_VIDEO_OK:
            if mover:
                shutil.move(str(caminho), str(destino))
            else:
                shutil.copy2(str(caminho), str(destino))
        elif ext in EXT_VIDEO_OUTROS:
            if not reescreve_video(caminho, destino):
                relatorio["ignorados"].append("%s: nao consegui abrir" % caminho.name)
                return
        else:
            relatorio["ignorados"].append("%s: extensao %s nao suportada" % (caminho.name, ext))
            return

        n = conta_frames(destino)
        if n < JANELA_MINIMA:
            relatorio["curtos"].append("%s -> %s (%d quadros)" % (caminho.name, rotulo, n))
        relatorio["videos"] += 1

    elif rotulo in LETRAS_ESTATICAS:
        destino_dir = DATA / "raw_images" / rotulo
        destino_dir.mkdir(parents=True, exist_ok=True)
        if ext in EXT_FOTO:
            destino = destino_dir / (prefixo + caminho.suffix.lower())
            if mover:
                shutil.move(str(caminho), str(destino))
            else:
                shutil.copy2(str(caminho), str(destino))
            relatorio["fotos"] += 1
        elif ext in EXT_VIDEO_OK or ext in EXT_VIDEO_OUTROS:
            n = video_para_fotos(caminho, destino_dir, prefixo)
            relatorio["fotos"] += n
            if mover:
                caminho.unlink()
        else:
            relatorio["ignorados"].append("%s: extensao %s nao suportada" % (caminho.name, ext))
    else:
        relatorio["ignorados"].append("%s: rotulo '%s' nao existe nos modelos" % (caminho.name, rotulo))


def puxa_do_celular():
    destino = Path(tempfile.mkdtemp(prefix="visuall_adb_"))
    print("Puxando %s do celular via adb..." % PASTA_ADB)
    r = subprocess.run(["adb", "pull", PASTA_ADB, str(destino)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("ERRO no adb pull:\n" + (r.stderr or r.stdout))
        print("\nConfira: celular conectado, depuracao USB ligada, `adb devices` lista o aparelho,")
        print("e a pasta %s existe no celular." % PASTA_ADB)
        return None
    interna = destino / Path(PASTA_ADB).name
    return interna if interna.exists() else destino


def main():
    ap = argparse.ArgumentParser(description="Importa midias externas pro dataset do VisuAll.")
    ap.add_argument("origem", nargs="?", help="pasta com subpastas por letra")
    ap.add_argument("--adb", action="store_true", help="puxa %s do celular" % PASTA_ADB)
    ap.add_argument("--rotulo", help="todos os arquivos da origem sao desta letra")
    ap.add_argument("--mover", action="store_true", help="move em vez de copiar")
    args = ap.parse_args()

    if args.adb:
        origem = puxa_do_celular()
        if origem is None:
            return 1
    elif args.origem:
        origem = Path(args.origem)
        if not origem.is_dir():
            print("ERRO: '%s' nao e uma pasta." % origem)
            return 1
    else:
        ap.print_help()
        return 1

    relatorio = {"videos": 0, "fotos": 0, "ignorados": [], "curtos": []}

    if args.rotulo:
        rotulo = args.rotulo.strip().upper()
        for arquivo in sorted(origem.iterdir()):
            if arquivo.is_file():
                importa_arquivo(arquivo, rotulo, args.mover, relatorio)
    else:
        subpastas = [p for p in sorted(origem.iterdir()) if p.is_dir()]
        if not subpastas:
            print("ERRO: '%s' nao tem subpastas por letra." % origem)
            print("Use --rotulo H se todos os arquivos forem da mesma letra.")
            return 1
        for pasta in subpastas:
            rotulo = pasta.name.strip().upper()
            for arquivo in sorted(pasta.iterdir()):
                if arquivo.is_file():
                    importa_arquivo(arquivo, rotulo, args.mover, relatorio)

    print("\n--- resumo ---")
    print("videos importados: %d" % relatorio["videos"])
    print("fotos importadas : %d" % relatorio["fotos"])
    if relatorio["curtos"]:
        print("\nAVISO: clipes curtos demais (extract_from_videos.py usa janela de %d"
              " quadros e nao gera NENHUMA amostra abaixo disso):" % JANELA_MINIMA)
        for linha in relatorio["curtos"]:
            print("  " + linha)
    if relatorio["ignorados"]:
        print("\nIgnorados:")
        for linha in relatorio["ignorados"]:
            print("  " + linha)
    print("\nDestino: %s" % DATA)
    print("Proximo passo: treino/Treinar.bat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
