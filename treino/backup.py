"""
Guarda uma copia dos modelos do app antes de sobrescreve-los.

Por que isto existe: treinar o modelo GERAL (tanto o de gestos corporais
quanto o de letras) nao acrescenta nada -- ele SUBSTITUI o modelo por um novo,
que sabe apenas o que estava nos seus videos. Se o modelo antigo foi treinado
com mais dados (ou com mais pessoas) que os seus, o novo pode sair PIOR, e sem
copia nao ha como voltar atras.

Cada treino guarda em treino/modelos_anteriores/<nome>_<data>_<hora>/.
"""
import shutil
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PASTA_BACKUP = RAIZ / "modelos_anteriores"
MANTER = 10          # backups mais antigos que isso sao descartados


def faz_backup(arquivos, nome):
    """Copia os arquivos que existirem. Devolve a pasta criada, ou None."""
    # resolve() antes de qualquer coisa: sem isso, um caminho relativo quebra
    # o relative_to(RAIZ) mais abaixo.
    arquivos = [Path(a).resolve() for a in arquivos]
    existentes = [a for a in arquivos if a.exists()]
    if not existentes:
        return None

    destino = PASTA_BACKUP / ("%s_%s" % (nome, datetime.now().strftime("%Y%m%d_%H%M%S")))
    destino.mkdir(parents=True, exist_ok=True)
    for arquivo in existentes:
        shutil.copy2(str(arquivo), str(destino / arquivo.name))

    # Guarda de onde cada arquivo veio, pra restauracao nao depender de
    # adivinhar o caminho.
    (destino / "origem.txt").write_text(
        "\n".join(str(Path(a).relative_to(RAIZ)) for a in arquivos) + "\n",
        encoding="utf-8")

    _limpa_antigos(nome)
    return destino


def _limpa_antigos(nome):
    if not PASTA_BACKUP.exists():
        return
    backups = sorted([p for p in PASTA_BACKUP.iterdir()
                      if p.is_dir() and p.name.startswith(nome + "_")])
    for velho in backups[:-MANTER]:
        shutil.rmtree(velho, ignore_errors=True)


def lista_backups():
    """Backups existentes, do mais novo pro mais antigo."""
    if not PASTA_BACKUP.exists():
        return []
    return sorted([p for p in PASTA_BACKUP.iterdir() if p.is_dir()],
                  key=lambda p: p.name, reverse=True)


def restaura(pasta):
    """Devolve os arquivos de um backup pros lugares de onde vieram."""
    pasta = Path(pasta)
    origem_txt = pasta / "origem.txt"
    if not origem_txt.exists():
        return None

    destinos = [RAIZ / linha.strip()
                for linha in origem_txt.read_text(encoding="utf-8").splitlines()
                if linha.strip()]
    restaurados = []
    for destino in destinos:
        guardado = pasta / destino.name
        if guardado.exists():
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(guardado), str(destino))
            restaurados.append(destino)
    return restaurados
