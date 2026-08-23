"""
Devolve um modelo guardado antes de um treino (desfaz o treino).

Cada treino que SUBSTITUI um modelo geral guarda antes uma copia em
treino/modelos_anteriores/. Este script lista essas copias e devolve a que
voce escolher pro lugar de onde veio.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backup import RAIZ, lista_backups, restaura  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROTULOS = {"gestos": "gestos corporais", "letras_estaticas": "letras paradas",
           "letras_dinamicas": "letras com movimento"}


def main():
    backups = lista_backups()
    if not backups:
        print("Nao ha nenhum modelo guardado.")
        print()
        print("As copias sao criadas automaticamente quando um treino")
        print("SUBSTITUI um modelo geral (TreinarCorpo.bat, ou Treinar.bat sem")
        print("--reforcar). Se voce so usou Reforcar.bat, nada foi substituido")
        print("-- ele acrescenta modelos por letra sem apagar nada.")
        return 0

    print()
    print("  Modelos guardados (o mais recente primeiro):")
    print()
    for i, pasta in enumerate(backups, 1):
        partes = pasta.name.rsplit("_", 2)
        tipo = partes[0]
        data, hora = (partes[1], partes[2]) if len(partes) == 3 else ("?", "?")
        quando = "%s/%s/%s as %s:%s" % (data[6:8], data[4:6], data[0:4],
                                        hora[0:2], hora[2:4]) if data != "?" else "?"
        arquivos = [a.name for a in pasta.iterdir() if a.name != "origem.txt"]
        print("   %d) %-18s  %s" % (i, ROTULOS.get(tipo, tipo), quando))
        print("      %s" % ", ".join(sorted(arquivos)))
    print()

    escolha = input("  Qual devolver? (numero, ou ENTER pra cancelar): ").strip()
    if not escolha:
        print("\n  Cancelado -- nada foi alterado.")
        return 0
    if not escolha.isdigit() or not (1 <= int(escolha) <= len(backups)):
        print("\n  '%s' nao e uma das opcoes. Nada foi alterado." % escolha)
        return 1

    pasta = backups[int(escolha) - 1]
    devolvidos = restaura(pasta)
    if not devolvidos:
        print("\n  ERRO: esse backup esta incompleto (sem origem.txt).")
        return 1

    print()
    for caminho in devolvidos:
        print("  devolvido: %s" % caminho.relative_to(RAIZ))
    print()
    print("  Pronto. Recompile o app (Android Studio -> Run) pra voltar")
    print("  a usar este modelo no celular.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
