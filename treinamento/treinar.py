"""
Treinar — roda a extração de landmarks e o treino dos modelos a partir do
que foi gravado com treinamento/Capturar.bat, e já deixa os arquivos novos
prontos em mobile/app/src/main/assets/ (é só recompilar o app depois).

Roda cada categoria (letra parada, letra com movimento, gesto corporal)
separadamente, e pula sozinho qualquer categoria sem amostras gravadas —
não precisa ter as três pra rodar.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXTRACAO_DIR = ROOT / "linear" / "backend" / "data_extraction"
TREINO_DIR = ROOT / "linear" / "backend" / "training"
ASSETS_DIR = ROOT / "mobile" / "app" / "src" / "main" / "assets"


def contar_arquivos(pasta_base, extensoes):
    if not pasta_base.exists():
        return 0
    total = 0
    for sub in pasta_base.iterdir():
        if not sub.is_dir():
            continue
        for ext in extensoes:
            total += len(list(sub.glob(f"*{ext}")))
    return total


def rodar_etapa(titulo, script_path):
    print(f"\n{'=' * 60}")
    print(f"  {titulo}")
    print(f"{'=' * 60}")
    resultado = subprocess.run([sys.executable, str(script_path)])
    if resultado.returncode != 0:
        print(f"\n⚠️  '{script_path.name}' terminou com erro (código {resultado.returncode}). "
              "Veja a mensagem acima antes de continuar.")
    return resultado.returncode == 0


def main():
    print("VisuAll — Treinar modelos a partir dos dados gravados")

    n_estatica = contar_arquivos(DATA_DIR / "raw_images", [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    n_dinamica = contar_arquivos(DATA_DIR / "raw_videos", [".mp4", ".mov"])
    n_corpo = contar_arquivos(DATA_DIR / "raw_videos_corpo", [".mp4", ".mov", ".avi"])

    print(f"\nAmostras encontradas em data/:")
    print(f"  Letras paradas (fotos):        {n_estatica}")
    print(f"  Letras com movimento (vídeos): {n_dinamica}")
    print(f"  Gestos corporais (vídeos):     {n_corpo}")

    if n_estatica == 0 and n_dinamica == 0 and n_corpo == 0:
        print("\n❌ Nenhuma amostra encontrada em nenhuma categoria.")
        print("   Grave alguns clipes com treinamento/Capturar.bat primeiro.")
        input("\nPressione ENTER pra fechar...")
        return

    atualizados = []

    if n_estatica > 0:
        ok1 = rodar_etapa("1/2 — Extraindo landmarks das FOTOS (letras paradas)",
                           EXTRACAO_DIR / "extract_from_images.py")
        if ok1:
            ok2 = rodar_etapa("2/2 — Treinando modelo de LETRAS PARADAS",
                               TREINO_DIR / "train_static_model.py")
            if ok2:
                atualizados.append("letras_estaticas/geral/model.onnx / labels.txt")
    else:
        print("\n(pulando letras paradas — nenhuma foto em data/raw_images/)")

    if n_dinamica > 0:
        ok1 = rodar_etapa("1/2 — Extraindo landmarks dos VÍDEOS (letras com movimento)",
                           EXTRACAO_DIR / "extract_from_videos.py")
        if ok1:
            ok2 = rodar_etapa("2/2 — Treinando modelo de LETRAS COM MOVIMENTO",
                               TREINO_DIR / "train_dynamic_model.py")
            if ok2:
                atualizados.append("letras_dinamicas/geral/model.onnx / labels.txt")
    else:
        print("\n(pulando letras com movimento — nenhum vídeo em data/raw_videos/)")

    if n_corpo > 0:
        ok1 = rodar_etapa("1/2 — Extraindo landmarks dos VÍDEOS (gestos corporais)",
                           EXTRACAO_DIR / "extract_from_videos_corpo.py")
        if ok1:
            ok2 = rodar_etapa("2/2 — Treinando modelo de GESTOS CORPORAIS",
                               TREINO_DIR / "train_body_model.py")
            if ok2:
                atualizados.append("gestos/geral/model.tflite / labels.txt")
    else:
        print("\n(pulando gestos corporais — nenhum vídeo em data/raw_videos_corpo/)")

    print(f"\n{'=' * 60}")
    if atualizados:
        print("✅ Terminado. Arquivos atualizados em mobile/app/src/main/assets/:")
        for nome in atualizados:
            print(f"   - {nome}")
        print("\nPróximo passo: recompilar o app Android (assembleDebug) pra usar os modelos novos.")
    else:
        print("❌ Nenhum modelo foi treinado com sucesso. Veja os erros acima.")
    print(f"{'=' * 60}")

    input("\nPressione ENTER pra fechar...")


if __name__ == "__main__":
    main()
