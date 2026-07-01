"""
VisuAll por partes
==================
Entrada principal do app.

Os detalhes foram separados em pequenos modulos:
  - m01_visuall_config.py
  - m02_visuall_modelos.py
  - m03_visuall_traducao.py
  - m04_visuall_alfabeto.py
  - m05_visuall_corpo.py
  - m06_visuall_rosto.py
  - m07_visuall_estado.py
  - m08_visuall_captura.py
  - m09_visuall_processamento.py
  - m10_visuall_servidor.py
"""
import asyncio
from threading import Thread

from m08_visuall_captura import capture_thread
from m09_visuall_processamento import process_thread
from m10_visuall_servidor import main


if __name__ == "__main__":
    print("=" * 50)
    print("  VisuAll por partes")
    print("=" * 50)
    Thread(target=capture_thread, daemon=True).start()
    Thread(target=process_thread, daemon=True).start()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Encerrado")
