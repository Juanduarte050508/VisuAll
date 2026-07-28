"""
Capturar — grava clipes de treino pra letras com movimento e gestos
corporais (e fotos pra letras paradas), direto da webcam.

Fluxo: escolhe categoria + rótulo, clica GRAVAR, espera a contagem de 3s,
grava ~3s sozinho e já salva no lugar certo. Ver treinamento/README.md pra
instruções completas e quantas amostras gravar de cada.

Não mexe em nada do app Android — só produz arquivos em VisuAll/data/, que
depois viram um .onnx/.tflite novo rodando treinamento/Treinar.bat.
"""
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "linear" / "backend" / "training"))
from training_common import BODY_LABELS, DYNAMIC_LABELS, STATIC_LABELS  # noqa: E402

CONTAGEM_S = 3.0
GRAVACAO_S = 3.0
FRAMES_ESTATICA = 8  # quantas fotos extrair do clipe de 3s pra letra parada
PREVIEW_LARGURA = 480

CATEGORIAS = {
    "Letra parada (estática)": ("estatica", sorted(STATIC_LABELS)),
    "Letra com movimento (dinâmica)": ("dinamica", DYNAMIC_LABELS),
    "Gesto corporal": ("corpo", BODY_LABELS),
}

PASTA_POR_CATEGORIA = {
    "estatica": ROOT / "data" / "raw_images",
    "dinamica": ROOT / "data" / "raw_videos",
    "corpo": ROOT / "data" / "raw_videos_corpo",
}

IDLE, CONTAGEM, GRAVANDO, SALVANDO = "idle", "contagem", "gravando", "salvando"


class CapturarApp:
    def __init__(self, root):
        self.root = root
        root.title("VisuAll — Capturar amostras de treino")
        root.geometry("640x680")
        root.resizable(False, False)

        self.estado = IDLE
        self.tempo_estado_inicio = 0.0
        self.frames_gravados = []
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Câmera não encontrada",
                "Não consegui abrir a webcam (índice 0).\n"
                "Feche outros programas que possam estar usando a câmera e reabra este.",
            )

        self._montar_ui()
        self._garantir_pastas()
        self._atualizar_contador()
        self.root.after(30, self._tick)
        self.root.protocol("WM_DELETE_WINDOW", self._fechar)

    # ── UI ───────────────────────────────────────────────────────────────
    def _montar_ui(self):
        pad = {"padx": 10, "pady": 6}

        frame_topo = ttk.Frame(self.root)
        frame_topo.pack(fill="x", **pad)

        ttk.Label(frame_topo, text="Categoria:").grid(row=0, column=0, sticky="w")
        self.var_categoria = tk.StringVar(value=list(CATEGORIAS.keys())[1])
        combo_categoria = ttk.Combobox(
            frame_topo, textvariable=self.var_categoria,
            values=list(CATEGORIAS.keys()), state="readonly", width=32,
        )
        combo_categoria.grid(row=0, column=1, sticky="w", padx=6)
        combo_categoria.bind("<<ComboboxSelected>>", lambda e: self._categoria_mudou())

        ttk.Label(frame_topo, text="Rótulo:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.var_rotulo = tk.StringVar()
        self.combo_rotulo = ttk.Combobox(
            frame_topo, textvariable=self.var_rotulo, state="readonly", width=32,
        )
        self.combo_rotulo.grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))
        self.combo_rotulo.bind("<<ComboboxSelected>>", lambda e: self._atualizar_contador())

        self.label_preview = ttk.Label(self.root)
        self.label_preview.pack(pady=8)

        self.label_status = ttk.Label(
            self.root, text="Pronto", font=("Segoe UI", 20, "bold"), anchor="center"
        )
        self.label_status.pack(fill="x", pady=4)

        self.label_contador = ttk.Label(
            self.root, text="", font=("Segoe UI", 11), anchor="center"
        )
        self.label_contador.pack(fill="x")

        self.botao_gravar = ttk.Button(
            self.root, text="GRAVAR", command=self._iniciar_captura
        )
        self.botao_gravar.pack(pady=14, ipadx=20, ipady=10)

        self._categoria_mudou()

    def _categoria_mudou(self):
        _, rotulos = CATEGORIAS[self.var_categoria.get()]
        self.combo_rotulo["values"] = rotulos
        if rotulos:
            self.var_rotulo.set(rotulos[0])
        self._atualizar_contador()

    def _garantir_pastas(self):
        for categoria_key, pasta_base in PASTA_POR_CATEGORIA.items():
            rotulos = next(r for c, r in CATEGORIAS.values() if c == categoria_key)
            for rotulo in rotulos:
                (pasta_base / rotulo).mkdir(parents=True, exist_ok=True)

    def _pasta_atual(self):
        categoria_key, _ = CATEGORIAS[self.var_categoria.get()]
        rotulo = self.var_rotulo.get()
        return PASTA_POR_CATEGORIA[categoria_key] / rotulo, categoria_key

    def _atualizar_contador(self):
        if not self.var_rotulo.get():
            self.label_contador.config(text="")
            return
        pasta, categoria_key = self._pasta_atual()
        if categoria_key == "estatica":
            total = len(list(pasta.glob("*.jpg"))) if pasta.exists() else 0
            unidade = "fotos"
        else:
            total = len(list(pasta.glob("*.mp4"))) if pasta.exists() else 0
            unidade = "clipes"
        self.label_contador.config(
            text=f"{self.var_rotulo.get()}: {total} {unidade} já salvos"
        )

    # ── loop principal (preview + estados) ──────────────────────────────
    def _tick(self):
        if self.cap.isOpened():
            ok, frame = self.cap.read()
            if ok:
                self._processar_frame(frame)
        self.root.after(30, self._tick)

    def _processar_frame(self, frame):
        agora = time.monotonic()

        if self.estado == GRAVANDO:
            self.frames_gravados.append(frame.copy())
            decorrido = agora - self.tempo_estado_inicio
            if decorrido >= GRAVACAO_S:
                self._salvar_captura()
            else:
                self.label_status.config(text="GRAVANDO")
                self.label_contador.config(
                    text=f"{decorrido:.1f}s / {GRAVACAO_S:.0f}s"
                )
        elif self.estado == CONTAGEM:
            decorrido = agora - self.tempo_estado_inicio
            restante = CONTAGEM_S - decorrido
            if restante <= 0:
                self.estado = GRAVANDO
                self.tempo_estado_inicio = agora
                self.frames_gravados = []
            else:
                self.label_status.config(text=f"Prepare-se: {int(restante) + 1}")

        # preview espelhado (só na tela, o que é salvo é o frame cru).
        preview = cv2.flip(frame, 1)
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        altura = int(PREVIEW_LARGURA * preview_rgb.shape[0] / preview_rgb.shape[1])
        preview_rgb = cv2.resize(preview_rgb, (PREVIEW_LARGURA, altura))
        imagem = ImageTk.PhotoImage(Image.fromarray(preview_rgb))
        self.label_preview.configure(image=imagem)
        self.label_preview.image = imagem  # segura referência (senão o GC apaga)

    # ── captura ──────────────────────────────────────────────────────────
    def _iniciar_captura(self):
        if self.estado != IDLE:
            return
        if not self.cap.isOpened():
            messagebox.showerror("Sem câmera", "A webcam não está disponível.")
            return
        if not self.var_rotulo.get():
            messagebox.showwarning("Escolha um rótulo", "Selecione o que vai gravar.")
            return
        self.estado = CONTAGEM
        self.tempo_estado_inicio = time.monotonic()
        self.botao_gravar.config(state="disabled")
        self.combo_rotulo.config(state="disabled")

    def _salvar_captura(self):
        self.estado = SALVANDO
        self.label_status.config(text="Salvando...")
        self.root.update_idletasks()

        pasta, categoria_key = self._pasta_atual()
        pasta.mkdir(parents=True, exist_ok=True)
        agora = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            if categoria_key == "estatica":
                self._salvar_fotos(pasta, agora)
            else:
                self._salvar_video(pasta, agora)
        except Exception as e:
            messagebox.showerror("Erro ao salvar", str(e))

        self.frames_gravados = []
        self.estado = IDLE
        self.label_status.config(text="Pronto! Pode gravar de novo.")
        self.botao_gravar.config(state="normal")
        self.combo_rotulo.config(state="readonly")
        self._atualizar_contador()

    def _salvar_fotos(self, pasta, prefixo):
        total = len(self.frames_gravados)
        if total == 0:
            return
        passo = max(1, total // FRAMES_ESTATICA)
        indices = list(range(0, total, passo))[:FRAMES_ESTATICA]
        for i, idx in enumerate(indices):
            caminho = pasta / f"{prefixo}_{i}.jpg"
            cv2.imwrite(str(caminho), self.frames_gravados[idx])

    def _salvar_video(self, pasta, prefixo):
        if not self.frames_gravados:
            return
        altura, largura = self.frames_gravados[0].shape[:2]
        fps_estimado = max(1.0, len(self.frames_gravados) / GRAVACAO_S)
        caminho = pasta / f"{prefixo}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(caminho), fourcc, fps_estimado, (largura, altura))
        for frame in self.frames_gravados:
            writer.write(frame)
        writer.release()

    def _fechar(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    CapturarApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
