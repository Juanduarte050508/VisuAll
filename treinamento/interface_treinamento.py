from __future__ import annotations

import queue
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import treinar_visuall as core


class QueueWriter:
    def __init__(self, log_queue: queue.Queue[str]) -> None:
        self.log_queue = log_queue
        self.buffer = ""

    def write(self, text: str) -> int:
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.log_queue.put(line)
        return len(text)

    def flush(self) -> None:
        if self.buffer.strip():
            self.log_queue.put(self.buffer.rstrip())
        self.buffer = ""


class TrainingApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("VisuAll - Treinamento")
        self.geometry("860x620")
        self.minsize(760, 520)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None

        self.labels = ["AUTO"] + core.STATIC_LABELS + core.DYNAMIC_LABELS + core.BODY_LABELS
        self.label_var = tk.StringVar(value="AUTO")
        self.path_var = tk.StringVar()
        self.types_var = tk.StringVar(value="todos")
        self.status_var = tk.StringVar(value="Pronto")
        self.buttons: list[ttk.Button] = []

        self._build()
        self.after(120, self._drain_log)

    def _build(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)

        title = ttk.Label(root, text="Treinamento VisuAll", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w")
        ttk.Label(root, textvariable=self.status_var, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 0))

        form = ttk.Frame(root)
        form.pack(fill="x", pady=(16, 8))

        ttk.Label(form, text="Letra ou gesto").grid(row=0, column=0, sticky="w")
        label_combo = ttk.Combobox(
            form,
            textvariable=self.label_var,
            values=self.labels,
            state="readonly",
            width=22,
        )
        label_combo.grid(row=1, column=0, sticky="ew", padx=(0, 12), pady=(4, 0))

        ttk.Label(form, text="Tipo de treino").grid(row=0, column=1, sticky="w")
        type_combo = ttk.Combobox(
            form,
            textvariable=self.types_var,
            values=["todos", "static", "dynamic", "body", "static,dynamic"],
            state="readonly",
            width=18,
        )
        type_combo.grid(row=1, column=1, sticky="ew", padx=(0, 12), pady=(4, 0))

        ttk.Label(form, text="Pasta ou arquivo").grid(row=0, column=2, sticky="w")
        path_entry = ttk.Entry(form, textvariable=self.path_var)
        path_entry.grid(row=1, column=2, sticky="ew", pady=(4, 0))

        form.columnconfigure(2, weight=1)

        pickers = ttk.Frame(root)
        pickers.pack(fill="x", pady=(0, 12))

        self.add_button(pickers, "Selecionar pasta", self.choose_folder).pack(side="left")
        self.add_button(pickers, "Selecionar arquivo", self.choose_file).pack(side="left", padx=(8, 0))
        self.add_button(pickers, "Status", self.run_status).pack(side="right")

        actions = ttk.Frame(root)
        actions.pack(fill="x", pady=(0, 12))

        self.add_button(actions, "Importar midias", self.run_import).pack(side="left")
        self.add_button(actions, "Extrair landmarks", self.run_extract).pack(side="left", padx=(8, 0))
        self.add_button(actions, "Treinar modelo", self.run_train).pack(side="left", padx=(8, 0))
        self.add_button(actions, "Importar + Extrair + Treinar", self.run_all).pack(side="left", padx=(8, 0))
        self.add_button(actions, "Limpar log", self.clear_log).pack(side="right")

        hint = (
            "Use AUTO quando a pasta ja tiver subpastas com nomes dos labels, como A, H ou AJUDAR. "
            "Escolha um label quando quiser importar uma pasta/arquivo inteiro para uma letra ou gesto especifico."
        )
        ttk.Label(root, text=hint, wraplength=800).pack(anchor="w", pady=(0, 8))

        log_frame = ttk.LabelFrame(root, text="Log")
        log_frame.pack(fill="both", expand=True)

        self.log = tk.Text(log_frame, wrap="word", height=18)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def add_button(self, parent, text: str, command) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command)
        self.buttons.append(button)
        return button

    def set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in self.buttons:
            if button.cget("text") != "Limpar log":
                button.configure(state=state)

    def choose_folder(self) -> None:
        path = filedialog.askdirectory(title="Selecione a pasta com fotos/videos")
        if path:
            self.path_var.set(path)

    def choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Selecione uma foto ou video",
            filetypes=[
                ("Midias", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.mov *.avi *.mkv *.webm *.m4v"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if path:
            self.path_var.set(path)

    def selected_label(self) -> str | None:
        label = self.label_var.get().strip()
        return None if label == "AUTO" else label

    def selected_types(self) -> set[str]:
        return core.parse_kinds(self.types_var.get().strip())

    def clear_log(self) -> None:
        self.log.delete("1.0", "end")

    def append_log(self, text: str) -> None:
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _drain_log(self) -> None:
        while True:
            try:
                text = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if text.startswith("__STATUS__:"):
                status = text.removeprefix("__STATUS__:")
                self.status_var.set(status)
                if status in {"FINALIZADO", "ERRO"}:
                    self.set_buttons_enabled(True)
            else:
                self.append_log(text)
        self.after(120, self._drain_log)

    def run_background(self, title: str, fn) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Aguarde", "Ja existe uma tarefa rodando.")
            return

        def task() -> None:
            self.log_queue.put(f"\n== {title} ==")
            self.log_queue.put("__STATUS__:RODANDO...")
            stdout, stderr = sys.stdout, sys.stderr
            writer = QueueWriter(self.log_queue)
            try:
                sys.stdout = writer
                sys.stderr = writer
                fn()
                writer.flush()
                self.log_queue.put("Concluido.")
                self.log_queue.put("__STATUS__:FINALIZADO")
            except BaseException as exc:
                writer.flush()
                if isinstance(exc, SystemExit):
                    self.log_queue.put(str(exc) or "Tarefa cancelada.")
                else:
                    self.log_queue.put(traceback.format_exc())
                self.log_queue.put("__STATUS__:ERRO")
            finally:
                sys.stdout = stdout
                sys.stderr = stderr

        self.set_buttons_enabled(False)
        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def require_path(self) -> Path | None:
        raw = self.path_var.get().strip()
        if not raw:
            messagebox.showwarning("Selecione entrada", "Escolha uma pasta ou arquivo primeiro.")
            return None
        path = Path(raw)
        if not path.exists():
            messagebox.showerror("Entrada invalida", f"Nao encontrei: {path}")
            return None
        return path

    def run_status(self) -> None:
        self.run_background("Status", core.print_status)

    def run_import(self) -> None:
        path = self.require_path()
        if path is None:
            return

        def work() -> None:
            stats = core.import_media(path, self.selected_label())
            self.log_queue.put(f"Importacao: {stats.copied} copiados, {stats.skipped} ignorados.")

        self.run_background("Importar midias", work)

    def run_extract(self) -> None:
        kinds = self.selected_types()
        self.run_background("Extrair landmarks", lambda: core.run_extract(kinds))

    def run_train(self) -> None:
        kinds = self.selected_types()
        self.run_background("Treinar modelo", lambda: core.run_train(kinds))

    def run_all(self) -> None:
        path = self.require_path()
        if path is None:
            return
        kinds = self.selected_types()

        def work() -> None:
            stats = core.import_media(path, self.selected_label())
            self.log_queue.put(f"Importacao: {stats.copied} copiados, {stats.skipped} ignorados.")
            core.run_extract(kinds)
            core.run_train(kinds)

        self.run_background("Importar + Extrair + Treinar", work)


if __name__ == "__main__":
    TrainingApp().mainloop()
