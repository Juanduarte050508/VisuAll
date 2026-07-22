from threading import Lock

raw_frame_lock = Lock()
raw_frame = {"img": None, "ts": 0}

data_lock = Lock()
camera_data = {
    "status": "Inicializando...",
    "hands_detected": 0,
    "frame": "",
    "timestamp": 0,
    "letra_atual": "-",
    "frase": "",
    "confianca": 0.0,
    "tokens": [],
    "palavra_atual": "",
    "historico": [],
    "gesto_limpar_progress": 0.0,
    "modo_app": "alfabeto",
    "marcador": False,
    "sobr_val": 0.0,
    "desenho_ativo": True,
}
