import asyncio
import json

import websockets

from m07_visuall_estado import camera_data, data_lock
from m03_visuall_traducao import montar_exibicao


async def send_data(websocket):
    print("🔌 Cliente conectado!")

    async def receive():
        async for message in websocket:
            try:
                cmd = json.loads(message)
                act = cmd.get("action")
                with data_lock:
                    if act == "modo":
                        novo = cmd.get("modo", "alfabeto")
                        if novo in ("alfabeto", "corpo"):
                            if camera_data["palavra_atual"].strip():
                                camera_data["tokens"].append(camera_data["palavra_atual"].strip())
                            camera_data["palavra_atual"] = ""
                            camera_data["modo_app"] = novo
                            camera_data["letra_atual"] = "-"
                            print(f"➡ Modo: {novo}")
                    elif act == "desenho":
                        camera_data["desenho_ativo"] = bool(cmd.get("ativo", True))
                        estado = "ON" if camera_data["desenho_ativo"] else "OFF"
                        print(f"➡ Linhas de detecção: {estado}")
                    elif act == "limpar":
                        render = montar_exibicao(
                            camera_data["tokens"], camera_data["palavra_atual"], False
                        )
                        if render.strip():
                            camera_data["historico"].insert(0, render)
                            camera_data["historico"] = camera_data["historico"][:15]
                        camera_data["tokens"] = []
                        camera_data["palavra_atual"] = ""
                    elif act == "espaco":
                        if camera_data["palavra_atual"].strip():
                            camera_data["tokens"].append(camera_data["palavra_atual"].strip())
                            camera_data["palavra_atual"] = ""
                    elif act == "apagar":
                        if camera_data["palavra_atual"]:
                            camera_data["palavra_atual"] = camera_data["palavra_atual"][:-1]
                        elif camera_data["tokens"]:
                            camera_data["tokens"].pop()
                    elif act == "limpar_historico":
                        camera_data["historico"] = []
                    elif act == "remover_item":
                        i = cmd.get("index", -1)
                        if 0 <= i < len(camera_data["historico"]):
                            camera_data["historico"].pop(i)
            except (json.JSONDecodeError, KeyError):
                pass

    async def send():
        while True:
            with data_lock:
                payload = dict(camera_data)
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(0.05)

    try:
        await asyncio.gather(receive(), send())
    except websockets.exceptions.ConnectionClosed:
        print("🔌 Cliente desconectado")


async def main():
    print("🚀 WebSocket na porta 8000...")
    async with websockets.serve(send_data, "localhost", 8000):
        print("✅ Servidor pronto! (modo inicial: alfabeto)")
        await asyncio.Future()
