import time

import cv2

from m07_visuall_estado import camera_data, data_lock, raw_frame, raw_frame_lock


def capture_thread():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with data_lock:
            camera_data["status"] = "❌ Câmera não encontrada"
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("✅ Câmera aberta")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)
        with raw_frame_lock:
            raw_frame["img"] = frame
            raw_frame["ts"] = time.monotonic()
