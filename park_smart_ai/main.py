import asyncio
import base64
import threading

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

from ocr import ocr_iz_cropa

app = FastAPI()

model = YOLO("runs/detect/park_smart/weights/best.pt")

frame_lock = threading.Lock()

# zadnja slika i rezultati za svaki izvor
latest_data = {
    "entry": {"frame": None, "results": None},   # kamera na ulazu
    "drone": {"frame": None, "results": None},
}

# ESP32-CAM na ulazu šalje "kamera", dashboard šalje "entry" - oboje je ulaz
CAMERA_ALIAS = {"kamera": "entry"}

PREKRSAJI = ["prekrsaj_invalidsko", "prekrsaj_nije_parking", "prekrsaj_van_okvira"]


def centar_unutar(box_mali, box_veliki):
    """Provjerava da li je centar manjeg boxa (tablice) unutar većeg boxa (prekršaja)."""
    x1, y1, x2, y2 = box_mali
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X1, Y1, X2, Y2 = box_veliki
    return X1 <= cx <= X2 and Y1 <= cy <= Y2


def to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    camera_type: str = Form(...),
    camera_id: str = Form(...)
):
    camera_type = CAMERA_ALIAS.get(camera_type, camera_type)

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model.predict(img, conf=0.5, verbose=False)
    annotated = results[0].plot()

    tablice = []      # (tekst, box)
    prekrsaji = []    # (klasa, box)

    for box in results[0].boxes:
        class_name = model.names[int(box.cls[0])]
        coords = tuple(map(int, box.xyxy[0]))
        print(f"Detektovano: {class_name}, confidence: {float(box.conf[0]):.2f}")

        if class_name == "tablica":
            x1, y1, x2, y2 = coords
            tekst = ocr_iz_cropa(img[y1:y2, x1:x2])
            if tekst:
                tablice.append((tekst, coords))
        elif class_name in PREKRSAJI:
            prekrsaji.append((class_name, coords))

    # svaka tablica dobija prekršaj čiji box je sadrži
    auti = []
    for tekst, box_tablice in tablice:
        violation = next((klasa for klasa, box_p in prekrsaji if centar_unutar(box_tablice, box_p)), None)
        auti.append({
            "camera_type": camera_type,
            "camera_id": camera_id,
            "tablica": tekst,
            "violation_id": violation,
        })

    img_base64 = to_base64(annotated)

    with frame_lock:
        if camera_type in latest_data:
            latest_data[camera_type]["frame"] = img_base64
            latest_data[camera_type]["results"] = auti

    return {
        "rezultati": auti,
        "annotated_image": img_base64,
    }


@app.post("/esp_feed")
async def esp_feed(
    file: UploadFile = File(...),
    camera_type: str = Form(...),
    camera_id: str = Form(...)
):
    # samo prijem slike za pregled uživo, bez detekcije
    # (slike sa ESP32-CAM nisu dovoljno jasne za OCR)
    camera_type = CAMERA_ALIAS.get(camera_type, camera_type)

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    with frame_lock:
        if camera_type in latest_data:
            latest_data[camera_type]["frame"] = to_base64(img)

    return {"status": "ok"}


@app.get("/stream/drone")
async def stream_drone():
    async def generate():
        while True:
            with frame_lock:
                frame = latest_data["drone"]["frame"]
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       base64.b64decode(frame) + b"\r\n")
            await asyncio.sleep(1)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/latest")
async def get_latest():
    with frame_lock:
        data = latest_data["entry"]
        if data["frame"] is None:
            return {"available": False}
        return {
            "available": True,
            "annotated_image": data["frame"],
            "rezultati": data["results"],
        }


@app.get("/latest_entry")
async def get_latest_entry():
    # za ESP8266 displej na ulazu
    with frame_lock:
        data = latest_data["entry"]
        if data["frame"] is None:
            return {"available": False, "tablica": None}

        tablica = None
        if data["results"]:
            tablica = data["results"][0].get("tablica")

        return {
            "available": True,
            "tablica": tablica,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)