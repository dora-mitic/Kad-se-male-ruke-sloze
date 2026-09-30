"""Local web UI: serves the page, the annotated video stream and the live state."""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from signtrainer.capture import CameraWorker

WEB_DIR = Path(__file__).parent / "web"


def create_app(camera_index: int = 0) -> FastAPI:
    worker = CameraWorker(camera_index)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker.start()
        yield
        worker.stop()

    app = FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/video")
    def video():
        # MJPEG: the browser shows it in a plain <img>, no JavaScript needed.
        def frames():
            last_id = -1
            while worker.is_alive():
                snap = worker.snapshot()
                if snap.jpeg is None or snap.frame_id == last_id:
                    time.sleep(0.01)
                    continue
                last_id = snap.frame_id
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + snap.jpeg + b"\r\n"

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/state")
    def state():
        snap = worker.snapshot()
        return {
            "error": snap.error,
            "camera_ready": snap.jpeg is not None,
            "hand_detected": snap.landmarks is not None,
            "handedness": snap.handedness,
            "fps": round(snap.fps, 1),
        }

    return app
