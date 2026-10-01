"""Local web UI: serves the page, the annotated video stream and the live state."""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from signtrainer import config
from signtrainer.capture import CameraWorker

WEB_DIR = Path(__file__).parent / "web"


def sign_images() -> dict[str, str]:
    """Map each label to its reference image file name, e.g. {"A": "A.png"}."""
    found = {}
    for path in sorted(config.SIGN_ASSETS_DIR.glob("*")):
        if path.suffix.lower() in config.SIGN_EXTENSIONS:
            found.setdefault(path.stem.upper(), path.name)
    return found


def create_app(camera_index: int = 0, recorder_factory=None) -> FastAPI:
    """Build the web app. `recorder_factory(worker)` enables the /record page."""
    worker = CameraWorker(camera_index)
    recorder = recorder_factory(worker) if recorder_factory else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker.start()
        yield
        worker.stop()

    app = FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    config.SIGN_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/signs", StaticFiles(directory=config.SIGN_ASSETS_DIR), name="signs")

    @app.middleware("http")
    async def revalidate_pages(request, call_next):
        # Without this, browsers reuse an old style.css/app.js for hours after an
        # update. "no-cache" still allows caching but checks the ETag every time.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith(("/static", "/record", "/signs")):
            response.headers["Cache-Control"] = "no-cache"
        return response

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

    @app.get("/api/signs")
    def signs():
        # Rescanned on every call, so new images show up without a restart.
        return sign_images()

    @app.get("/api/state")
    def state():
        snap = worker.snapshot()
        return {
            "error": snap.error,
            "camera_ready": snap.jpeg is not None,
            "hand_detected": snap.landmarks is not None,
            "handedness": snap.handedness,
            "body_detected": snap.pose is not None,
            "near_anchor": snap.near_anchor,
            "model_loaded": snap.model_loaded,
            "prediction": snap.prediction,
            "confidence": round(snap.confidence, 3),
            "fps": round(snap.fps, 1),
        }

    if recorder is not None:

        @app.get("/record")
        def record_page():
            return FileResponse(WEB_DIR / "record.html")

        @app.get("/api/record/config")
        def record_config():
            return {"labels": recorder.labels, "rounds": recorder.rounds, **recorder.meta}

        @app.get("/api/record/status")
        def record_status():
            return recorder.status()

        @app.post("/api/record/start/{label}")
        def record_start(label: str):
            if label not in recorder.labels:
                raise HTTPException(400, f"unknown label {label}")
            return {"started": recorder.start(label)}

        @app.post("/api/record/undo")
        def record_undo():
            return {"deleted": recorder.delete_last()}

    return app
