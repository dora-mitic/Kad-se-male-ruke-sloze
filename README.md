# Kad se male ruke slože: ASL sign trainer

A real-time, webcam-based trainer for **American Sign Language (ASL)**. You sign in
front of a normal laptop webcam and the app shows what you signed as live subtitles:

- **Fingerspelling:** letters A to Z (J and Z, which involve motion, come later).
- **Phrase signs:** a small, fixed set of common signs such as HELLO, THANK YOU,
  I LOVE YOU and GOODBYE.
- **Subtitles:** each recognised letter or sign appears at the bottom of the screen;
  after a pause, the words are completed into a full sentence from a local list of
  predefined phrases.

Everything runs **offline on a Windows laptop without a GPU**. Recognition uses
MediaPipe hand landmarks and a classifier trained and evaluated for this project.

This is not a general sign language translator: it only recognises the letters, signs
and phrases listed in `docs/SIGNS.md` (generated once the vocabulary exists).

Built for Smotra Sveučilišta u Zagrebu, representing FER.

## Status

| Milestone | Status |
|---|---|
| M0 Repo setup | done |
| M1 Live landmarks (local web UI) | in progress |
| M2 Data pipeline and baseline model | planned |
| M3 Real-time recognition and subtitle bar | planned |
| M4 Visitor experience | planned |
| M5 Robustness and fair readiness | planned |
| M6 Dynamic signs (J, Z, phrase signs) | planned |
| M7 Subtitles and phrases | planned |
| M8 Reference sign preview | planned |

## Install (Windows)

Requires Python 3.11 or 3.12 (MediaPipe does not reliably support newer versions).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scripts/download_models.py   # one time; afterwards everything runs offline
```

## Run

```powershell
python -m signtrainer
```

This starts a local server and opens <http://127.0.0.1:8000> in the browser: the
mirrored webcam view with the 21 tracked hand points, a status indicator, and space
reserved for subtitles and the reference sign. Stop with Ctrl+C.

Options: `--camera 1` (another webcam), `--port 8001`, `--no-browser`.

### How it works

```
webcam -> OpenCV -> MediaPipe HandLandmarker -> normalisation -> (classifier, M2+)
                          |
                          +-> FastAPI server -> browser: video (MJPEG) + live state (JSON)
```

The UI is a local web page rather than an OpenCV window because subtitles, fonts
with Croatian characters and animated GIFs are much easier to do well in HTML.

## Test

```powershell
pytest
```

## Data and assets

- How datasets are obtained and recorded: [DATA.md](DATA.md)
- Sources and licences of images/GIFs: [ASSETS.md](ASSETS.md)

## Limitations

Filled in with real evaluation numbers after M2.
