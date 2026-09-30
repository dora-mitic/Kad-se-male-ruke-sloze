// Polls the server for the live state and updates the status pill and messages.

const statusEl = document.getElementById("status");
const messageEl = document.getElementById("message");
const fpsEl = document.getElementById("fps");
const videoEl = document.getElementById("video");
const subtitleEl = document.getElementById("subtitle-text");

// Until M3 adds smoothing and hold-to-confirm, the subtitle shows the raw guess.
function showPrediction(s) {
  if (!s.model_loaded) {
    subtitleEl.className = "subtitle-placeholder";
    subtitleEl.textContent = "Model još nije istreniran";
  } else if (!s.hand_detected || !s.prediction) {
    subtitleEl.className = "subtitle-placeholder";
    subtitleEl.textContent = "Pokaži slovo…";
  } else {
    const pct = Math.round(s.confidence * 100);
    subtitleEl.className = s.confidence >= 0.6 ? "guess" : "guess guess-unsure";
    subtitleEl.innerHTML = `${s.prediction}<small>${pct} %</small>`;
  }
}

const ERRORS = {
  camera_unavailable: "Kamera nije dostupna. Provjeri je li spojena i koristi li je neki drugi program.",
};

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = `pill pill-${kind}`;
}

function showMessage(text) {
  messageEl.textContent = text;
  messageEl.hidden = !text;
}

async function poll() {
  try {
    const res = await fetch("/api/state");
    const s = await res.json();

    if (s.error) {
      setStatus("Kamera ne radi", "bad");
      showMessage(ERRORS[s.error] ?? s.error);
    } else if (!s.camera_ready) {
      setStatus("Pokrećem kameru…", "wait");
      showMessage("");
    } else if (s.hand_detected) {
      setStatus(s.handedness === "Left" ? "Lijeva ruka ✋" : "Desna ruka ✋", "ok");
      showMessage("");
    } else {
      setStatus("Pokaži ruku kameri", "wait");
      showMessage("");
    }
    showPrediction(s);
    fpsEl.textContent = s.fps ? `${Math.round(s.fps)} FPS` : "";
  } catch {
    setStatus("Server ne radi", "bad");
  }
}

// If the video stream drops (e.g. server restart), reconnect.
videoEl.addEventListener("error", () => {
  setTimeout(() => { videoEl.src = `/video?t=${Date.now()}`; }, 1000);
});

setInterval(poll, 150);
poll();
