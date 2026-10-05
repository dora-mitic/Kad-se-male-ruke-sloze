// Polls the server for the live state and updates the status pill and messages.

const statusEl = document.getElementById("status");
const messageEl = document.getElementById("message");
const fpsEl = document.getElementById("fps");
const videoEl = document.getElementById("video");
const subtitleEl = document.getElementById("subtitle-text");
const readingEl = document.getElementById("reading");
const readingLetter = document.getElementById("reading-letter");
const ringEl = document.getElementById("ring");
const RING = 2 * Math.PI * 44; // circumference of the progress ring (r = 44)
ringEl.style.strokeDasharray = RING;
const anchorEl = document.getElementById("anchor");
const signCard = document.getElementById("sign-card");
const signImg = document.getElementById("sign-img");

// Reference images available on the server, e.g. {"A": "A.png"}.
let signImages = {};
async function loadSignImages() {
  try { signImages = await (await fetch("/api/signs")).json(); } catch {}
}
loadSignImages();
setInterval(loadSignImages, 5000);

// Show the last confidently recognised sign; it stays until a new one appears,
// so the card doesn't flicker on every uncertain frame.
let shownSign = null;
function showSignCard(s) {
  if (!s.hand_detected || !s.prediction || s.confidence < 0.5) return;
  const file = signImages[s.prediction];
  if (!file || s.prediction === shownSign) return;
  shownSign = s.prediction;
  signImg.src = `/signs/${encodeURIComponent(file)}`;
  signImg.alt = `Znak ${s.prediction}`;
  signCard.hidden = false;
}

const ANCHOR_LABELS = { forehead: "Ruka kod čela", chin: "Ruka kod brade", chest: "Ruka kod prsa" };

function showAnchor(s) {
  const label = ANCHOR_LABELS[s.near_anchor];
  anchorEl.hidden = !label;
  if (label) anchorEl.textContent = `📍 ${label}`;
}

// The letter being read right now: half transparent, with a ring that fills
// while it is held. The server decides when it is confirmed (subtitles.py).
function showReading(s) {
  const letter = s.hand_detected ? s.candidate : null;
  readingEl.hidden = !letter;
  if (!letter) return;
  if (readingLetter.textContent !== letter) {
    readingLetter.textContent = letter;
    ringEl.classList.add("ring-reset"); // jump back to empty instead of animating down
    ringEl.style.strokeDashoffset = RING;
    ringEl.getBoundingClientRect();
    ringEl.classList.remove("ring-reset");
  }
  ringEl.style.strokeDashoffset = RING * (1 - s.progress);
}

// Confirmed text at full strength; a newly added letter pops in.
// Split into graphemes so an emoji like ❤️ counts as one letter.
const segmenter = new Intl.Segmenter("hr", { granularity: "grapheme" });
const graphemes = (text) => Array.from(segmenter.segment(text), (g) => g.segment);
let shownText = null;
function showSubtitle(s) {
  if (!s.model_loaded) {
    subtitleEl.className = "subtitle-placeholder";
    subtitleEl.textContent = "Model još nije istreniran";
    shownText = null;
    return;
  }
  if (s.subtitle === shownText) return;
  const grew = shownText !== null && s.subtitle.length > shownText.length && s.subtitle.startsWith(shownText);
  shownText = s.subtitle;
  if (!s.subtitle.trim()) {
    subtitleEl.className = "subtitle-placeholder";
    subtitleEl.textContent = "Pokaži slovo i kratko ga zadrži";
    return;
  }
  subtitleEl.className = "subtitle-text";
  subtitleEl.textContent = "";
  const chars = graphemes(s.subtitle);
  subtitleEl.append((grew ? chars.slice(0, -1) : chars).join(""));
  if (grew) {
    const last = document.createElement("span");
    last.className = "pop";
    last.textContent = chars.at(-1);
    subtitleEl.append(last);
  }
  const caret = document.createElement("span");
  caret.className = "caret";
  subtitleEl.append(caret);
}

// Editing the subtitle: buttons beside it, or the keyboard
// (Backspace = delete a letter, Space = space, Esc = clear everything).
async function editSubtitle(action) {
  try {
    const r = await (await fetch(`/api/subtitle/${action}`, { method: "POST" })).json();
    showSubtitle({ model_loaded: true, subtitle: r.text });
  } catch {}
}
document.querySelectorAll("[data-edit]").forEach((btn) =>
  btn.addEventListener("click", () => { editSubtitle(btn.dataset.edit); btn.blur(); }));
const KEYS = { Backspace: "backspace", " ": "space", Escape: "clear" };
document.addEventListener("keydown", (e) => {
  const action = KEYS[e.key];
  if (!action || e.target.closest("input, textarea")) return;
  e.preventDefault();
  editSubtitle(action);
});

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
    showReading(s);
    showSubtitle(s);
    showSignCard(s);
    showAnchor(s);
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
