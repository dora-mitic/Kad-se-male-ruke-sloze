// Recording flow: show a letter, count down, record a short clip, move to the next.

const $ = (id) => document.getElementById(id);
let labels = [], rounds = 1, step = 0, busy = false, handVisible = false;

const total = () => labels.length * rounds;
const current = () => labels[step % labels.length];

function render(instruction) {
  if (step >= total()) {
    $("target").textContent = "✓";
    $("progress").textContent = "Gotovo!";
    $("progress").className = "pill pill-ok";
    $("instruction").textContent = "Sve je snimljeno, hvala! Možeš zatvoriti prozor.";
    return;
  }
  $("target").textContent = current();
  const round = Math.floor(step / labels.length) + 1;
  $("progress").textContent = `Krug ${round}/${rounds} · ${(step % labels.length) + 1}/${labels.length}`;
  $("instruction").textContent = instruction ?? `Napravi znak ${current()} i pritisni RAZMAK`;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function record() {
  if (busy || step >= total()) return;
  busy = true;
  const cd = $("countdown");
  cd.hidden = false;
  for (const n of [3, 2, 1]) { cd.textContent = n; await sleep(700); }
  cd.hidden = true;

  render("Snimam… drži znak i lagano miči ruku (bliže, dalje, zakreni)");
  await fetch(`/api/record/start/${current()}`, { method: "POST" });
  await sleep(300);
  let s;
  do { await sleep(100); s = await (await fetch("/api/record/status")).json(); } while (s.recording);

  if (s.last_frames < 10) {
    render(`Ruka se nije dobro vidjela (${s.last_frames} frameova). Pokušaj ponovno: RAZMAK`);
    if (s.last_file) await fetch("/api/record/undo", { method: "POST" });
  } else {
    step++;
    render(step < total() ? `✓ Spremljeno (${s.last_frames}). Sljedeće: ${current()}. RAZMAK kad si spremna.` : undefined);
  }
  busy = false;
}

async function undo() {
  if (busy || step === 0) return;
  const r = await (await fetch("/api/record/undo", { method: "POST" })).json();
  step--;
  render(r.deleted ? `Obrisano. Ponovi ${current()}: RAZMAK` : undefined);
}

document.addEventListener("keydown", (e) => {
  if (e.code === "Space") { e.preventDefault(); record(); }
  else if (e.key === "r" || e.key === "R") undo();
  else if (e.key === "ArrowRight" && !busy && step < total()) { step++; render(); }
  else if (e.key === "ArrowLeft" && !busy && step > 0) { step--; render(); }
});

async function pollState() {
  try {
    const s = await (await fetch("/api/state")).json();
    handVisible = s.hand_detected;
    $("target").classList.toggle("hand-ok", handVisible);
    $("fps").textContent = s.error ? "Kamera ne radi" : `${Math.round(s.fps)} FPS`;
  } catch { $("fps").textContent = "Server ne radi"; }
}

(async () => {
  const c = await (await fetch("/api/record/config")).json();
  labels = c.labels; rounds = c.rounds;
  $("who").textContent = `· ${c.person} / ${c.session}`;
  render();
  setInterval(pollState, 200);
})();
