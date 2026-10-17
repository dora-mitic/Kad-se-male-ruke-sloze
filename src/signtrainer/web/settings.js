// Settings panel: how long a letter must be held and how sure the model must be.
// The server keeps the values (subtitles.py); "Zadano" goes back to config.py.

(function () {
  const panel = document.getElementById("settings");
  const toggle = document.getElementById("settings-toggle");
  const confirmInput = document.getElementById("confirm-seconds");
  const confidenceInput = document.getElementById("min-confidence");
  const confirmValue = document.getElementById("confirm-value");
  const confidenceValue = document.getElementById("confidence-value");
  const resetBtn = document.getElementById("settings-reset");
  let defaults = null;

  function show(s) {
    confirmInput.value = s.confirm_seconds;
    confidenceInput.value = s.min_confidence;
    label();
    const isDefault = defaults
      && Number(confirmInput.value) === defaults.confirm_seconds
      && Number(confidenceInput.value) === defaults.min_confidence;
    resetBtn.disabled = isDefault;
  }

  function label() {
    confirmValue.textContent = `${Number(confirmInput.value).toFixed(1)} s`;
    confidenceValue.textContent = `${Math.round(confidenceInput.value * 100)} %`;
  }

  async function load() {
    try {
      const s = await (await fetch("/api/settings")).json();
      defaults = s.defaults;
      [confirmInput.min, confirmInput.max] = s.ranges.confirm_seconds;
      [confidenceInput.min, confidenceInput.max] = s.ranges.min_confidence;
      show(s);
    } catch {}
  }

  async function save(values) {
    try {
      const res = await fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      if (res.ok) show(await res.json());
    } catch {}
  }

  const current = () => ({
    confirm_seconds: Number(confirmInput.value),
    min_confidence: Number(confidenceInput.value),
  });

  // Labels follow the slider at once; the server is told when it is let go.
  [confirmInput, confidenceInput].forEach((input) => {
    input.addEventListener("input", label);
    input.addEventListener("change", () => save(current()));
  });
  resetBtn.addEventListener("click", () => { if (defaults) save(defaults); resetBtn.blur(); });

  function setOpen(open) {
    panel.hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
    if (open) load();
  }
  toggle.addEventListener("click", () => { setOpen(panel.hidden); toggle.blur(); });
  document.addEventListener("click", (e) => {
    if (!panel.hidden && !panel.contains(e.target) && e.target !== toggle) setOpen(false);
  });

  load();
})();
