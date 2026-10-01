// Light/dark switch. The choice is remembered per browser; storage may be unavailable.
(function () {
  const root = document.documentElement;
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch {}
  if (saved === "dark") root.dataset.theme = "dark";

  function label(btn) {
    const dark = root.dataset.theme === "dark";
    btn.textContent = dark ? "☀" : "🌙";
    btn.title = dark ? "Svijetla tema" : "Tamna tema";
  }

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("theme-toggle");
    if (!btn) return;
    label(btn);
    btn.addEventListener("click", () => {
      const dark = root.dataset.theme !== "dark";
      if (dark) root.dataset.theme = "dark"; else delete root.dataset.theme;
      try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch {}
      label(btn);
      btn.blur();  // keep SPACE for recording, not for re-clicking the button
    });
  });
})();
