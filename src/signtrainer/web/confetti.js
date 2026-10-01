// Background: playful scattered gold shapes ("Memphis" confetti), drawn on a
// canvas behind the page. Colours come from the theme's --pattern-* variables.
(function () {
  const root = document.documentElement;

  function seeded(seed) {
    let s = seed;
    return () => (s = (s * 16807) % 2147483647) / 2147483647;
  }

  function draw(canvas) {
    const css = getComputedStyle(root);
    const strong = css.getPropertyValue("--pattern-strong").trim();
    const line = css.getPropertyValue("--pattern-line").trim();
    const dpr = window.devicePixelRatio || 1;
    const w = innerWidth, h = innerHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.lineWidth = 2;
    ctx.lineCap = "round";

    // Same seed every time, so the layout doesn't jump on resize or theme change.
    const rnd = seeded(7);
    const count = Math.max(4, Math.round((w * h) / 7000));
    for (let i = 0; i < count; i++) {
      const x = rnd() * w, y = rnd() * h, s = 6 + rnd() * 10, kind = Math.floor(rnd() * 5);
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rnd() * Math.PI * 2);
      ctx.strokeStyle = ctx.fillStyle = rnd() < 0.5 ? strong : line;
      ctx.beginPath();
      if (kind === 0) ctx.arc(0, 0, s * 0.7, 0, Math.PI * 2);                     // ring
      else if (kind === 1) { ctx.arc(0, 0, s * 0.25, 0, Math.PI * 2); ctx.fill(); } // dot
      else if (kind === 2) {                                                       // triangle
        ctx.moveTo(0, -s); ctx.lineTo(s * 0.87, s * 0.5); ctx.lineTo(-s * 0.87, s * 0.5); ctx.closePath();
      } else if (kind === 3) {                                                     // zigzag
        ctx.moveTo(-s, 0);
        for (let k = 1; k <= 4; k++) ctx.lineTo(-s + (k * s) / 2, k % 2 ? -s / 3 : s / 3);
      } else {                                                                     // plus
        ctx.moveTo(-s / 2, 0); ctx.lineTo(s / 2, 0); ctx.moveTo(0, -s / 2); ctx.lineTo(0, s / 2);
      }
      ctx.stroke();
      ctx.restore();
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.createElement("canvas");
    canvas.id = "bg-canvas";
    document.body.prepend(canvas);
    draw(canvas);
    let timer;
    addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(() => draw(canvas), 150); });
    new MutationObserver(() => draw(canvas)).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  });
})();
