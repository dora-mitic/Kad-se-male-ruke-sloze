// Background pattern: "paper" (CSS grid), "topo" (contour lines drawn on a canvas)
// or "none". Pick with ?bg=..., the choice is remembered per browser.
(function () {
  const root = document.documentElement;
  const PATTERNS = ["paper", "topo", "none"];
  let bg = new URLSearchParams(location.search).get("bg");
  try {
    if (PATTERNS.includes(bg)) localStorage.setItem("bg", bg);
    else bg = localStorage.getItem("bg");
  } catch {}
  if (!PATTERNS.includes(bg)) bg = "paper";
  root.dataset.bg = bg;
  if (bg !== "topo") return;

  // Smooth random "terrain": a sum of sine waves over normalised coordinates.
  function heightField(seed) {
    let s = seed;
    const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
    const waves = Array.from({ length: 8 }, () => ({
      kx: (rnd() - 0.5) * 18, ky: (rnd() - 0.5) * 18, p: rnd() * Math.PI * 2, a: 0.5 + rnd(),
    }));
    return (x, y) => waves.reduce((sum, w) => sum + w.a * Math.sin(w.kx * x + w.ky * y + w.p), 0);
  }

  const LEVEL_SPACING = 0.12;  // smaller = denser lines
  const CELL = 4;              // marching-squares cell size in CSS pixels

  function draw(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = innerWidth, h = innerHeight, size = Math.max(w, h);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.strokeStyle = getComputedStyle(root).getPropertyValue("--topo-line");
    ctx.lineWidth = 1;

    const f = heightField(1234);
    const cols = Math.ceil(w / CELL) + 1, rows = Math.ceil(h / CELL) + 1;
    const v = new Float32Array(cols * rows);
    for (let j = 0; j < rows; j++)
      for (let i = 0; i < cols; i++) v[j * cols + i] = f((i * CELL) / size, (j * CELL) / size);

    // Marching squares: for every level, find where it crosses each cell's edges.
    ctx.beginPath();
    for (let t = -5; t <= 5; t += LEVEL_SPACING) {
      for (let j = 0; j < rows - 1; j++) for (let i = 0; i < cols - 1; i++) {
        const a = v[j * cols + i], b = v[j * cols + i + 1];
        const c = v[(j + 1) * cols + i + 1], d = v[(j + 1) * cols + i];
        const x = i * CELL, y = j * CELL, pts = [];
        const edge = (p, q, x1, y1, x2, y2) => {
          if ((p < t) !== (q < t)) {
            const k = (t - p) / (q - p);
            pts.push(x1 + (x2 - x1) * k, y1 + (y2 - y1) * k);
          }
        };
        edge(a, b, x, y, x + CELL, y);
        edge(b, c, x + CELL, y, x + CELL, y + CELL);
        edge(d, c, x, y + CELL, x + CELL, y + CELL);
        edge(a, d, x, y, x, y + CELL);
        for (let k = 0; k + 3 < pts.length; k += 4) {
          ctx.moveTo(pts[k], pts[k + 1]);
          ctx.lineTo(pts[k + 2], pts[k + 3]);
        }
      }
    }
    ctx.stroke();
  }

  document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.createElement("canvas");
    canvas.id = "bg-topo";
    document.body.prepend(canvas);
    draw(canvas);
    let timer;
    addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(() => draw(canvas), 150); });
    // Line colour depends on the theme, so redraw when it changes.
    new MutationObserver(() => draw(canvas)).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  });
})();
