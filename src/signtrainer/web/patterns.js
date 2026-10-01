// Background patterns, all generated in code (no image files, works offline).
// Picked with the buttons in #pattern-rail or ?bg=<id>; the choice is remembered
// per browser. Colours come from the theme's --pattern-* CSS variables.
(function () {
  const root = document.documentElement;
  const DEFAULT = "honeycomb";

  const svg = (w, h, body) =>
    `url("data:image/svg+xml,${encodeURIComponent(
      `<svg xmlns='http://www.w3.org/2000/svg' width='${w}' height='${h}'>${body}</svg>`,
    )}")`;

  // Each CSS pattern returns background-image / -size for the given colours.
  const PATTERNS = [
    { id: "none", name: "Bez uzorka", css: () => ({ image: "none", size: "auto" }) },
    {
      id: "honeycomb", name: "Saće",
      css: (c) => {
        const w = 27.71, h = 48;  // pointy-top hexagons, r = 16
        return {
          image: svg(w, h, `<path d='M13.86 0V8 M13.86 40V48 M13.86 8L27.71 16V32L13.86 40L0 32V16Z M27.71 16V32 M0 16V32'
            fill='none' stroke='${c.line}' stroke-width='1.6'/>`),
          size: `${w}px ${h}px`,
        };
      },
    },
    {
      id: "triangles", name: "Trokuti",
      css: (c) => ({
        image: [0, 60, -60]
          .map((a) => `repeating-linear-gradient(${a}deg, ${c.line} 0 1.2px, transparent 1.2px 26px)`)
          .join(","),
        size: "auto",
      }),
    },
    {
      id: "dots", name: "Točkice",
      css: (c) => ({ image: `radial-gradient(${c.strong} 2.2px, transparent 2.8px)`, size: "24px 24px" }),
    },
    {
      id: "chevron", name: "Cik-cak",
      css: (c) => ({
        image: svg(28, 16, `<polyline points='0,13 14,3 28,13' fill='none' stroke='${c.line}' stroke-width='2' stroke-linejoin='round'/>`),
        size: "28px 16px",
      }),
    },
    {
      id: "circles", name: "Krugovi",
      css: (c) => ({
        image: svg(36, 36, `<g fill='none' stroke='${c.line}' stroke-width='1.4'>
          <circle cx='18' cy='18' r='18'/><circle cx='0' cy='0' r='18'/><circle cx='36' cy='0' r='18'/>
          <circle cx='0' cy='36' r='18'/><circle cx='36' cy='36' r='18'/></g>`),
        size: "36px 36px",
      }),
    },
    {
      id: "diamonds", name: "Rombovi",
      css: (c) => ({
        image: [45, -45]
          .map((a) => `repeating-linear-gradient(${a}deg, ${c.line} 0 1.4px, transparent 1.4px 22px)`)
          .join(","),
        size: "auto",
      }),
    },
    {
      id: "waves", name: "Valovi",
      css: (c) => ({
        image: svg(60, 18, `<path d='M0 9 Q15 0 30 9 T60 9' fill='none' stroke='${c.line}' stroke-width='1.8'/>`),
        size: "60px 18px",
      }),
    },
    {
      id: "sunburst", name: "Zrake",
      css: (c) => ({
        image: `repeating-conic-gradient(from 0deg at 0% 100%, ${c.soft} 0 2.5deg, transparent 2.5deg 7.5deg)`,
        size: "auto",
      }),
    },
    { id: "memphis", name: "Konfeti", draw: drawMemphis },
    { id: "topo", name: "Topografija", draw: drawTopo },
    {
      id: "paper", name: "Milimetarski",
      css: (c) => ({
        image: [
          `linear-gradient(${c.faint} 1px, transparent 1px)`,
          `linear-gradient(90deg, ${c.faint} 1px, transparent 1px)`,
          `linear-gradient(${c.line} 1px, transparent 1px)`,
          `linear-gradient(90deg, ${c.line} 1px, transparent 1px)`,
        ].join(","),
        size: "14px 14px, 14px 14px, 70px 70px, 70px 70px",
      }),
    },
    {
      id: "plus", name: "Plusići",
      css: (c) => ({
        image: svg(30, 30, `<path d='M15 10v10M10 15h10' stroke='${c.strong}' stroke-width='2' stroke-linecap='round'/>`),
        size: "30px 30px",
      }),
    },
  ];

  function colours() {
    const css = getComputedStyle(root);
    const v = (name) => css.getPropertyValue(name).trim();
    return { line: v("--pattern-line"), strong: v("--pattern-strong"), soft: v("--pattern-soft"), faint: v("--pattern-faint") };
  }

  // ---- canvas patterns -----------------------------------------------------

  function seeded(seed) {
    let s = seed;
    return () => (s = (s * 16807) % 2147483647) / 2147483647;
  }

  // Playful scattered shapes, "Memphis design" style.
  function drawMemphis(ctx, w, h, c) {
    const rnd = seeded(7);
    const count = Math.max(4, Math.round((w * h) / 7000));
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    for (let i = 0; i < count; i++) {
      const x = rnd() * w, y = rnd() * h, s = 6 + rnd() * 10, kind = Math.floor(rnd() * 5);
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rnd() * Math.PI * 2);
      ctx.strokeStyle = ctx.fillStyle = rnd() < 0.5 ? c.strong : c.line;
      ctx.beginPath();
      if (kind === 0) ctx.arc(0, 0, s * 0.7, 0, Math.PI * 2);
      else if (kind === 1) { ctx.arc(0, 0, s * 0.25, 0, Math.PI * 2); ctx.fill(); }
      else if (kind === 2) { ctx.moveTo(0, -s); ctx.lineTo(s * 0.87, s * 0.5); ctx.lineTo(-s * 0.87, s * 0.5); ctx.closePath(); }
      else if (kind === 3) { ctx.moveTo(-s, 0); for (let k = 1; k <= 4; k++) ctx.lineTo(-s + k * s / 2, k % 2 ? -s / 3 : s / 3); }
      else { ctx.moveTo(-s / 2, 0); ctx.lineTo(s / 2, 0); ctx.moveTo(0, -s / 2); ctx.lineTo(0, s / 2); }
      ctx.stroke();
      ctx.restore();
    }
  }

  // Contour lines of a smooth random height field (marching squares).
  function drawTopo(ctx, w, h, c) {
    const rnd = seeded(1234);
    const waves = Array.from({ length: 8 }, () => ({
      kx: (rnd() - 0.5) * 18, ky: (rnd() - 0.5) * 18, p: rnd() * Math.PI * 2, a: 0.5 + rnd(),
    }));
    const size = Math.max(w, h, 600);
    const f = (x, y) => waves.reduce((sum, q) => sum + q.a * Math.sin(q.kx * x / size + q.ky * y / size + q.p), 0);
    const cell = 4, cols = Math.ceil(w / cell) + 1, rows = Math.ceil(h / cell) + 1;
    const v = new Float32Array(cols * rows);
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) v[j * cols + i] = f(i * cell, j * cell);
    ctx.strokeStyle = c.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let t = -5; t <= 5; t += 0.12) {
      for (let j = 0; j < rows - 1; j++) for (let i = 0; i < cols - 1; i++) {
        const a = v[j * cols + i], b = v[j * cols + i + 1], cc = v[(j + 1) * cols + i + 1], d = v[(j + 1) * cols + i];
        const x = i * cell, y = j * cell, pts = [];
        const edge = (p, q, x1, y1, x2, y2) => {
          if ((p < t) !== (q < t)) { const k = (t - p) / (q - p); pts.push(x1 + (x2 - x1) * k, y1 + (y2 - y1) * k); }
        };
        edge(a, b, x, y, x + cell, y);
        edge(b, cc, x + cell, y, x + cell, y + cell);
        edge(d, cc, x, y + cell, x + cell, y + cell);
        edge(a, d, x, y, x, y + cell);
        for (let k = 0; k + 3 < pts.length; k += 4) { ctx.moveTo(pts[k], pts[k + 1]); ctx.lineTo(pts[k + 2], pts[k + 3]); }
      }
    }
    ctx.stroke();
  }

  function canvasImage(pattern, w, h, c) {
    const canvas = document.createElement("canvas");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    pattern.draw(ctx, w, h, c);
    return canvas;
  }

  // ---- applying ------------------------------------------------------------

  const byId = Object.fromEntries(PATTERNS.map((p) => [p.id, p]));
  let current = DEFAULT;
  let pageCanvas = null;

  function apply() {
    const p = byId[current];
    const c = colours();
    root.dataset.bg = current;
    if (p.css) {
      const { image, size } = p.css(c);
      root.style.backgroundImage = image;
      root.style.backgroundSize = size;
      if (pageCanvas) pageCanvas.hidden = true;
    } else {
      root.style.backgroundImage = "none";
      if (!document.body) return;  // drawn once the page has loaded
      if (!pageCanvas) {
        pageCanvas = document.createElement("canvas");
        pageCanvas.id = "bg-canvas";
        document.body.prepend(pageCanvas);
      }
      const img = canvasImage(p, innerWidth, innerHeight, c);
      pageCanvas.width = img.width;
      pageCanvas.height = img.height;
      pageCanvas.getContext("2d").drawImage(img, 0, 0);
      pageCanvas.hidden = false;
    }
  }

  function select(id) {
    current = byId[id] ? id : DEFAULT;
    try { localStorage.setItem("pattern", current); } catch {}
    apply();
    for (const b of document.querySelectorAll("#pattern-rail button")) {
      b.setAttribute("aria-pressed", String(b.dataset.id === current));
    }
  }

  function buildRail() {
    const rail = document.getElementById("pattern-rail");
    if (!rail) return;
    const grid = rail.querySelector(".pattern-grid");
    grid.innerHTML = "";
    // Buttons are tiny, so previews use stronger colours than the page.
    const base = colours();
    const c = { line: base.strong, strong: base.strong, soft: base.line, faint: base.line };
    for (const p of PATTERNS) {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.id = p.id;
      b.title = p.name;
      b.setAttribute("aria-label", p.name);
      b.setAttribute("aria-pressed", String(p.id === current));
      if (p.css) {
        const { image, size } = p.css(c);
        b.style.backgroundImage = image;
        b.style.backgroundSize = size;
      } else {
        b.style.backgroundImage = `url(${canvasImage(p, 48, 48, c).toDataURL()})`;
        b.style.backgroundSize = "cover";
      }
      b.addEventListener("click", () => { select(p.id); b.blur(); });
      grid.append(b);
    }
  }

  // Initial choice: ?bg=... wins, then the remembered one, then the default.
  let wanted = new URLSearchParams(location.search).get("bg");
  if (!byId[wanted]) {
    try { wanted = localStorage.getItem("pattern"); } catch { wanted = null; }
  }
  current = byId[wanted] ? wanted : DEFAULT;
  apply();

  document.addEventListener("DOMContentLoaded", () => {
    apply();
    buildRail();
    let timer;
    addEventListener("resize", () => {
      clearTimeout(timer);
      timer = setTimeout(() => { if (!byId[current].css) apply(); }, 150);
    });
    // Pattern colours depend on the light/dark theme.
    new MutationObserver(() => { apply(); buildRail(); })
      .observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  });
})();
