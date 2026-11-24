// The interactive citation graph (export/html.py). The layout comes laid
// out from graph.py; this draws it on two canvases — the base (every edge
// and node) and an overlay (the focused lemma's edges and neighbours over
// a wash), so a hover costs O(its degree), not O(the graph).
//
// The base is never drawn in the frame that needs it. Its picture is
// rendered off screen, in slices of a few milliseconds a frame, and each
// frame only copies the last finished picture onto the screen, scaled and
// shifted from the view it was rendered at to the view now. So panning and
// zooming cost one image copy a frame however big the graph is; the sharp
// picture for the new view is rendered once the view has been still for a
// moment, and replaces the stretched one when it is complete.
(() => {
  "use strict";
  const D = JSON.parse(document.getElementById("data").textContent);
  const N = D.nodes, E = D.edges, n = N.length;
  const cites = Array.from({ length: n }, () => []);
  const citers = Array.from({ length: n }, () => []);
  for (const [a, b] of E) { cites[a].push(b); citers[b].push(a); }
  const index = new Map(N.map((d, i) => [d.id, i]));

  // Hit testing: nodes grouped by y (one group per row, and the strip),
  // each group sorted by x, so a pointer is two binary searches away.
  const byY = new Map();
  N.forEach((d, i) => { if (!byY.has(d.y)) byY.set(d.y, []); byY.get(d.y).push(i); });
  const ys = [...byY.keys()].sort((a, b) => a - b);
  const groups = ys.map((y) => byY.get(y).sort((a, b) => N[a].x - N[b].x));
  const maxR = N.reduce((m, d) => Math.max(m, d.r), 1);
  // Draw order: small nodes first, so a big node never hides a small one's
  // edge of colour entirely.
  const bySize = N.map((_, i) => i).sort((a, b) => N[a].r - N[b].r);

  const $ = (id) => document.getElementById(id);
  const stage = $("stage"), base = $("base"), over = $("over");
  const tip = $("tip"), panel = $("panel");
  const bctx = base.getContext("2d"), octx = over.getContext("2d");
  let W = 0, H = 0, dpr = 1;
  let k = 1, tx = 0, ty = 0, fitK = 1; // screen = world * k + t
  let hover = -1, pinned = -1, chain = false, fitted = false;
  let C = {};

  const COL = 24; // graph.py's COL: the gap between neighbours in a row
  // Fit margins: the legend above, the stacked labels below.
  const TOP = 96, PAD = 48, BOTTOM = 80;

  // ---------------------------------------------------------------- colours
  function readColours() {
    const cs = getComputedStyle(document.documentElement);
    const v = (name) => cs.getPropertyValue(name).trim();
    C = {
      surface: v("--surface"), ink: v("--ink"), ink2: v("--ink2"),
      muted: v("--muted"), edgeInk: v("--edge-ink"), edgeAlpha: parseFloat(v("--edge-alpha")) || 0.1,
      wash: v("--wash"), hl: v("--hl"), settle: v("--settle"),
      users: [0, 1, 2, 3, 4, 5, 6, 7].map((i) => v("--u" + i)), other: v("--other"),
    };
  }
  const userColour = (o) => (o < 8 ? C.users[o] : C.other);
  // A lemma that settles the conjecture: its kind ("proved", "disproved")
  // in d.res, ringed twice at these offsets past its radius, and labelled.
  const SETTLE_RINGS = [3.5, 7];
  const SETTLES_TEXT = { proved: "proves the conjecture", disproved: "disproves the conjecture" };
  const userVar = (o) => (o < 8 ? `var(--u${o})` : "var(--other)");

  // ----------------------------------------------------------------- view
  // A node's on-screen radius: with the zoom while zoomed out (the layout
  // keeps neighbours apart at that scale), with its square root zoomed in,
  // so close up the nodes stay small beside the gaps; never under 1.5px.
  const rsOf = (r) => Math.max(1.5, k < 1 ? r * k : r * Math.sqrt(k));
  const rs = (d) => rsOf(d.r);
  const sx = (d) => d.x * k + tx;
  const sy = (d) => d.y * k + ty;

  function fit() {
    const w = Math.max(D.width, 1), h = Math.max(D.height, 1);
    k = Math.min((W - 2 * PAD) / w, (H - TOP - BOTTOM) / h, 3);
    if (!isFinite(k) || k <= 0) k = 1;
    fitK = k;
    tx = (W - D.width * k) / 2;
    ty = TOP + (H - TOP - BOTTOM - D.height * k) / 2;
  }

  function zoomAt(px, py, f) {
    const nk = Math.min(Math.max(k * f, fitK / 4), 40);
    f = nk / k;
    tx = px - (px - tx) * f;
    ty = py - (py - ty) * f;
    k = nk;
    moved();
  }

  // Centre a lemma in the part of the stage the panel leaves visible.
  function centre(i) {
    const d = N[i];
    const side = !panel.hidden && W > 640 ? panel.offsetWidth : 0;
    const bottom = !panel.hidden && W <= 640 ? panel.offsetHeight : 0;
    if (COL * k < 10) k = Math.min(10 / COL, 40);
    tx = (W - side) / 2 - d.x * k;
    ty = (H - bottom) / 2 - d.y * k;
    moved();
  }

  function resize() {
    const r = stage.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    W = r.width; H = r.height;
    for (const c of [base, over, ...pictures.flatMap((p) => [p.e, p.n])]) {
      c.width = Math.max(1, Math.round(W * dpr));
      c.height = Math.max(1, Math.round(H * dpr));
    }
    if (!fitted && W > 0 && H > 0) { fit(); fitted = true; }
    rerender();
  }

  // --------------------------------------------------------------- drawing
  // A picture: the edges (drawn opaque, so overlapping edges merge as one
  // stroke would merge them, then copied at the edge alpha) and the nodes
  // and labels, each on its own canvas, and the view they were drawn at.
  // Two of them: the one on screen, and the one being rendered.
  const picture = () => ({ e: document.createElement("canvas"), n: document.createElement("canvas"), view: null });
  const pictures = [picture(), picture()];
  let front = pictures[0], job = null, settle = 0, frame = 0;
  const SLICE_MS = 10, SETTLE_MS = 140;

  // A repaint of the screen: the picture, moved to the current view, and
  // the overlay. Cheap — what every hover and every gesture frame does.
  function draw() {
    if (!frame) frame = requestAnimationFrame(() => { frame = 0; compose(); drawOver(); });
  }

  // The view moved: repaint now from the picture there is, and render a
  // sharp one for the new view once it has been still for SETTLE_MS.
  function moved() {
    if (job) { job.cancelled = true; job = null; }
    clearTimeout(settle);
    settle = setTimeout(() => render(false), SETTLE_MS);
    draw();
  }

  // What is drawn changed (the size, the theme): the picture on screen is
  // no longer right, so render the new one at once — synchronously, inside
  // a budget of SYNC_MS, which covers all but very large graphs, so it
  // lands in the same frame as the page's own colours. Past the budget the
  // rest is rendered in slices, shown as it fills in.
  const SYNC_MS = 250;
  function rerender() {
    clearTimeout(settle);
    front.view = null;
    render(true, SYNC_MS);
  }

  function compose() {
    const c = bctx;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.globalAlpha = 1;
    c.fillStyle = C.surface;
    c.fillRect(0, 0, base.width, base.height);
    const p = job && job.live ? job.pic : front;
    if (!p.view) return;
    // A point the picture drew at p.view lands, in the current view, at
    // (its picture position - p.view.t) * f + t.
    const f = k / p.view.k;
    c.setTransform(dpr * f, 0, 0, dpr * f, dpr * (tx - p.view.tx * f), dpr * (ty - p.view.ty * f));
    c.globalAlpha = C.edgeAlpha;
    c.drawImage(p.e, 0, 0, W, H);
    c.globalAlpha = 1;
    c.drawImage(p.n, 0, 0, W, H);
  }

  // Render the current view into the picture not on screen: first as much
  // as fits in budget milliseconds at once, then a slice per frame. A move
  // cancels it (moved()); when it completes, it goes on screen. live shows
  // it while it fills in.
  function render(live, budget = 0) {
    if (job) job.cancelled = true;
    const pic = pictures[0] === front ? pictures[1] : pictures[0];
    const j = job = { pic, live, cancelled: false, phase: 0, i: 0, view: { k, tx, ty } };
    for (const c of [pic.e, pic.n]) {
      const x = c.getContext("2d");
      x.setTransform(1, 0, 0, 1, 0, 0);
      x.clearRect(0, 0, c.width, c.height);
    }
    const finish = () => {
      pic.view = j.view;
      front = pic;
      job = null;
      draw();
    };
    if (budget > 0) {
      const until = performance.now() + budget;
      while (performance.now() < until && j.phase < 3) slice(j);
      // Done: paint it now, not at the next frame.
      if (j.phase >= 3) { finish(); compose(); drawOver(); return; }
    }
    const step = () => {
      if (j.cancelled) return;
      const until = performance.now() + SLICE_MS;
      while (performance.now() < until && j.phase < 3) slice(j);
      if (j.phase < 3) { if (j.live) draw(); requestAnimationFrame(step); return; }
      finish();
    };
    requestAnimationFrame(step);
  }

  // The visible part of the world, in world units, for culling: an edge's
  // curve stays inside the box of its two ends (its control points share
  // their x and lie between their y), so a box test is exact.
  function visible() {
    const m = 40 / k;
    return [(0 - tx) / k - m, (0 - ty) / k - m, (W - tx) / k + m, (H - ty) / k + m];
  }
  function edgeVisible(v, a, b) {
    const A = N[a], B = N[b];
    return Math.max(A.x, B.x) >= v[0] && Math.min(A.x, B.x) <= v[2] &&
           Math.max(A.y, B.y) >= v[1] && Math.min(A.y, B.y) <= v[3];
  }

  // One slice of a render: a run of edges, a run of nodes, or the labels.
  function slice(j) {
    if (j.phase === 0) {
      const c = j.pic.e.getContext("2d"), v = visible();
      c.setTransform(dpr * k, 0, 0, dpr * k, dpr * tx, dpr * ty);
      c.lineWidth = 1 / k;
      c.strokeStyle = C.edgeInk;
      c.beginPath();
      const end = Math.min(E.length, j.i + 4000);
      for (; j.i < end; j.i++) {
        const [a, b] = E[j.i];
        if (edgeVisible(v, a, b)) edgePath(c, a, b);
      }
      c.stroke();
      if (j.i >= E.length) { j.phase = 1; j.i = 0; }
    } else if (j.phase === 1) {
      const c = j.pic.n.getContext("2d");
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      const end = Math.min(n, j.i + 4000);
      paintNodes(c, bySize.slice(j.i, end));
      j.i = end;
      if (j.i >= n) j.phase = 2;
    } else {
      labels(j.pic.n.getContext("2d"));
      j.phase = 3;
    }
  }

  function edgePath(c, a, b) {
    const A = N[a], B = N[b], dy = (B.y - A.y) / 2;
    c.moveTo(A.x, A.y);
    c.bezierCurveTo(A.x, A.y + dy, B.x, B.y - dy, B.x, B.y);
  }

  function strokeEdges(c, list, colour, px) {
    if (!list.length) return;
    const v = visible();
    c.setTransform(dpr * k, 0, 0, dpr * k, dpr * tx, dpr * ty);
    c.lineWidth = px / k;
    c.strokeStyle = colour;
    c.beginPath();
    for (const [a, b] of list) if (edgeVisible(v, a, b)) edgePath(c, a, b);
    c.stroke();
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // Nodes batched by (owner, suspended): one path, one fill and one ring
  // stroke per batch; then the settling lemmas' double rings, in one
  // stroke. Off-screen nodes are skipped.
  function paintNodes(c, list) {
    const batches = new Map(), rings = [];
    for (const i of list) {
      const d = N[i], x = sx(d), y = sy(d), r = rs(d);
      const reach = r + (d.res ? SETTLE_RINGS[1] + 1 : 0);
      if (x < -reach || x > W + reach || y < -reach || y > H + reach) continue;
      if (d.res) rings.push(x, y, r);
      const key = d.o * 2 + (d.s ? 1 : 0);
      if (!batches.has(key)) batches.set(key, []);
      batches.get(key).push(x, y, r);
    }
    for (const [key, pts] of batches) {
      const col = userColour(key >> 1), hollow = key & 1;
      c.beginPath();
      for (let j = 0; j < pts.length; j += 3) {
        c.moveTo(pts[j] + pts[j + 2], pts[j + 1]);
        c.arc(pts[j], pts[j + 1], pts[j + 2], 0, 2 * Math.PI);
      }
      c.fillStyle = hollow ? C.surface : col;
      c.fill();
      c.strokeStyle = hollow ? col : C.surface;
      c.lineWidth = hollow ? 1.6 : 1;
      c.stroke();
    }
    if (rings.length) {
      c.beginPath();
      for (let j = 0; j < rings.length; j += 3) {
        for (const off of SETTLE_RINGS) {
          c.moveTo(rings[j] + rings[j + 2] + off, rings[j + 1]);
          c.arc(rings[j], rings[j + 1], rings[j + 2] + off, 0, 2 * Math.PI);
        }
      }
      c.strokeStyle = C.settle;
      c.lineWidth = 1.5;
      c.stroke();
    }
  }

  function label(c, text, x, y, above, colour, size) {
    c.font = `${size}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
    c.textAlign = "center";
    c.textBaseline = above ? "bottom" : "top";
    c.lineJoin = "round";
    c.lineWidth = 4;
    c.strokeStyle = C.surface;
    c.strokeText(text, x, y);
    c.fillStyle = colour;
    c.fillText(text, x, y);
  }

  const shortId = (id) => {
    const p = id.split("/");
    return p.length >= 3 ? `${p[0]}/…/${p[p.length - 1]}` : id;
  };
  const isTop = (d) => d.row === D.rows - 1 && d.n === 0;
  // The permanent labels (graph.labelled), placed greedily: each at the
  // first of three steps away from its node (above for a top-row lemma,
  // below otherwise) that overlaps no label already placed, or not at all.
  // A settling lemma's comes first (graph.labelled puts it there), above
  // its rings, in the ink, saying what it settles.
  function permanentLabels(c) {
    const boxes = [], h = 16;
    c.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    for (const i of D.labels) {
      const d = N[i], top = isTop(d) || !!d.res;
      const r = rs(d) + (d.res ? SETTLE_RINGS[1] : 0);
      const text = d.res ? `${shortId(d.id)} · ${SETTLES_TEXT[d.res]}`
        : shortId(d.id) + (top ? "" : ` (${d.n})`);
      const w = c.measureText(text).width + 8, x = sx(d);
      for (let step = 0; step < 3; step++) {
        const off = r + 6 + step * h, y = top ? sy(d) - off : sy(d) + off;
        const box = [x - w / 2, top ? y - h : y, w, h];
        if (boxes.some((b) => b[0] < box[0] + box[2] && box[0] < b[0] + b[2] &&
                              b[1] < box[1] + box[3] && box[1] < b[1] + b[3])) continue;
        boxes.push(box);
        label(c, text, x, y, top, d.res ? C.ink : C.ink2, 12);
        break;
      }
    }
  }

  function labels(c) {
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    // Zoomed in far enough that neighbours are 56px apart: every visible
    // lemma gets its id's last part.
    const labelled = new Set(D.labels);
    if (COL * k >= 56) {
      for (let i = 0; i < n; i++) {
        if (labelled.has(i)) continue;
        const d = N[i], x = sx(d), y = sy(d);
        if (x < -50 || x > W + 50 || y < -20 || y > H + 20) continue;
        label(c, d.id.split("/").pop(), x, y + rs(d) + 4, false, C.muted, 10);
      }
    }
    permanentLabels(c);
    // The strip's caption: left of its first lemma, or, with no room there
    // on screen (the strip is the widest row, so at the fitted view its
    // first lemma is near the edge), under it, left-aligned with it —
    // nothing is ever drawn below the strip, past the id labels a close
    // zoom puts under each lemma.
    if (D.strip) {
      const first = groups[groups.length - 1][0];
      const d = N[first], text = "not yet connected";
      c.font = "italic 13px system-ui, -apple-system, sans-serif";
      c.fillStyle = C.muted;
      if (sx(d) - 14 - c.measureText(text).width >= 8) {
        c.textAlign = "right";
        c.textBaseline = "middle";
        c.fillText(text, sx(d) - 14, sy(d));
      } else {
        c.textAlign = "left";
        c.textBaseline = "top";
        c.fillText(text, Math.max(8, sx(d) - rs(d)), sy(d) + rs(d) + (COL * k >= 56 ? 20 : 8));
      }
    }
  }

  // The full chain of a lemma: everything it rests on (down through what
  // it cites) and everything resting on it (up through what cites it),
  // with the edges along the way. Cached per lemma.
  const chains = new Map();
  function chainOf(p) {
    if (chains.has(p)) return chains.get(p);
    const walk = (next) => {
      const seen = new Set([p]), queue = [p], edges = [];
      while (queue.length) {
        const u = queue.pop();
        for (const v of next[u]) {
          edges.push(next === cites ? [u, v] : [v, u]);
          if (!seen.has(v)) { seen.add(v); queue.push(v); }
        }
      }
      return { seen, edges };
    };
    const down = walk(cites), up = walk(citers);
    const out = {
      nodes: [...new Set([...down.seen, ...up.seen])],
      edges: down.edges.concat(up.edges),
      below: down.seen.size - 1,
      above: up.seen.size - 1,
    };
    chains.set(p, out);
    return out;
  }

  function drawOver() {
    const c = octx;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    const f = hover >= 0 ? hover : pinned;
    if (f < 0) return;
    let nodes, edges;
    if (f === pinned && chain) {
      ({ nodes, edges } = chainOf(f));
    } else {
      nodes = [f, ...cites[f], ...citers[f]];
      edges = cites[f].map((b) => [f, b]).concat(citers[f].map((a) => [a, f]));
    }
    c.fillStyle = C.wash;
    c.fillRect(0, 0, W, H);
    strokeEdges(c, edges, C.hl, 1.4);
    paintNodes(c, nodes.sort((a, b) => N[a].r - N[b].r));
    // The focus ring, outside a settling lemma's own rings.
    const d = N[f], fr = rs(d) + (d.res ? SETTLE_RINGS[1] + 3.5 : 3);
    c.beginPath();
    c.arc(sx(d), sy(d), fr, 0, 2 * Math.PI);
    c.strokeStyle = C.hl;
    c.lineWidth = 2;
    c.stroke();
    label(c, d.res ? `${shortId(d.id)} · ${SETTLES_TEXT[d.res]}` : shortId(d.id),
          sx(d), sy(d) - fr - 4, true, C.ink, 12);
  }

  // ----------------------------------------------------------- hit testing
  function lowerBound(arr, v, key) {
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (key(arr[m]) < v) lo = m + 1; else hi = m; }
    return lo;
  }

  // The lemma under (px, py): among the nodes whose radius (plus 8px of
  // slack) reaches the pointer, the one it is nearest the centre of,
  // relative to its size — so a small node beside a big one stays
  // reachable.
  function hit(px, py) {
    const wx = (px - tx) / k, wy = (py - ty) / k, slack = 8 / k;
    const reach = rsOf(maxR) / k + slack;
    let best = -1, bestScore = Infinity;
    for (let g = lowerBound(ys, wy - reach, (y) => y); g < ys.length && ys[g] <= wy + reach; g++) {
      const arr = groups[g];
      for (let j = lowerBound(arr, wx - reach, (i) => N[i].x); j < arr.length; j++) {
        const d = N[arr[j]];
        if (d.x > wx + reach) break;
        const rad = rs(d) / k + slack;
        const dist = Math.hypot(d.x - wx, d.y - wy);
        if (dist <= rad && dist / rad < bestScore) { bestScore = dist / rad; best = arr[j]; }
      }
    }
    return best;
  }

  // ------------------------------------------------------------------- TeX
  // Statements and proofs are LaTeX text: the math goes through KaTeX, and
  // the text-mode commands models commonly write are turned into HTML. The
  // text is escaped before any of it is, so nothing in a lemma's text
  // reaches the page as markup; anything not recognised stays as source.
  const esc = (s) => s.replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
  const MATH = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|(\\begin\{(equation|align|gather|multline|alignat|flalign|eqnarray)(\*?)\}[\s\S]*?\\end\{\5\6\})|\$((?:\\[\s\S]|[^$\\])+?)\$/g;
  const INLINE = [
    [/\\(?:textbf|bf)\{([^{}]*)\}/g, "<b>$1</b>"],
    [/\\(?:emph|textit|textsl|it)\{([^{}]*)\}/g, "<i>$1</i>"],
    [/\\texttt\{([^{}]*)\}/g, "<code>$1</code>"],
    [/\\underline\{([^{}]*)\}/g, "<u>$1</u>"],
    [/\\(?:text|textup|textrm|textnormal|mbox|textsc)\{([^{}]*)\}/g, "$1"],
    [/\\(?:(?:sub)*section|paragraph)\*?\{([^{}]*)\}/g, "<b>$1</b> "],
    [/\\label\{[^{}]*\}/g, ""],
    [/\\(?:eq)?ref\{([^{}]*)\}/g, "($1)"],
    [/\\cite(?:\[([^\]]*)\])?\{([^{}]*)\}/g, (m, a, b) => `[${b}${a ? ", " + a : ""}]`],
  ];
  const ENV_NAME = (e) => {
    const s = e.replace(/\*$/, "");
    return s.charAt(0).toUpperCase() + s.slice(1);
  };

  function katexHtml(tex, display) {
    if (typeof katex === "undefined") return `<code>${esc(tex)}</code>`;
    try {
      return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore", output: "html" });
    } catch (e) {
      return `<code>${esc(tex)}</code>`;
    }
  }

  function texToHtml(src) {
    // Escaped characters the scan must not mistake for delimiters or
    // comments, held aside: \\ (a line break), \$ and \%.
    let s = String(src).replace(/[\u0000-\u0003]/g, "")
      .replace(/\\\\/g, "\u0002").replace(/\\\$/g, "\u0001").replace(/\\%/g, "\u0003")
      .replace(/%.*$/gm, "");
    const math = [];
    const unhold = (t) => t.replace(/\u0001/g, "\\$").replace(/\u0002/g, "\\\\").replace(/\u0003/g, "\\%");
    s = s.replace(MATH, (m, dd, br, pa, env, envName, star, inl) => {
      let html;
      if (env) html = katexHtml(unhold(envName === "eqnarray" ? env.replace(/eqnarray/g, "align") : env), true);
      else if (inl !== undefined) html = katexHtml(unhold(inl), false);
      else html = katexHtml(unhold(dd ?? br ?? pa), dd !== undefined || br !== undefined);
      math.push(html);
      return `\u0000${math.length - 1}\u0000`;
    });
    // Markdown the models mix in: rules, headings, bold, italics, inline
    // code, and list items or numbered steps starting a line. An italic
    // needs a space or bracket outside each star, so an exponent star
    // (F^*) stays as written.
    s = esc(s)
      .replace(/^[ \t]*([=*_-])\1{2,}[ \t]*$/gm, "<hr>")
      .replace(/^[ \t]*#{1,6}[ \t]+(.+)$/gm, "<b>$1</b><br>")
      .replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>")
      .replace(/(^|[\s(])\*([^\s*](?:[^*\n]*[^\s*])?)\*(?=[\s).,;:!?]|$)/gm, "$1<i>$2</i>")
      .replace(/``/g, "“")
      .replace(/`([^`\n]+)`/g, "<code>$1</code>")
      .replace(/\n(?=[ \t]*(?:[-*•][ \t]|\(?\d+[.)][ \t]|\([a-z]{1,4}\)[ \t]))/g, "<br>")
      .replace(/\\begin\{(itemize|enumerate|description)\}/g, (m, e) => (e === "enumerate" ? "<ol>" : "<ul>"))
      .replace(/\\end\{(itemize|enumerate|description)\}/g, (m, e) => (e === "enumerate" ? "</ol>" : "</ul>"))
      .replace(/\\item\s*\[([^\]]*)\]/g, "<li><b>$1</b> ")
      .replace(/\\item\b/g, "<li>")
      .replace(/\\begin\{([A-Za-z]+\*?)\}(?:\[([^\]]*)\])?/g, (m, e, t) =>
        `<div class="env env-${e.replace(/\*$/, "")}"><span class="env-name">${ENV_NAME(e)}${t ? ` (${t})` : ""}.</span> `)
      .replace(/\\end\{[A-Za-z]+\*?\}/g, "</div>");
    for (let pass = 0; pass < 6; pass++) {
      const before = s;
      for (const [re, rep] of INLINE) s = s.replace(re, rep);
      if (s === before) break;
    }
    s = s
      .replace(/\\(?:noindent|medskip|smallskip|bigskip|qedhere|par|newline|indent|hfill|qquad|quad)\b/g, " ")
      .replace(/\\(?:ldots|dots)\b/g, "…")
      .replace(/\\([&#_{}])/g, "$1").replace(/\\&amp;/g, "&amp;")
      .replace(/---/g, "—").replace(/--/g, "–")
      .replace(/&#39;&#39;/g, "”")
      .replace(/~/g, "&nbsp;")
      .replace(/\n\s*\n/g, '<span class="pb"></span>')
      .replace(/\u0001/g, "$").replace(/\u0002/g, "<br>").replace(/\u0003/g, "%");
    return s.replace(/\u0000(\d+)\u0000/g, (m, i) => math[+i]);
  }

  const rendered = new Map();
  function tex(i, field) {
    const key = i + field;
    if (!rendered.has(key)) rendered.set(key, texToHtml(N[i][field] || ""));
    return rendered.get(key);
  }

  // ------------------------------------------------------- tooltip & panel
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function meta(i) {
    const d = N[i], m = el("div", "meta");
    m.textContent = `${D.users[d.o].id} · cites ${cites[i].length} · cited by ${citers[i].length}`;
    if (d.res) m.append(el("span", "badge settle", SETTLES_TEXT[d.res]));
    if (d.s) m.append(el("span", "badge", "suspended"));
    return m;
  }

  function showTip(i, px, py) {
    tip.replaceChildren();
    tip.append(el("div", "mono", N[i].id), meta(i));
    const body = el("div", "tex");
    if (N[i].st) body.innerHTML = tex(i, "st");
    else body.append(el("span", "empty", "No statement recorded."));
    tip.append(body);
    if (i !== pinned) tip.append(el("div", "more", "Click for the proof."));
    tip.hidden = false;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let x = px + 16, y = py + 16;
    if (x + tw > W - 8) x = Math.max(8, px - tw - 16);
    if (y + th > H - 8) y = Math.max(8, H - th - 8);
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }

  function linkList(ids) {
    const box = el("div", "links");
    for (const id of ids) {
      const j = index.get(id);
      if (j === undefined) {
        const s = el("span", "", id);
        s.title = "Not in this export";
        box.append(s);
        continue;
      }
      const b = el("button");
      const dot = el("span", "dot");
      dot.style.background = userVar(N[j].o);
      b.append(dot, document.createTextNode(shortId(id)));
      b.title = id;
      b.addEventListener("click", () => { pin(j); centre(j); });
      box.append(b);
    }
    return box;
  }

  function section(title, child) {
    panel.append(el("h2", "", title), child);
  }

  function fillPanel(i) {
    const d = N[i];
    panel.replaceChildren();
    const head = el("div", "head");
    const dot = el("span", "dot");
    dot.style.background = userVar(d.o);
    dot.style.marginTop = "5px";
    const close = el("button", "close", "×");
    close.setAttribute("aria-label", "Close");
    close.addEventListener("click", () => pin(-1));
    head.append(dot, el("div", "id mono", d.id), close);
    panel.append(head, meta(i));

    const box = el("label");
    const cb = el("input");
    cb.type = "checkbox";
    cb.checked = chain;
    const count = el("span");
    const setCount = () => {
      const ch = chainOf(i);
      count.textContent = chain
        ? `Full chain: rests on ${ch.below}, ${ch.above} rest on it`
        : "Highlight the full chain";
    };
    cb.addEventListener("change", () => { chain = cb.checked; setCount(); draw(); });
    setCount();
    box.append(cb, count);
    panel.append(box);

    const st = el("div", "tex");
    if (d.st) st.innerHTML = tex(i, "st");
    else st.append(el("span", "empty", "No statement recorded."));
    section("Statement", st);
    const pf = el("div", "tex");
    if (d.pf) pf.innerHTML = tex(i, "pf");
    else pf.append(el("span", "empty", "No proof recorded."));
    section("Proof", pf);
    const cited = cites[i].map((j) => N[j].id).concat(d.ext);
    section(`Cites (${cited.length})`, cited.length ? linkList(cited) : el("div", "empty", "Nothing."));
    section(`Cited by (${citers[i].length})`,
            citers[i].length ? linkList(citers[i].map((j) => N[j].id)) : el("div", "empty", "Nothing yet."));
    if (d.refs.length) {
      const refs = el("div", "links");
      for (const r of d.refs) refs.append(el("span", "", r));
      section(`References (${d.refs.length})`, refs);
    }
    panel.scrollTop = 0;
  }

  function pin(i) {
    pinned = i;
    if (i < 0) {
      panel.hidden = true;
    } else {
      fillPanel(i);
      panel.hidden = false;
    }
    tip.hidden = true;
    draw();
  }

  function setHover(i, px, py) {
    if (i !== hover) { hover = i; draw(); }
    stage.classList.toggle("point", i >= 0);
    if (i >= 0) showTip(i, px, py);
    else tip.hidden = true;
  }

  // --------------------------------------------------------------- events
  const pointers = new Map();
  let drag = null, pinch = null;
  const local = (e) => {
    const r = stage.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };

  stage.addEventListener("pointerdown", (e) => {
    try { stage.setPointerCapture(e.pointerId); } catch (err) { /* a pointer the browser no longer tracks */ }
    pointers.set(e.pointerId, local(e));
    if (pointers.size === 2) {
      const [p, q] = [...pointers.values()];
      pinch = { dist: Math.hypot(p[0] - q[0], p[1] - q[1]) };
      drag = null;
    } else if (pointers.size === 1) {
      const [x, y] = local(e);
      drag = { x, y, tx, ty, moved: false };
    }
  });
  stage.addEventListener("pointermove", (e) => {
    const [x, y] = local(e);
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, [x, y]);
    if (pinch && pointers.size === 2) {
      const [p, q] = [...pointers.values()];
      const dist = Math.hypot(p[0] - q[0], p[1] - q[1]);
      if (pinch.dist > 0) zoomAt((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, dist / pinch.dist);
      pinch.dist = dist;
      return;
    }
    if (drag) {
      const dx = x - drag.x, dy = y - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) > 4) { drag.moved = true; stage.classList.add("grab"); setHover(-1); }
      if (drag.moved) { tx = drag.tx + dx; ty = drag.ty + dy; moved(); return; }
    }
    setHover(hit(x, y), x, y);
  });
  const release = (e) => {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinch = null;
    if (drag && !drag.moved && e.type === "pointerup") {
      const [x, y] = local(e);
      pin(hit(x, y));
    }
    if (pointers.size === 0) drag = null;
    stage.classList.remove("grab");
  };
  stage.addEventListener("pointerup", release);
  stage.addEventListener("pointercancel", release);
  stage.addEventListener("pointerleave", () => { if (!drag) setHover(-1); });
  stage.addEventListener("wheel", (e) => {
    e.preventDefault();
    const [x, y] = local(e);
    const step = e.deltaMode === 1 ? 0.05 : 0.0015;
    zoomAt(x, y, Math.exp(-e.deltaY * step));
    setHover(-1);
  }, { passive: false });
  stage.addEventListener("dblclick", (e) => { const [x, y] = local(e); zoomAt(x, y, 2); });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { pin(-1); return; }
    if (e.target instanceof HTMLInputElement) return;
    if (e.key === "+" || e.key === "=") zoomAt(W / 2, H / 2, 1.25);
    else if (e.key === "-") zoomAt(W / 2, H / 2, 0.8);
    else if (e.key === "0") { fit(); moved(); }
  });

  $("fit").addEventListener("click", () => { fit(); moved(); });

  // Search: an exact id, or the one id containing what was typed (a hex
  // suffix is usually enough).
  const search = $("search");
  const list = $("ids");
  for (const d of N) { const o = document.createElement("option"); o.value = d.id; list.append(o); }
  function find() {
    const q = search.value.trim();
    if (!q) return;
    let i = index.get(q);
    if (i === undefined) {
      const ql = q.toLowerCase();
      const hits = N.map((d, j) => j).filter((j) => N[j].id.toLowerCase().includes(ql));
      if (hits.length === 1) i = hits[0];
    }
    search.setAttribute("aria-invalid", i === undefined ? "true" : "false");
    search.style.borderColor = i === undefined ? "var(--u7)" : "";
    if (i !== undefined) { pin(i); centre(i); }
  }
  search.addEventListener("change", find);
  search.addEventListener("keydown", (e) => { if (e.key === "Enter") find(); });

  // Theme: Light, Dark, or Theme (the docs' palette, "proofs", light or
  // dark by the OS setting). The page opens in the one html.py stamped on
  // <html data-theme> unless the viewer picked another before, remembered
  // per browser when it allows it.
  const THEME_KEY = "proofs-graph-theme";
  const root = document.documentElement;
  const themeBtns = [...$("theme").querySelectorAll("button")];
  const themes = themeBtns.map((b) => b.dataset.theme);
  function applyTheme() {
    for (const b of themeBtns) b.setAttribute("aria-pressed", String(b.dataset.theme === root.dataset.theme));
    readColours();
    if (W > 0) rerender();
  }
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (themes.includes(saved)) root.dataset.theme = saved;
  } catch (e) { /* storage unavailable: the exported theme */ }
  for (const b of themeBtns) {
    b.addEventListener("click", () => {
      root.dataset.theme = b.dataset.theme;
      try { localStorage.setItem(THEME_KEY, b.dataset.theme); } catch (e) { /* not remembered */ }
      applyTheme();
    });
  }
  // Theme follows the OS setting: a change repaints it (the stylesheet has
  // already switched its tokens).
  window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
    if (root.dataset.theme === "proofs") applyTheme();
  });

  // ---------------------------------------------------------------- legend
  const legend = $("legend");
  D.users.slice(0, 8).forEach((u, o) => {
    const row = el("div");
    const dot = el("span", "dot");
    dot.style.background = userVar(o);
    row.append(dot, el("span", "mono", `${u.id} (${u.count} lemma${u.count === 1 ? "" : "s"})`));
    legend.append(row);
  });
  if (D.users.length > 8) {
    const rest = D.users.slice(8);
    const row = el("div");
    const dot = el("span", "dot");
    dot.style.background = "var(--other)";
    row.append(dot, el("span", "mono", `${rest.length} other users (${rest.reduce((s, u) => s + u.count, 0)} lemmas)`));
    legend.append(row);
  }
  if (N.some((d) => d.s)) {
    const row = el("div");
    row.append(el("span", "dot hollow"), el("span", "", "suspended"));
    legend.append(row);
  }
  for (const kind of ["proved", "disproved"]) {
    if (!N.some((d) => d.res === kind)) continue;
    const row = el("div", "settled");
    row.append(el("span", "dot settle"), el("span", "", SETTLES_TEXT[kind]));
    legend.append(row);
  }
  legend.append(el("div", "note",
    `${n} lemma${n === 1 ? "" : "s"} · ${E.length} citation${E.length === 1 ? "" : "s"} · node size ∝ times cited`));

  applyTheme();
  new ResizeObserver(resize).observe(stage);
})();
