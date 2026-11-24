/* ==========================================================================
   Proofs — Example DAG Viewer (home page)
   Zero dependencies · Reads docs/examples/dags.json (with file:// fallback)
   ========================================================================== */

(function () {
  "use strict";

  // Used when the manifest can't be fetched (e.g. the page is opened via file://).
  const FALLBACK_DAGS = [
    {
      file: "semisimplicity.html",
      title: "Semisimplicity",
      stats: "1,141 lemmas · 2,855 citations · 3 contributors",
      size: "8.7 MB",
      command: "proofs export semisimplicity -o examples -n semisimplicity --format html"
    }
  ];

  async function loadManifest() {
    try {
      const res = await fetch("examples/dags.json", { cache: "no-cache" });
      if (!res.ok) throw new Error("manifest fetch failed");
      const data = await res.json();
      if (Array.isArray(data)) {
        const valid = data.filter((d) => d && typeof d.file === "string" && d.file.trim());
        if (valid.length) return valid;
      }
    } catch {
      /* fall through */
    }
    return FALLBACK_DAGS;
  }

  function initDagViewer() {
    const frame = document.getElementById("dag-frame");
    if (!frame) return;
    const $ = (id) => document.getElementById(id);
    const poster = $("dag-poster");
    const loadBtn = $("dag-load");
    const navGroup = document.querySelector(".dag-nav-group");

    let dags = FALLBACK_DAGS.slice();
    let idx = 0;
    let loaded = false; // the visitor opted in to the (large) iframe

    const srcFor = (entry) => (entry.file.includes("/") ? entry.file : `examples/${entry.file}`);
    const setText = (el, text) => {
      if (!el) return;
      el.textContent = text || "";
      el.hidden = !text;
    };

    function render() {
      const entry = dags[idx] || FALLBACK_DAGS[0];
      const url = srcFor(entry);
      const title = entry.title || entry.file;

      setText($("dag-viewer-title"), title);
      setText($("dag-viewer-stats"), entry.stats);
      setText($("dag-viewer-command"), entry.command);
      setText($("dag-size"), entry.size ? `${entry.size} · math rendered with KaTeX, works offline` : "");
      if ($("dag-viewer-count")) $("dag-viewer-count").textContent = `${idx + 1} / ${dags.length}`;
      if ($("dag-open-full")) $("dag-open-full").setAttribute("href", url);
      if (navGroup) navGroup.hidden = dags.length < 2;

      frame.setAttribute("title", `${title} — interactive citation graph`);
      if (loaded) {
        frame.setAttribute("src", url);
        frame.hidden = false;
        if (poster) poster.hidden = true;
      }
    }

    function step(delta) {
      if (dags.length < 2) return;
      idx = (idx + delta + dags.length) % dags.length;
      render();
    }

    if ($("dag-prev")) $("dag-prev").addEventListener("click", () => step(-1));
    if ($("dag-next")) $("dag-next").addEventListener("click", () => step(1));
    if (loadBtn) {
      loadBtn.addEventListener("click", () => {
        loaded = true;
        render();
        frame.focus();
      });
    } else {
      loaded = true; // no poster on the page: load straight away
    }

    render();
    loadManifest().then((list) => {
      dags = list;
      idx = 0;
      render();
    });
  }

  document.addEventListener("DOMContentLoaded", initDagViewer);
})();
