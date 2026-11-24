/* ==========================================================================
   Proofs Docs — Lightweight Client Reactivity (Zero Dependencies)
   - Mobile sidebar drawer
   - Cross-page search built from page headings (/ or Ctrl+K)
   - Copy-to-clipboard buttons on code blocks
   - Reactive tabs & filterable reference tables
   ========================================================================== */

(function () {
  "use strict";

  // Pages searched, in sidebar order. The index itself is built from each
  // page's headings (h1, and h2/h3 with an id) plus the text under them, so it
  // stays in step with the content; nothing else needs editing when a page changes.
  const PAGES = [
    { file: "index.html", section: "Home" },
    { file: "getting-started.html", section: "Getting Started" },
    { file: "structure.html", section: "Structure" },
    { file: "verification.html", section: "Verification" },
    { file: "collaboration.html", section: "Collaboration" },
    { file: "miscellaneous.html", section: "Miscellaneous" },
    { file: "reference.html", section: "CLI & Settings" }
  ];

  /* Mobile Sidebar Drawer ------------------------------------------------- */
  function initSidebar() {
    const btn = document.getElementById("menu-toggle");
    const backdrop = document.getElementById("sidebar-backdrop");
    if (!btn) return;

    function setOpen(open) {
      document.body.classList.toggle("sidebar-open", open);
      btn.setAttribute("aria-expanded", String(open));
    }

    btn.addEventListener("click", () => {
      setOpen(!document.body.classList.contains("sidebar-open"));
    });
    if (backdrop) {
      backdrop.addEventListener("click", () => setOpen(false));
    }
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") setOpen(false);
    });
    document.querySelectorAll(".sidebar a").forEach((a) => {
      a.addEventListener("click", () => setOpen(false));
    });
  }

  /* Instant Cross-Page Search --------------------------------------------- */
  function currentFile() {
    const last = location.pathname.split("/").pop();
    return last && last.endsWith(".html") ? last : "index.html";
  }

  // One entry per heading: its title, link, and the text up to the next heading.
  function entriesFrom(doc, page) {
    const root = doc.querySelector("main") || doc.body;
    const heads = Array.from(root.querySelectorAll("h1, h2[id], h3[id]"));
    return heads.map((h, i) => {
      const next = heads[i + 1];
      let text = "";
      for (let n = h.nextElementSibling; n && n !== next && !n.contains(next); n = n.nextElementSibling) {
        text += " " + n.textContent;
      }
      if (next && !text) {
        // heading is nested (e.g. h1 inside a section): fall back to its parent's text
        text = (h.parentElement && h.parentElement.textContent) || "";
      }
      const title = h.textContent.trim().replace(/\s+/g, " ");
      return {
        title,
        page: h.tagName === "H1" ? page.file : `${page.file}#${h.id}`,
        section: page.section,
        titleLc: title.toLowerCase(),
        textLc: text.replace(/\s+/g, " ").toLowerCase()
      };
    });
  }

  let indexPromise = null;
  function buildIndex() {
    if (indexPromise) return indexPromise;
    const here = currentFile();
    indexPromise = Promise.all(
      PAGES.map(async (page) => {
        if (page.file === here) return entriesFrom(document, page);
        try {
          const res = await fetch(page.file);
          if (!res.ok) throw new Error(res.status);
          const doc = new DOMParser().parseFromString(await res.text(), "text/html");
          return entriesFrom(doc, page);
        } catch {
          // file:// or offline: still offer the page itself by name
          return [{ title: page.section, page: page.file, section: page.section,
                    titleLc: page.section.toLowerCase(), textLc: "" }];
        }
      })
    ).then((lists) => lists.flat());
    return indexPromise;
  }

  function initSearch() {
    const input = document.getElementById("doc-search");
    const panel = document.getElementById("search-results");
    if (!input || !panel) return;

    async function render(query) {
      const q = query.trim().toLowerCase();
      if (!q) {
        panel.hidden = true;
        panel.innerHTML = "";
        return;
      }
      const index = await buildIndex();
      if (input.value.trim().toLowerCase() !== q) return; // a newer keystroke won
      const tokens = q.split(/\s+/);
      const matches = index
        .map((item) => {
          let score = 0;
          for (const t of tokens) {
            if (item.titleLc.includes(t)) score += 3;
            else if (item.textLc.includes(t)) score += 1;
            else return null;
          }
          return { item, score };
        })
        .filter(Boolean)
        .sort((a, b) => b.score - a.score)
        .slice(0, 8)
        .map((m) => m.item);

      if (!matches.length) {
        panel.innerHTML = `<div class="search-empty">No matches for "${escapeHtml(query)}"</div>`;
      } else {
        panel.innerHTML = matches
          .map(
            (m) =>
              `<a class="search-item" href="${escapeHtml(m.page)}">` +
              `<span class="search-item-title">${escapeHtml(m.title)}</span>` +
              `<span class="search-item-meta">${escapeHtml(m.section)}</span>` +
              `</a>`
          )
          .join("");
      }
      panel.hidden = false;
    }

    input.addEventListener("input", () => render(input.value));
    input.addEventListener("focus", () => {
      buildIndex();
      if (input.value.trim()) render(input.value);
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        const first = panel.querySelector(".search-item");
        if (first) location.href = first.getAttribute("href");
      }
    });
    document.addEventListener("click", (e) => {
      if (!panel.contains(e.target) && e.target !== input) {
        panel.hidden = true;
      }
    });
    document.addEventListener("keydown", (e) => {
      const tag = (document.activeElement && document.activeElement.tagName) || "";
      if ((e.key === "/" && !/INPUT|TEXTAREA/.test(tag)) || ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k")) {
        e.preventDefault();
        input.focus();
        input.select();
      } else if (e.key === "Escape" && document.activeElement === input) {
        panel.hidden = true;
        input.blur();
      }
    });
  }

  /* Copy-to-Clipboard on Code Blocks -------------------------------------- */
  function initCopyButtons() {
    document.querySelectorAll("pre").forEach((pre) => {
      if (pre.dataset.noCopy !== undefined) return;
      const wrap = document.createElement("div");
      wrap.className = "code-block";
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(pre);

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "copy-btn";
      btn.textContent = "Copy";
      btn.addEventListener("click", async () => {
        const code = pre.querySelector("code");
        const text = (code ? code.innerText : pre.innerText).trim();
        try {
          await navigator.clipboard.writeText(text);
          btn.textContent = "Copied";
          btn.classList.add("copied");
          setTimeout(() => {
            btn.textContent = "Copy";
            btn.classList.remove("copied");
          }, 1400);
        } catch {
          btn.textContent = "Failed";
          setTimeout(() => (btn.textContent = "Copy"), 1200);
        }
      });
      wrap.appendChild(btn);
    });
  }

  /* Reactive Tabs --------------------------------------------------------- */
  function initTabs() {
    document.querySelectorAll("[data-tabs]").forEach((group) => {
      const buttons = Array.from(group.querySelectorAll("[role='tab']"));
      const panels = Array.from(group.querySelectorAll("[role='tabpanel']"));

      function activate(targetId) {
        buttons.forEach((b) => {
          const on = b.getAttribute("aria-controls") === targetId;
          b.setAttribute("aria-selected", String(on));
        });
        panels.forEach((p) => {
          p.hidden = p.id !== targetId;
        });
      }

      buttons.forEach((b) => {
        b.addEventListener("click", () => activate(b.getAttribute("aria-controls")));
      });
    });
  }

  /* Filterable Tables (Reference page) ------------------------------------ */
  function initTableFilters() {
    document.querySelectorAll("[data-filter-target]").forEach((input) => {
      const selector = input.getAttribute("data-filter-target");
      const countEl = document.getElementById(input.getAttribute("data-filter-count") || "");
      const rows = Array.from(document.querySelectorAll(`${selector} tbody tr`));

      function update() {
        const q = input.value.trim().toLowerCase();
        let shown = 0;
        rows.forEach((row) => {
          const match = !q || row.textContent.toLowerCase().includes(q);
          row.hidden = !match;
          if (match) shown += 1;
        });
        if (countEl) {
          countEl.textContent = `${shown} / ${rows.length} shown`;
        }
      }

      input.addEventListener("input", update);
      update();
    });
  }

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (ch) => {
      const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
      return map[ch];
    });
  }

  /* Manual Light / Dark Theme Toggle -------------------------------------- */
  const THEME_KEY = "proofs-docs-theme";

  function initThemeToggle() {
    const btn = document.getElementById("theme-toggle");
    if (!btn) return;
    const mq = window.matchMedia("(prefers-color-scheme: light)");

    function effective() {
      const explicit = document.documentElement.dataset.theme;
      if (explicit === "light" || explicit === "dark") return explicit;
      return mq.matches ? "light" : "dark";
    }

    function syncLabel() {
      const cur = effective();
      const label = cur === "dark" ? "Switch to light theme" : "Switch to dark theme";
      btn.setAttribute("aria-label", label);
      btn.setAttribute("title", label);
    }

    function syncFrame(theme) {
      const frame = document.getElementById("dag-frame");
      if (!frame || frame.hidden) return;
      try {
        const fdoc = frame.contentDocument;
        if (!fdoc) return;
        const fcur = fdoc.documentElement.dataset.theme || (mq.matches ? "light" : "dark");
        const fbtn = fdoc.getElementById("theme");
        if (fbtn && fcur !== theme) fbtn.click();
      } catch {
        /* cross-origin or not loaded yet */
      }
    }

    btn.addEventListener("click", () => {
      const next = effective() === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try {
        localStorage.setItem(THEME_KEY, next);
      } catch {
        /* storage unavailable */
      }
      syncLabel();
      syncFrame(next);
    });

    if (mq.addEventListener) {
      mq.addEventListener("change", syncLabel);
    }
    syncLabel();
  }

  document.addEventListener("DOMContentLoaded", () => {
    initThemeToggle();
    initSidebar();
    initSearch();
    initCopyButtons();
    initTabs();
    initTableFilters();
  });
})();
