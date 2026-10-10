(() => {
  "use strict";
  const root = document.querySelector(".mfm[data-mfm-root]");
  const dataEl = document.getElementById("mfm-data");
  if (!root || !dataEl || root.dataset.mfmReady) return;
  let data;
  try {
    data = JSON.parse(dataEl.textContent);
  } catch {
    return;
  }
  if (!data || !Array.isArray(data.features)) return;
  root.dataset.mfmReady = "1";

  const NS = "http://www.w3.org/2000/svg";
  const SLOTS = 10;
  const CARRIERS = [["electricity", "Electricity", "el"], ["gas", "Gas", "gas"], ["heat", "Heat", "heat"]];
  const ICON = {
    el: '<path d="M7.4.5 2 7.1h3.4l-1 4.4 5.6-6.7H6.5z"/>',
    gas: '<path d="M6.2.5c.6 2.3 3.6 3.4 3.6 6.8a3.8 3.8 0 0 1-7.6 0c0-1.9 1-3 1.9-3.9.1 1.3.6 2 1.4 2.2-.4-1.9-.2-3.5.7-5.1z"/>',
    heat: '<path d="M3.8 11.2C1.6 9.4 5.6 8 3.6 6S5.8 2.6 3.8.8M8.4 11.2C6.2 9.4 10.2 8 8.2 6s2.2-3.4.2-5.2" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/>',
  };
  const P = { pad: 18, ls: 8, lab: 15, gap: 24, peel: 42, pitch: 30, vgap: 9, side: 18, first: 38, hdGap: 15, bend: 14, hub: 40, lane: 9 };
  const CANDIDATES = [230, 205, 185, 172, 162, 152, 142, 134, 126].map((w) => [1, w])
    .concat([230, 205, 185, 168, 154, 142, 130, 120, 110].map((w) => [2, w]), [185, 154, 130, 110].map((w) => [3, w]));

  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const behavior = () => (reduced.matches ? "instant" : "smooth");
  const str = (v) => (v == null ? "" : String(v));
  const arr = (v) => (Array.isArray(v) ? v : []);
  const esc = (s) => str(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const md = (s) => esc(s).replace(/`([^`]+)`/g, (m, c) => `<code>${c.replace(/([._(])/g, "$1<wbr>")}</code>`);
  const plain = (s) => str(s).replace(/`/g, "");
  const icon = (k) => `<svg class="mfm-ic mfm-ic-${k}" viewBox="0 0 12 12" aria-hidden="true" focusable="false">${ICON[k]}</svg>`;
  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html) n.innerHTML = html;
    return n;
  };
  const svgEl = (tag, attrs) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  };
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

  let groups = arr(data.groups).filter((g) => g && g.id != null).map((g, i) => ({
    id: str(g.id), title: str(g.title) || str(g.id), tagline: str(g.tagline), open: g.style === "open",
    slot: Number.isInteger(g.slot) ? g.slot : (i % SLOTS) + 1, order: Number(g.order) || 0, items: [],
  }));
  if (groups.some((g) => g.order)) groups.sort((a, b) => a.order - b.order);
  const gById = new Map(groups.map((g) => [g.id, g]));
  const byId = new Map();
  const KNOWN_C = new Set(CARRIERS.map((c) => c[0]));
  for (const raw of data.features) {
    if (!raw || raw.id == null || byId.has(str(raw.id))) continue;
    let g = gById.get(str(raw.group));
    if (!g) {
      g = gById.get("other");
      if (!g) {
        g = { id: "other", title: "More features", tagline: "Features that are not on a line yet.", open: false, slot: 0, items: [] };
        groups.push(g);
        gById.set(g.id, g);
      }
    }
    const rc = arr(raw.carriers).map(str);
    const f = {
      id: str(raw.id), group: g.id, title: str(raw.title) || str(raw.id), short: str(raw.short), tagline: str(raw.tagline),
      details: str(raw.details), core: raw.level === "core", snippet: str(raw.snippet),
      highlights: arr(raw.highlights).map(str).filter(Boolean), api: arr(raw.api).map(str).filter(Boolean),
      docs: arr(raw.docs).filter((d) => d && d.url).map((d) => ({ label: str(d.label) || str(d.url), kind: str(d.kind) || "other", url: str(d.url) })),
      all: !rc.length || rc.includes("all"), carriers: rc.filter((c) => KNOWN_C.has(c)),
      keywords: arr(raw.keywords).map(str), related: arr(raw.related).map(str), tasks: [],
    };
    g.items.push(f);
    byId.set(f.id, f);
  }
  groups = groups.filter((g) => g.items.length);
  groups.forEach((g, i) => {
    g.n = i + 1;
    g.items.sort((a, b) => b.core - a.core);
  });
  const ordered = groups.flatMap((g) => g.items);
  const tasks = arr(data.tasks).filter((t) => t && t.id && t.want);
  for (const t of tasks) for (const id of arr(t.features)) if (byId.has(id)) byId.get(id).tasks.push(t);
  const fold = (s) => str(s).toLowerCase()
    .replace(/([a-z])is(e|ed|es|er|ers|ing|ation|ations)\b/g, "$1iz$2")
    .replace(/([a-z])ys(e|ed|es|er|ers|ing)\b/g, "$1yz$2")
    .replace(/[^a-z0-9]+/g, " ");
  const hay = (parts) => " " + fold(plain(parts.join("\n")).replace(/([a-z])([A-Z])/g, "$1 $2")) + " ";
  for (const f of ordered) {
    f.rel = [...new Set(f.related)].filter((id) => id !== f.id && byId.has(id));
    f.tasks.sort((a, b) => arr(a.features).length - arr(b.features).length);
    f.hays = [[4, hay([f.title])], [3, hay([f.short, ...f.keywords])], [2, hay(f.api)], [1, hay([f.tagline, ...f.tasks.map((t) => t.want)])]];
  }
  const starters = groups.map((g) => g.items.find((f) => f.core)).filter(Boolean);
  const nCore = ordered.filter((f) => f.core).length;
  const carrierCount = Object.fromEntries(CARRIERS.map((c) => [c[0], ordered.filter((f) => !f.all && f.carriers.includes(c[0])).length]));

  const carrierIcons = (f) => (f.all ? "" : CARRIERS.filter((c) => f.carriers.includes(c[0])).map((c) => icon(c[2])).join(""));
  const carrierWords = (f) => (f.all ? "every carrier" : CARRIERS.filter((c) => f.carriers.includes(c[0])).map((c) => c[1].toLowerCase()).join(", "));
  const num = (g) => `<span class="mfm-num" data-p="${g.slot}" aria-hidden="true">${g.n}</span>`;

  const state = { sel: null, hover: null, hoverG: null, kb: null, q: "", carriers: new Set(), rove: null };
  const sts = new Map();
  let view = { mode: null };
  let introDone = false;

  root.classList.add("mfm-js");
  root.querySelectorAll(".mfm-fallback [id]").forEach((n) => n.removeAttribute("id"));

  const ui = el("div", "mfm-ui");
  ui.innerHTML = `<div class="mfm-bar">
<label class="mfm-search"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m20 20-4.6-4.6"/></svg><span class="mfm-sr">Search features</span><input type="search" placeholder="Search features, keywords, API" autocomplete="off" spellcheck="false"></label>
<div class="mfm-carriers" role="group" aria-label="Highlight features for one energy carrier">${CARRIERS.map(
    (c) => `<button type="button" class="mfm-cbtn" data-c="${c[0]}" aria-pressed="false">${icon(c[2])}${c[1]} <span class="mfm-cnt">${carrierCount[c[0]]}<span class="mfm-sr"> features</span></span></button>`,
  ).join("")}</div>
<p class="mfm-status"><span class="mfm-status-t"></span><button type="button" class="mfm-totask" hidden>Try the task filter</button><button type="button" class="mfm-clear" hidden>Clear</button></p>
</div>
<nav class="mfm-jump" aria-label="Jump to a line">${groups.map((g) => `<a href="#mfm-sg-${esc(g.id)}" data-g="${esc(g.id)}" data-p="${g.slot}" aria-label="${g.n}. ${esc(g.title)}">${g.n}</a>`).join("")}</nav>
<div class="mfm-frame"><div class="mfm-stage"></div>
<div class="mfm-cap"><div class="mfm-cap-i" aria-hidden="true"></div><div class="mfm-cap-x" hidden><button type="button" class="mfm-cap-go"></button><button type="button" class="mfm-cap-clr" aria-label="Clear the selection"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8m0-8-8 8"/></svg></button></div></div></div>
<section class="mfm-panel" aria-label="Feature details"></section>
<div class="mfm-sr" role="status"></div>`;
  root.appendChild(ui);
  const $ = (s) => ui.querySelector(s);
  const stage = $(".mfm-stage");
  const panel = $(".mfm-panel");
  const input = $(".mfm-search input");
  const statusT = $(".mfm-status-t");
  const clearBtn = $(".mfm-clear");
  const toTasks = $(".mfm-totask");
  const cap = $(".mfm-cap");
  const capInfo = $(".mfm-cap-i");
  const capX = $(".mfm-cap-x");
  const capGo = $(".mfm-cap-go");
  const live = ui.lastElementChild;
  const tasksRoot = document.querySelector(".mfm.mfm-tasks");
  const tasksSection = tasksRoot && tasksRoot.closest("section[id]");

  let sayTimer = 0;
  function say(msg, delay = 30) {
    clearTimeout(sayTimer);
    sayTimer = setTimeout(() => {
      live.textContent = live.textContent === msg ? msg + "\u00a0" : msg;
    }, delay);
  }

  let lastPt = null, hoverLock = null;
  document.addEventListener("pointermove", (e) => {
    lastPt = [e.screenX, e.screenY];
    if (hoverLock && (lastPt[0] !== hoverLock[0] || lastPt[1] !== hoverLock[1])) {
      hoverLock = null;
      stage.classList.remove("is-hover-locked");
    }
  }, { passive: true });
  function lockHover() {
    hoverLock = lastPt || [-1, -1];
    stage.classList.add("is-hover-locked");
    if (state.hover || state.hoverG) {
      state.hover = state.hoverG = null;
      paint();
    }
  }

  function stationEl(f) {
    const g = gById.get(f.group);
    const b = el("button", "mfm-st" + (f.core ? " is-core" : ""));
    b.type = "button";
    b.tabIndex = -1;
    b.id = "feature-" + f.id;
    b.dataset.id = f.id;
    b.dataset.p = g.slot;
    b.setAttribute("aria-label", `${f.title}, ${g.title}, ${f.core ? "core" : "advanced"} feature${f.all || f.carriers.length ? ", " + carrierWords(f) : ""}`);
    if (f.tagline) b.setAttribute("aria-description", plain(f.tagline));
    const ci = carrierIcons(f);
    b.innerHTML = `<span class="mfm-mk" aria-hidden="true"></span><span class="mfm-lb">${esc(f.title)}${ci ? `<span class="mfm-cr">${ci}</span>` : ""}</span>`;
    sts.set(f.id, b);
    return b;
  }

  function roundedPath(pts, r) {
    const f = (v) => Math.round(v * 10) / 10;
    let d = `M${f(pts[0][0])} ${f(pts[0][1])}`;
    for (let i = 1; i < pts.length - 1; i++) {
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i], [x2, y2] = pts[i + 1];
      const l1 = Math.hypot(x1 - x0, y1 - y0), l2 = Math.hypot(x2 - x1, y2 - y1);
      if (!l1 || !l2) continue;
      const rr = Math.min(r, l1 / 2, l2 / 2);
      d += `L${f(x1 - ((x1 - x0) / l1) * rr)} ${f(y1 - ((y1 - y0) / l1) * rr)}Q${f(x1)} ${f(y1)} ${f(x1 + ((x2 - x1) / l2) * rr)} ${f(y1 + ((y2 - y1) / l2) * rr)}`;
    }
    const last = pts[pts.length - 1];
    return d + `L${f(last[0])} ${f(last[1])}`;
  }

  function buildMap() {
    sts.clear();
    const map = el("div", "mfm-map", '<h2 class="mfm-sr">Feature map</h2>');
    const svg = svgEl("svg", { class: "mfm-svg", "aria-hidden": "true", focusable: "false" });
    map.appendChild(svg);
    const hds = new Map();
    for (const g of groups) {
      const wrap = el("div", "mfm-grp");
      wrap.dataset.p = g.slot;
      const hd = el("h3", "mfm-hd", `${num(g)}<span class="mfm-hd-t">${esc(g.title)}</span>${g.tagline ? `<span class="mfm-sr">. ${esc(plain(g.tagline))}</span>` : ""}`);
      hd.id = "mfm-hd-" + g.id;
      hd.dataset.g = g.id;
      const ul = el("ul", "mfm-ul");
      ul.setAttribute("aria-labelledby", hd.id);
      for (const f of g.items) ul.appendChild(el("li")).appendChild(stationEl(f));
      wrap.append(hd, ul);
      hds.set(g.id, hd);
      map.appendChild(wrap);
    }
    const origin = el("p", "mfm-origin-t");
    origin.textContent = "import monee";
    origin.setAttribute("aria-hidden", "true");
    const probe = el("span", "mfm-lb mfm-probe", "Electrolysers and turbines");
    probe.setAttribute("aria-hidden", "true");
    map.append(origin, probe);
    stage.replaceChildren(map);
    view = { mode: "map", map, svg, hds, origin, probe, cache: new Map(), sig: 0, key: "" };
  }

  function measure() {
    const { map, hds, origin } = view;
    sts.forEach((b) => (b.style.left = b.style.top = "0"));
    hds.forEach((h) => (h.style.left = h.style.top = "0"));
    void map.offsetWidth;
    const lab = new Map(), hd = new Map();
    sts.forEach((b, id) => {
      const r = b.lastElementChild.getBoundingClientRect();
      lab.set(id, { w: r.width, h: r.height });
    });
    hds.forEach((n, id) => {
      const r = n.getBoundingClientRect();
      hd.set(id, { w: r.width, h: r.height });
    });
    const o = origin.getBoundingClientRect();
    return { lab, hd, origin: { w: o.width, h: o.height } };
  }

  function computeLayout(m, W, gaps, nTiers) {
    const per = Math.ceil(groups.length / nTiers);
    const tiers = [];
    for (let t = 0; t < nTiers; t++) tiers.push(groups.slice(t * per, (t + 1) * per));
    const ox = P.pad + P.hub;
    const gl = new Map();
    let right = 0;
    const tierInfo = tiers.map((tg, t) => {
      const gap = Array.isArray(gaps) ? gaps[t] : gaps;
      let tright = 0;
      const lanes = tg.filter((g, j) => j % 2 === 0).concat(tg.filter((g, j) => j % 2 === 1).reverse());
      const half = ((lanes.length - 1) / 2) * P.ls;
      const laneY = new Map(lanes.map((g, i) => [g.id, i * P.ls - half]));
      const vT = -half - P.side, vB = half + P.side;
      const hub = t ? Math.max(22, half + 10) : P.hub;
      let minY = -hub - 4, maxY = hub + 4;
      if (t === 0) minY = -hub - 8 - m.origin.h;
      const X = [];
      tg.forEach((g, j) => {
        const top = j % 2 === 0;
        const d = top ? laneY.get(g.id) - vT : vB - laneY.get(g.id);
        const colW = Math.max(m.hd.get(g.id).w - 12, ...g.items.map((f) => P.lab + m.lab.get(f.id).w + 9));
        const leftW = Math.max(0, ...g.items.map((f) => (f.all ? 0 : f.carriers.length * 13 + 9)));
        let x = Math.max(ox + Math.sqrt(hub * hub - half * half) + 10 + d, ox + 18 + leftW, t ? 0 : ox + m.origin.w / 2 + 12);
        if (j > 0) x = Math.max(x, X[j - 1].x + P.peel);
        if (j > 1) x = Math.max(x, X[j - 2].x + X[j - 2].colW + Math.max(gap, leftW + 6));
        X[j] = { x: Math.round(x), colW };
        tright = Math.max(tright, X[j].x + colW + P.pad);
        const s = top ? -1 : 1;
        let y = (top ? vT : vB) + s * P.first, prev = 0;
        const Y = new Map();
        g.items.forEach((f, k) => {
          const h = Math.max(18, m.lab.get(f.id).h);
          if (k) y += s * Math.max(P.pitch, (prev + h) / 2 + P.vgap);
          Y.set(f.id, y);
          prev = h;
        });
        const hh = Math.max(24, m.hd.get(g.id).h);
        const hy = y + s * (prev / 2 + P.hdGap + hh / 2);
        minY = Math.min(minY, hy - hh / 2);
        maxY = Math.max(maxY, hy + hh / 2);
        gl.set(g.id, { t, s, x: X[j].x, d, lane: laneY.get(g.id), v: top ? vT : vB, Y, hy });
      });
      right = Math.max(right, tright);
      const n = tg.length;
      return { hub, minY, maxY, right: tright, slots: Math.max(Math.ceil(n / 2), Math.floor(n / 2)) - 1 };
    });
    if (right > W) return null;
    let y = P.pad;
    for (const ti of tierInfo) {
      ti.cy = y - ti.minY;
      y = ti.cy + ti.maxY + 34;
    }
    return { gl, tiers: tierInfo, ox, right, height: y - 34 + P.pad };
  }

  function placeMap(W) {
    const { map, svg, hds, origin, probe } = view;
    const sig = probe.getBoundingClientRect().width;
    if (sig !== view.sig) {
      view.sig = sig;
      view.cache.clear();
    }
    let m = null, lay = null, pick = null, measured = false;
    for (const [t, w] of CANDIDATES) {
      m = view.cache.get(w);
      if (!m) {
        map.style.setProperty("--mfm-lw", w + "px");
        view.cache.set(w, (m = measure()));
        measured = true;
      }
      lay = computeLayout(m, W, P.gap, t);
      if (lay) {
        pick = [t, w];
        break;
      }
    }
    if (!lay) return false;
    const [tiers, lw] = pick;
    const gaps = lay.tiers.map((ti) => P.gap + Math.max(0, Math.min(tiers > 1 ? 150 : 48, (W - ti.right - 2) / (Math.max(1, ti.slots) + 0.25))));
    lay = computeLayout(m, W, gaps, tiers) || lay;
    const sx = Math.max(0, Math.floor((W - lay.right) / 2));
    const H = Math.ceil(lay.height);
    const key = [tiers, lw, sx, H, ...gaps.map(Math.round)].join();
    svg.setAttribute("width", W);
    // measure() parks every station at the origin, so a fresh measurement always needs a re-placement.
    if (key === view.key && !measured) return true;
    view.key = key;
    map.style.setProperty("--mfm-lw", lw + "px");
    map.dataset.layout = tiers + "x" + lw;
    const ox = lay.ox + sx;
    map.style.height = H + "px";
    svg.setAttribute("height", H);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.replaceChildren();
    lay.tiers.forEach((ti, t) => {
      if (t) svg.appendChild(svgEl("path", { class: "mfm-trunk", d: `M${ox} ${lay.tiers[t - 1].cy}V${ti.cy}` }));
    });
    const lines = new Map(), pos = new Map(), geo = new Map();
    for (const g of groups) {
      const L = lay.gl.get(g.id), cy = lay.tiers[L.t].cy;
      const x = L.x + sx, ly = L.lane + cy, vy = L.v + cy;
      const d = roundedPath([[ox, ly], [x - L.d, ly], [x, vy], [x, L.hy + cy]], P.bend);
      const set = [svgEl("path", { d, class: "mfm-ln", "data-p": g.slot })];
      if (g.open) set.push(svgEl("path", { d, class: "mfm-ln mfm-ln-in", "data-p": g.slot }));
      const cv = vy + L.s * 22;
      set.push(svgEl("path", { d: `M${x - 4.6} ${cv - L.s * 2.2}L${x} ${cv + L.s * 3}L${x + 4.6} ${cv - L.s * 2.2}Z`, class: "mfm-dir" + (g.open ? " is-open" : ""), "data-p": g.slot }));
      set.forEach((p) => svg.appendChild(p));
      lines.set(g.id, set);
      geo.set(g.id, { t: L.t, s: L.s, x, c: vy + L.s * P.lane });
      const hd = hds.get(g.id), hr = m.hd.get(g.id);
      hd.style.left = x - 12 + "px";
      hd.style.top = L.hy + cy - hr.h / 2 + "px";
      for (const f of g.items) {
        const b = sts.get(f.id), lr = m.lab.get(f.id);
        const bh = Math.max(24, lr.h + 6);
        const y = L.Y.get(f.id) + cy;
        b.style.left = x - 12 + "px";
        b.style.top = y - bh / 2 + "px";
        b.style.height = bh + "px";
        pos.set(f.id, { x, y });
      }
    }
    const arcs = svgEl("g", { class: "mfm-arcs" });
    svg.appendChild(arcs);
    for (const ti of lay.tiers) svg.appendChild(svgEl("circle", { class: "mfm-origin", cx: ox, cy: ti.cy, r: ti.hub }));
    origin.style.left = ox - m.origin.w / 2 + "px";
    origin.style.top = lay.tiers[0].cy - lay.tiers[0].hub - 6 - m.origin.h + "px";
    if (!introDone && !reduced.matches) {
      groups.forEach((g, i) => {
        for (const p of lines.get(g.id)) {
          p.style.setProperty("--len", Math.ceil(p.getTotalLength()));
          p.style.animationDelay = i * 45 + "ms";
        }
        hds.get(g.id).style.animationDelay = i * 45 + 260 + "ms";
        for (const f of g.items) sts.get(f.id).style.animationDelay = i * 45 + 220 + "ms";
      });
      map.classList.add("is-intro");
      setTimeout(() => map.classList.remove("is-intro"), 1600);
    }
    introDone = true;
    Object.assign(view, { lines, pos, geo, arcs, ox });
    return true;
  }

  function drawArcs(fid) {
    if (view.mode !== "map" || !view.arcs) return;
    view.arcs.replaceChildren();
    if (!fid) return;
    const f = byId.get(fid), a = view.pos.get(fid), ga = view.geo.get(f.group);
    let k = 0;
    for (const rid of f.rel) {
      const r = byId.get(rid), b = view.pos.get(rid), gb = view.geo.get(r.group);
      if (r.group === f.group || !b) continue;
      const pts = ga.t === gb.t
        ? [[a.x, a.y], [a.x, ga.s === gb.s ? ga.c : gb.c], [b.x, ga.s === gb.s ? ga.c : gb.c], [b.x, b.y]]
        : [[a.x, a.y], [a.x, ga.c], [view.ox, ga.c], [view.ox, gb.c], [b.x, gb.c], [b.x, b.y]];
      const p = svgEl("path", { d: roundedPath(pts, 9), class: "mfm-arc", "data-p": gById.get(f.group).slot });
      view.arcs.appendChild(p);
      p.style.setProperty("--len", Math.ceil(p.getTotalLength()));
      p.style.animationDelay = k++ * 40 + "ms";
    }
  }

  function renderStrip() {
    sts.clear();
    const wrap = el("div", "mfm-strip", `<h2 class="mfm-sr">Feature map</h2><details class="mfm-keyc"><summary><span>How to read the map</span><span class="mfm-keyc-k" aria-hidden="true"><i class="mfm-kc"></i>core <i class="mfm-kt"></i>advanced ${icon("el")}${icon("gas")}${icon("heat")}</span></summary>${legendHTML(true)}</details><p class="mfm-strip-hub" aria-hidden="true"><span></span>import monee</p>`);
    for (const g of groups) {
      const sec = el("section", "mfm-sg", `<div class="mfm-sg-hd">${num(g)}<div><h3 class="mfm-sg-t" id="mfm-sg-${esc(g.id)}" tabindex="-1">${esc(g.title)}</h3>${g.tagline ? `<p class="mfm-sg-tag">${md(g.tagline)}</p>` : ""}</div></div>`);
      sec.dataset.p = g.slot;
      sec.setAttribute("aria-labelledby", "mfm-sg-" + g.id);
      const ol = el("ol");
      for (const f of g.items) ol.appendChild(el("li")).appendChild(stationEl(f));
      sec.appendChild(ol);
      wrap.appendChild(sec);
    }
    stage.replaceChildren(wrap);
    view = { mode: "strip" };
  }

  // Short real words that would otherwise prefix unrelated ones ("wind" in "window").
  const WHOLE = new Set(["wind", "cap", "car"]);
  const hits = (h, t) => (t.length > 2 && !WHOLE.has(t) ? h.includes(" " + t) : h.includes(` ${t} `) || h.includes(` ${t}s `));

  function textMatcher(ts) {
    if (!ts.length) return { need: 0, ok: () => true, score: () => 0 };
    const res = new Map(ordered.map((f) => {
      const ws = ts.map((t) => Math.max(0, ...f.hays.filter(([, h]) => hits(h, t)).map(([w]) => w))).filter(Boolean);
      return [f.id, { n: ws.length, score: ws.reduce((a, b) => a + b, 0) }];
    }));
    const best = Math.max(0, ...[...res.values()].map((r) => r.n));
    return { need: best, partial: best > 0 && best < ts.length, ok: (f) => best > 0 && res.get(f.id).n === best, score: (f) => res.get(f.id).score };
  }

  function rankedMatches() {
    const tm = textMatcher(terms());
    return { tm, list: ordered.filter((f) => matchLevel(f, tm) === 2).sort((a, b) => tm.score(b) - tm.score(a)) };
  }

  function matchLevel(f, tm) {
    if (!tm.ok(f)) return 0;
    if (!state.carriers.size) return 2;
    if (f.all) return 1;
    return f.carriers.some((c) => state.carriers.has(c)) ? 2 : 0;
  }

  const terms = () => fold(state.q).split(" ").filter(Boolean);

  function capFeature(f) {
    const g = gById.get(f.group);
    const meta = [carrierIcons(f), f.rel.length ? `<span>${plural(f.rel.length, "related feature")}</span>` : ""].join("");
    return `<p class="mfm-cap-1">${num(g)}<span class="mfm-cap-g">${esc(g.title)}</span>${g.tagline ? `<span class="mfm-cap-gt">${md(g.tagline)}</span>` : ""}</p>
<p class="mfm-cap-2"><span class="mfm-cap-t">${esc(f.title)}</span>${f.tagline ? `<span class="mfm-cap-tag">${md(f.tagline)}</span>` : ""}${meta ? `<span class="mfm-cap-m">${meta}</span>` : ""}</p>`;
  }

  function capGroup(g) {
    const core = g.items.filter((f) => f.core).length;
    return `<p class="mfm-cap-1">${num(g)}<span class="mfm-cap-g">${esc(g.title)}</span><span class="mfm-cap-gt">${plural(g.items.length, "station")}${core ? `, ${core} core` : ""}</span></p>
<p class="mfm-cap-2"><span class="mfm-cap-tag">${g.tagline ? md(g.tagline) : "Read this line from the hub outward."}</span></p>`;
  }

  function capFilter(filt) {
    const hint = filt.first
      ? filt.text ? `Press Enter to open the first match, ${esc(filt.first.title)}.` : `First match: ${esc(filt.first.title)}.`
      : "Try other words, or the task filter below the map.";
    return `<p class="mfm-cap-1"><span class="mfm-cap-g">${esc(filt.msg)}</span></p>
<p class="mfm-cap-2"><span class="mfm-cap-tag">${hint}</span></p>`;
  }

  function paintCap(preview, filt) {
    let html, p = "";
    if (state.hoverG) {
      const g = gById.get(state.hoverG);
      html = capGroup(g);
      p = g.slot;
    } else if (preview || state.sel) {
      const f = byId.get(preview || state.sel);
      html = capFeature(f);
      p = gById.get(f.group).slot;
    } else if (filt) {
      html = capFilter(filt);
    } else {
      html = `<p class="mfm-cap-1"><span class="mfm-cap-g">${ordered.length} features on ${groups.length} lines</span><span class="mfm-cap-gt">Each line reads from the hub outward, core stations first.</span></p>
<p class="mfm-cap-2"><span class="mfm-cap-tag">Hover or focus a station to preview it here. Click it, or press Enter, to open its details below.</span></p>`;
    }
    if (capInfo.innerHTML !== html) capInfo.innerHTML = html;
    if (p === "") delete cap.dataset.p;
    else cap.dataset.p = p;
    cap.classList.toggle("is-sel", !!state.sel && !preview && !state.hoverG);
    const sel = state.sel && byId.get(state.sel);
    capX.hidden = !sel;
    if (sel) {
      const label = `${sel.title} details`;
      capGo.dataset.p = gById.get(sel.group).slot;
      if (capGo.getAttribute("aria-label") !== label) {
        capGo.setAttribute("aria-label", label);
        capGo.innerHTML = `<span>Details</span><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 3v10m-4-4 4 4 4-4"/></svg>`;
      }
    }
  }

  function paint() {
    const preview = (state.hover || state.kb) === state.sel ? null : state.hover || state.kb;
    const focusId = preview || state.sel;
    const hot = new Set();
    if (focusId) {
      hot.add(focusId);
      byId.get(focusId).rel.forEach((id) => hot.add(id));
    }
    const ts = terms();
    const { tm, list: matches } = rankedMatches();
    const filtering = ts.length > 0 || state.carriers.size > 0;
    const nMatch = filtering ? matches.length : 0;
    const nShared = filtering ? ordered.filter((f) => matchLevel(f, tm) === 1).length : 0;
    const first = (filtering && matches[0]) || null;
    const hotGroups = new Set([...hot].map((id) => byId.get(id).group));
    sts.forEach((b, id) => {
      const f = byId.get(id);
      let v = "on", match = false;
      if (filtering) {
        const lvl = matchLevel(f, tm);
        match = lvl === 2;
        v = lvl === 2 ? "on" : lvl === 1 ? "mid" : "off";
      }
      if (state.hoverG) v = f.group === state.hoverG ? "on" : "dim";
      else if (preview) v = hot.has(id) ? "on" : id === state.sel ? "mid" : "dim";
      else if (state.sel && hot.has(id)) v = "on";
      else if (state.sel && !filtering) v = "mid";
      b.dataset.v = v;
      b.classList.toggle("is-match", match && !preview);
      b.classList.toggle("is-hot", hot.has(id) && !state.hoverG);
      b.classList.toggle("is-sel", id === state.sel);
      b.classList.toggle("is-focus", id === focusId);
      if (id === state.sel) b.setAttribute("aria-current", "true");
      else b.removeAttribute("aria-current");
    });
    stage.classList.toggle("is-dim", Boolean(focusId || filtering || state.hoverG));
    const only = state.carriers.size === 1 ? CARRIERS.find((c) => state.carriers.has(c[0]))[2] : null;
    if (only) stage.style.setProperty("--mfm-hit", `var(--mfm-${only})`);
    else stage.style.removeProperty("--mfm-hit");
    let msg = "";
    if (filtering) {
      const q = state.q.trim();
      const names = CARRIERS.filter((c) => state.carriers.has(c[0])).map((c) => c[1].toLowerCase()).join(" or ");
      if (!nMatch && !nShared) msg = ts.length ? `No feature matches "${q}"` : "No matching feature";
      else if (state.carriers.size) msg = `${nMatch} of ${ordered.length} specific to ${names}${nShared ? `, ${nShared} for every carrier` : ""}`;
      else if (tm.partial) msg = `No feature matches every word of "${q}"; ${nMatch} match ${tm.need} of ${ts.length}`;
      else msg = ts.length ? `${nMatch} of ${ordered.length} features match "${q}"` : `${nMatch} of ${ordered.length} features match`;
    }
    if (view.mode === "map") {
      const fg = state.hoverG || (focusId && byId.get(focusId).group);
      view.lines.forEach((ps, gid) => ps.forEach((p) => p.classList.toggle("is-hot", gid === fg)));
      view.hds.forEach((n, gid) => n.classList.toggle("is-hot", state.hoverG ? gid === fg : hotGroups.has(gid)));
      drawArcs(state.hoverG ? null : focusId);
      paintCap(preview, filtering ? { msg, first, text: ts.length > 0 } : null);
    }
    if (statusT.textContent !== msg) {
      statusT.textContent = msg;
      if (filtering || view.filtered) say(msg || `All ${ordered.length} features shown`, 450);
    }
    view.filtered = filtering;
    view.matches = filtering ? matches : [];
    clearBtn.hidden = !filtering;
    toTasks.hidden = !(filtering && ts.length && !nMatch && tasksRoot && tasksRoot.querySelector(".mfm-tfilter:not([hidden])"));
    if (tasksRoot) tasksRoot.querySelectorAll(".mfm-chip[data-fid]").forEach((c) => c.classList.toggle("is-sel", c.dataset.fid === state.sel));
  }

  const PY_KW = new Set("and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split(" "));
  function highlightPy(src) {
    const re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)|([A-Za-z_]\w*)|([()[\]{},:;])|([^\sA-Za-z_\d#"'()[\]{},:;]+)/g;
    let out = "", last = 0, m;
    while ((m = re.exec(src))) {
      out += esc(src.slice(last, m.index));
      last = re.lastIndex;
      const t = m[0];
      let cls = "o";
      if (m[1]) cls = "c1";
      else if (m[2]) cls = "s2";
      else if (m[3]) cls = "mi";
      else if (m[4]) cls = t === "import" || t === "from" ? "kn" : PY_KW.has(t) ? "k" : ["None", "True", "False"].includes(t) ? "kc" : "n";
      else if (m[5]) cls = "p";
      out += `<span class="${cls}">${esc(t)}</span>`;
    }
    return out + esc(src.slice(last));
  }

  const chip = (f) => `<button type="button" class="mfm-chip" data-fid="${esc(f.id)}" data-p="${gById.get(f.group).slot}">${esc(f.title)}</button>`;
  const GOALS_SHOWN = 5;

  function featureHTML(f) {
    const g = gById.get(f.group);
    const [h1, h2] = view.mode === "strip" ? ["h4", "h5"] : ["h2", "h3"];
    const section = (title, body) => (body ? `<section><${h2} class="mfm-p-h">${title}</${h2}>${body}</section>` : "");
    const carriers = f.all
      ? `<span class="mfm-badge">${icon("el")}${icon("gas")}${icon("heat")}Every carrier</span>`
      : CARRIERS.filter((c) => f.carriers.includes(c[0])).map((c) => `<span class="mfm-badge">${icon(c[2])}${c[1]}</span>`).join("");
    const byKind = new Map();
    for (const d of f.docs) byKind.set(d.kind, (byKind.get(d.kind) || []).concat(d));
    const docs = [...byKind].map(([k, ds]) => `<p class="mfm-dk">${esc(k[0].toUpperCase() + k.slice(1))}</p><ul>${ds.map((d) => `<li><a href="${esc(d.url)}">${esc(d.label)}</a></li>`).join("")}</ul>`).join("");
    const goal = (t) => `<li><a href="#${esc(t.id)}" data-task="${esc(t.id)}">${md(t.want)}</a></li>`;
    const more = f.tasks.slice(GOALS_SHOWN);
    const goals = f.tasks.length
      ? `<ul class="mfm-tasks-of">${f.tasks.slice(0, GOALS_SHOWN).map(goal).join("")}</ul>${more.length ? `<details class="mfm-more"><summary>Show ${plural(more.length, "more goal")}</summary><ul class="mfm-tasks-of">${more.map(goal).join("")}</ul></details>` : ""}`
      : "";
    return `<div class="mfm-p-head">
<p class="mfm-p-meta">${num(g)}<span class="mfm-p-g">${esc(g.title)}</span>${g.tagline ? `<span class="mfm-p-gt">${md(g.tagline)}</span>` : ""}</p>
<div class="mfm-p-trow"><${h1} class="mfm-p-title" tabindex="-1">${esc(f.title)}</${h1}><a class="mfm-p-link" href="#feature-${esc(f.id)}" aria-label="Copy a link to ${esc(f.title)}" title="Copy link"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6.6 9.4a2.6 2.6 0 0 0 3.7 0l2.4-2.4a2.6 2.6 0 0 0-3.7-3.7l-.9.9M9.4 6.6a2.6 2.6 0 0 0-3.7 0L3.3 9a2.6 2.6 0 0 0 3.7 3.7l.9-.9"/></svg></a></div>
${f.tagline ? `<p class="mfm-p-tag">${md(f.tagline)}</p>` : ""}
<div class="mfm-p-car"><span class="mfm-p-lvl">${f.core ? "Core feature" : "Advanced"}</span>${carriers}</div>
<button type="button" class="mfm-p-close" aria-label="Close details"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8m0-8-8 8"/></svg></button>
</div>
<div class="mfm-p-body"><div class="mfm-p-main">
${f.details ? `<p>${md(f.details)}</p>` : ""}
${f.highlights.length ? `<ul class="mfm-p-hl">${f.highlights.map((h) => `<li>${md(h)}</li>`).join("")}</ul>` : ""}
${f.snippet ? `<div class="highlight"><pre><code>${highlightPy(f.snippet)}</code></pre></div>` : ""}
${f.api.length ? `<section class="mfm-p-apis"><${h2} class="mfm-p-h">API</${h2}><ul class="mfm-api">${f.api.map((a) => `<li><code>${esc(a)}</code></li>`).join("")}</ul></section>` : ""}
</div><div class="mfm-p-side">
${section("Read more", docs && `<div class="mfm-docs">${docs}</div>`)}
${section("Goals it serves", goals)}
${section("Related features", f.rel.length && `<div class="mfm-p-rel">${f.rel.map((id) => chip(byId.get(id))).join("")}</div>`)}
</div></div>
<p class="mfm-p-foot"><button type="button" class="mfm-back"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 13V3M4 7l4-4 4 4"/></svg>Back to the map</button></p>`;
  }

  function legendHTML(compact) {
    const g1 = groups[0] ? groups[0].slot : 1;
    const line = `<svg viewBox="0 0 46 22"><path class="mfm-key-ln" d="M3 17h12l9-11h19" data-p="${g1}"/><path class="mfm-dir" d="M31.5 1.6 37 6l-5.5 4.4z" data-p="${g1}"/></svg>`;
    const core = `<svg viewBox="0 0 46 22"><path class="mfm-key-ln" d="M23 -2v26" data-p="${g1}"/><circle cx="23" cy="11" r="6" class="mfm-key-core"/></svg>`;
    const tick = `<svg viewBox="0 0 46 22"><path class="mfm-key-ln" d="M23 -2v26" data-p="${g1}"/><rect x="23" y="8.75" width="9" height="4.5" class="mfm-key-tick" data-p="${g1}"/></svg>`;
    const arc = `<svg viewBox="0 0 46 22"><path class="mfm-key-arc" d="M6 17V8q0-4 4-4h26q4 0 4 4v9" data-p="${g1}"/><circle cx="6" cy="17" r="3.5" class="mfm-key-core"/><circle cx="40" cy="17" r="3.5" class="mfm-key-core"/></svg>`;
    const items = [
      [line, `Each line is one step of a typical study, numbered 1 to ${groups.length} in workflow order. Read every line from the hub outward, as the arrows show.`],
      [core, "Core feature, a good first stop. Core stations sit nearest the hub."],
      [tick, "Advanced feature for specific studies, further out on the line."],
      [`${icon("el")}${icon("gas")}${icon("heat")}`, "Specific to electricity, gas or heat. Stations without an icon work on every carrier."],
    ];
    if (groups.some((g) => g.open)) items.push([`<svg viewBox="0 0 46 22"><path class="mfm-key-ln" d="M3 11h40" data-p="${(groups.find((g) => g.open) || {}).slot}"/><path class="mfm-key-ln mfm-ln-in" d="M3 11h40"/></svg>`, "A hollow line is where your own code plugs in."]);
    if (!compact) items.push([arc, "Hover or focus a station to trace its related features. Click it, or press Enter, to read the details here."]);
    return `<ul class="mfm-legend">${items.map(([k, t]) => `<li><span class="mfm-key">${k}</span><span>${t}</span></li>`).join("")}</ul>${compact ? `<p class="mfm-hint">Tap a station to open its details, documentation and related features.</p>` : ""}`;
  }

  function introHTML() {
    return `<div class="mfm-intro"><div>
<h2 class="mfm-p-title" tabindex="-1">How to read the map</h2>${legendHTML(false)}
<p class="mfm-hint">Keyboard: Tab into the map, then use the arrow keys to move between stations and Home or End to jump to the ends of a line.</p></div>
<div class="mfm-intro-start"><p class="mfm-stats"><span><span class="mfm-stat">${ordered.length}</span> features</span><span><span class="mfm-stat">${nCore}</span> core</span>${tasks.length && tasksSection ? `<a href="#${esc(tasksSection.id)}"><span class="mfm-stat">${tasks.length}</span> goals in the task table</a>` : ""}</p>
<h3 class="mfm-p-h">Start here</h3><p>The first core station of each line.</p><div class="mfm-p-rel">${starters.map(chip).join("")}</div></div></div>`;
  }

  function renderPanel() {
    const f = state.sel && byId.get(state.sel);
    if (f) {
      panel.dataset.p = gById.get(f.group).slot;
      panel.innerHTML = featureHTML(f);
    } else {
      delete panel.dataset.p;
      panel.innerHTML = view.mode === "strip" ? "" : introHTML();
    }
  }

  function placePanel() {
    const slot = stage.querySelector(".mfm-slot");
    panel.hidden = view.mode === "strip" && !state.sel;
    if (view.mode === "strip" && state.sel && sts.get(state.sel)) {
      const li = sts.get(state.sel).closest("li");
      const target = slot || el("li", "mfm-slot");
      target.appendChild(panel);
      if (li.nextElementSibling !== target) li.after(target);
      return;
    }
    if (slot) slot.remove();
    if (panel.previousElementSibling !== stage.parentNode) stage.parentNode.after(panel);
  }

  function setRove(id) {
    if (!sts.has(id)) id = state.sel && sts.has(state.sel) ? state.sel : ordered[0] && ordered[0].id;
    state.rove = id;
    sts.forEach((b, k) => (b.tabIndex = k === id ? 0 : -1));
  }

  const MARK = { mfm: 1 };
  function setHash(hash, push) {
    const url = hash || location.pathname + location.search;
    if (push) {
      history.replaceState(MARK, "", location.href);
      history.pushState(MARK, "", url);
    } else history.replaceState(MARK, "", url);
  }

  function select(id, opt = {}) {
    const f = byId.get(id);
    if (!f) return;
    state.sel = id;
    if (opt.hash !== false) setHash("#feature-" + id, opt.push);
    renderPanel();
    placePanel();
    setRove(id);
    paint();
    say(`${f.title} selected. ${view.mode === "strip" ? "Details follow the station." : "Details are below the map."}`);
    const b = sts.get(id);
    if (opt.focus && b) b.focus({ preventScroll: true });
    if (opt.scroll && b) bringIntoView(b, opt.instant);
    if (opt.focusPanel) panel.querySelector(".mfm-p-title").focus({ preventScroll: true });
  }

  function bringIntoView(b, instant) {
    lockHover();
    const vh = window.innerHeight, y0 = window.scrollY;
    const top = (n) => n.getBoundingClientRect().top + y0;
    const r = b.getBoundingClientRect();
    let y = y0 + r.top + r.height / 2 - vh / 2;
    if (view.mode === "map") {
      const frame = stage.parentNode, fr = frame.getBoundingClientRect();
      if (fr.height + 24 <= vh) y = top(frame) - 12;
    } else {
      y = y0 + r.top - 110;
    }
    window.scrollTo({ top: Math.max(0, y), behavior: instant ? "instant" : behavior() });
  }

  function viewport() {
    let top = 8, bottom = 8;
    const hdr = document.querySelector(".mobile-header");
    if (hdr && /sticky|fixed/.test(getComputedStyle(hdr).position)) top += Math.max(0, hdr.getBoundingClientRect().bottom);
    if (view.mode === "map") bottom += Math.max(0, window.innerHeight - cap.getBoundingClientRect().top);
    return { top, bottom: window.innerHeight - bottom };
  }

  const shift = (a, b, vp) => (a < vp.top ? a - vp.top : b > vp.bottom ? Math.min(b - vp.bottom, a - vp.top) : 0);

  function scrollByLocked(dy, smooth) {
    if (!dy) return;
    // The scroll moves content under a resting pointer, and the pointerover that follows is not the user's.
    lockHover();
    window.scrollBy({ top: dy, behavior: smooth ? behavior() : "instant" });
  }

  function ensureVisible(n, smooth) {
    const r = n.getBoundingClientRect();
    scrollByLocked(shift(r.top, r.bottom, viewport()), smooth);
  }

  function revealMatches() {
    const ms = view.mode === "map" && terms().length ? view.matches.map((f) => sts.get(f.id)).filter(Boolean) : [];
    if (!ms.length) return;
    const vp = viewport();
    const rs = ms.map((b) => b.getBoundingClientRect());
    const lo = Math.min(...rs.map((r) => r.top)), hi = Math.max(...rs.map((r) => r.bottom));
    const keepBar = Math.max(0, input.getBoundingClientRect().top - vp.top);
    const all = hi - lo <= vp.bottom - vp.top ? shift(lo, hi, vp) : null;
    const best = shift(rs[0].top, rs[0].bottom, vp);
    const dy = [all, best].find((d) => d !== null && d <= keepBar) ?? keepBar;
    scrollByLocked(dy, true);
  }

  function showCard() {
    lockHover();
    panel.scrollIntoView({ block: "start", behavior: behavior() });
    const t = panel.querySelector(".mfm-p-title");
    if (t) t.focus({ preventScroll: true });
  }

  function clearSelection(refocus, reveal = refocus) {
    const prev = state.sel;
    if (!prev) return;
    state.sel = null;
    if (/^#feature-/.test(location.hash)) setHash("", false);
    renderPanel();
    placePanel();
    setRove(prev);
    paint();
    say("Selection cleared.");
    const b = refocus && sts.get(prev);
    if (!b) return;
    b.focus({ preventScroll: true });
    if (!reveal) return;
    const r = b.getBoundingClientRect(), vp = viewport();
    if (r.top < vp.top || r.bottom > vp.bottom) bringIntoView(b);
  }

  function hashTarget() {
    const m = /^#feature-(.+)$/.exec(location.hash);
    if (!m) return null;
    let id;
    try {
      id = decodeURIComponent(m[1]);
    } catch {
      return null;
    }
    return byId.has(id) ? id : null;
  }

  function nearest(b, dx, dy) {
    const center = (n) => {
      const r = n.getBoundingClientRect();
      return [r.left + 12, r.top + r.height / 2];
    };
    const [x0, y0] = center(b);
    let best = null, score = Infinity;
    sts.forEach((n) => {
      if (n === b) return;
      const [x, y] = center(n);
      const along = (x - x0) * dx + (y - y0) * dy;
      if (along < 4) return;
      const s = along + Math.abs((x - x0) * dy - (y - y0) * dx) * 2.5;
      if (s < score) {
        score = s;
        best = n;
      }
    });
    return best;
  }

  const content = root.closest(".content");
  function computeBleed() {
    const targets = [root, tasksRoot, content].filter(Boolean);
    targets.forEach((n) => n.style.setProperty("--mfm-bleed", "0px"));
    const r = root.getBoundingClientRect();
    let limit = document.documentElement.clientWidth;
    const toc = document.querySelector(".toc-drawer");
    if (toc && toc.offsetParent && getComputedStyle(toc).position !== "fixed") {
      const tr = toc.getBoundingClientRect();
      if (tr.width && tr.left >= r.right - 1) limit = tr.left;
    }
    let extra = Math.floor(Math.min(limit - r.right - 48, 1160 - r.width));
    if (extra < 120) extra = 0;
    targets.forEach((n) => n.style.setProperty("--mfm-bleed", extra + "px"));
  }

  const FOCUSABLE = "a[href], button, summary, [tabindex]";
  let observedProbe = null, probeW = 0;
  const probeObserver = window.ResizeObserver && new ResizeObserver(() => {
    const w = observedProbe && observedProbe.getBoundingClientRect().width;
    if (!w || w === probeW) return;
    const first = !probeW;
    probeW = w;
    if (!first) requestAnimationFrame(() => layout(true));
  });

  let lastW = 0;
  function layout(force) {
    computeBleed();
    const W = stage.clientWidth;
    if (!force && W === lastW) return;
    lastW = W;
    const active = document.activeElement && document.activeElement.closest ? document.activeElement.closest(".mfm-st") : null;
    const keep = active && active.dataset.id;
    const panelFocus = panel.contains(document.activeElement) ? [...panel.querySelectorAll(FOCUSABLE)].indexOf(document.activeElement) : -1;
    ui.classList.toggle("mfm-narrow", W < 700);
    const prevMode = view.mode;
    let ok = false;
    if (W >= 600) {
      if (view.mode !== "map") buildMap();
      ok = placeMap(W);
    }
    if (!ok && view.mode !== "strip") renderStrip();
    ui.classList.toggle("mfm-is-strip", view.mode === "strip");
    if (prevMode !== view.mode) {
      renderPanel();
      setRove(state.rove);
    }
    placePanel();
    paint();
    if (keep && sts.get(keep) && document.activeElement !== sts.get(keep)) sts.get(keep).focus({ preventScroll: true });
    if (panelFocus !== -1 && !panel.hidden && !panel.contains(document.activeElement)) {
      const target = panel.querySelectorAll(FOCUSABLE)[panelFocus] || panel.querySelector(".mfm-p-title");
      if (target) target.focus({ preventScroll: true });
    }
    if (view.probe && view.probe !== observedProbe && probeObserver) {
      probeObserver.disconnect();
      probeObserver.observe((observedProbe = view.probe));
    }
  }

  stage.addEventListener("click", (e) => {
    const b = e.target.closest(".mfm-st");
    if (b) select(b.dataset.id);
  });
  const onPointer = (e) => {
    if (e.pointerType === "touch") return;
    if (hoverLock) {
      if (e.screenX === hoverLock[0] && e.screenY === hoverLock[1]) return;
      hoverLock = null;
      stage.classList.remove("is-hover-locked");
    }
    const b = e.target.closest(".mfm-st");
    const h = !b && e.target.closest(".mfm-hd");
    const id = b ? b.dataset.id : null, gid = h ? h.dataset.g : null;
    if (id !== state.hover || gid !== state.hoverG) {
      state.hover = id;
      state.hoverG = gid;
      paint();
    }
  };
  stage.addEventListener("pointerover", onPointer);
  stage.addEventListener("pointermove", onPointer);
  stage.addEventListener("pointerleave", () => {
    if (state.hover || state.hoverG) {
      state.hover = state.hoverG = null;
      paint();
    }
  });
  stage.addEventListener("focusin", (e) => {
    const b = e.target.closest(".mfm-st");
    if (!b) return;
    setRove(b.dataset.id);
    if (b.matches(":focus-visible")) {
      state.kb = b.dataset.id;
      state.hover = state.hoverG = null;
      paint();
    }
  });
  stage.addEventListener("focusout", () => {
    if (state.kb) {
      state.kb = null;
      paint();
    }
  });
  stage.addEventListener("keydown", (e) => {
    const b = e.target.closest(".mfm-st");
    if (!b) return;
    const f = byId.get(b.dataset.id);
    const dir = { ArrowUp: [0, -1], ArrowDown: [0, 1], ArrowLeft: [-1, 0], ArrowRight: [1, 0] }[e.key];
    let next;
    if (dir && view.mode === "map") next = nearest(b, dir[0], dir[1]);
    else if (dir) next = dir[1] && sts.get((ordered[ordered.indexOf(f) + dir[1]] || {}).id);
    else if (e.key === "Home" || e.key === "End") {
      const items = gById.get(f.group).items;
      next = sts.get(items[e.key === "Home" ? 0 : items.length - 1].id);
    } else return;
    e.preventDefault();
    if (!next) return;
    next.focus({ preventScroll: true });
    ensureVisible(next);
  });
  capGo.addEventListener("click", showCard);
  // detail is 0 for keyboard-activated clicks; only those should scroll back to the station.
  $(".mfm-cap-clr").addEventListener("click", (e) => clearSelection(true, e.detail === 0));
  $(".mfm-jump").addEventListener("click", (e) => {
    const a = e.target.closest("a[data-g]");
    if (!a) return;
    e.preventDefault();
    const h = document.getElementById("mfm-sg-" + a.dataset.g);
    if (!h) return;
    h.closest(".mfm-sg").scrollIntoView({ block: "start", behavior: behavior() });
    h.focus({ preventScroll: true });
  });

  function copyLink(a) {
    const url = new URL(a.getAttribute("href"), location.href).href;
    const done = () => {
      a.classList.add("is-done");
      setTimeout(() => a.classList.remove("is-done"), 1600);
      say("Link copied.");
    };
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(url).then(done, () => say("Copy failed. The address bar holds the link."));
    else say("The address bar holds the link to this feature.");
  }

  function goToTask(id) {
    const row = tasksRoot && document.getElementById(id);
    if (!row) return false;
    if (row.closest("[hidden]") && tasksRoot.resetFilter) tasksRoot.resetFilter();
    setHash("#" + id, true);
    lockHover();
    row.scrollIntoView({ block: "center", behavior: behavior() });
    row.classList.remove("is-flash");
    void row.offsetWidth;
    row.classList.add("is-flash");
    const a = row.querySelector("a[href]");
    if (a) a.focus({ preventScroll: true });
    say(`Goal: ${plain(tasks.find((t) => t.id === id).want)}`);
    return true;
  }

  panel.addEventListener("click", (e) => {
    const c = e.target.closest(".mfm-chip[data-fid]");
    if (c) return select(c.dataset.fid, { focusPanel: true });
    if (e.target.closest(".mfm-p-close")) return clearSelection(true, e.detail === 0);
    const back = e.target.closest(".mfm-back");
    if (back && sts.get(state.sel)) {
      const b = sts.get(state.sel);
      bringIntoView(b);
      b.focus({ preventScroll: true });
      return;
    }
    const plainClick = !(e.button || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey);
    const link = e.target.closest(".mfm-p-link");
    if (link && plainClick) {
      e.preventDefault();
      return copyLink(link);
    }
    const goal = e.target.closest("a[data-task]");
    if (goal && plainClick && goToTask(goal.dataset.task)) e.preventDefault();
  });
  let revealTimer = 0;
  input.addEventListener("input", () => {
    state.q = input.value;
    paint();
    clearTimeout(revealTimer);
    revealTimer = setTimeout(revealMatches, 250);
  });
  toTasks.addEventListener("click", () => {
    const tin = tasksRoot && tasksRoot.querySelector(".mfm-tfilter input");
    if (!tin) return;
    tin.value = input.value;
    tin.dispatchEvent(new Event("input"));
    lockHover();
    tin.closest(".mfm-tfilter").scrollIntoView({ block: "start", behavior: behavior() });
    tin.focus({ preventScroll: true });
  });
  input.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    e.preventDefault();
    const first = rankedMatches().list[0];
    if (first && (terms().length || state.carriers.size)) select(first.id, { focus: true, scroll: true });
  });
  ui.querySelectorAll(".mfm-cbtn").forEach((btn) =>
    btn.addEventListener("click", () => {
      const c = btn.dataset.c;
      if (state.carriers.has(c)) state.carriers.delete(c);
      else state.carriers.add(c);
      btn.setAttribute("aria-pressed", String(state.carriers.has(c)));
      paint();
    }),
  );
  clearBtn.addEventListener("click", () => {
    input.value = "";
    state.q = "";
    state.carriers.clear();
    ui.querySelectorAll(".mfm-cbtn").forEach((b) => b.setAttribute("aria-pressed", "false"));
    paint();
    input.focus();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || e.defaultPrevented) return;
    const a = document.activeElement;
    if (a && a !== document.body && !ui.contains(a)) return;
    if (a === input && (input.value || state.carriers.size)) return clearBtn.click();
    if (state.sel) clearSelection(ui.contains(a));
  });
  window.addEventListener("hashchange", () => {
    const id = hashTarget();
    // On entries this script created the change is a Back or Forward, and the browser restores the scroll.
    if (id) select(id, { scroll: !(history.state && history.state.mfm), hash: false });
  });
  let raf = 0;
  const onResize = () => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => layout(false));
  };
  window.addEventListener("resize", onResize);
  if (window.ResizeObserver) new ResizeObserver(onResize).observe(root);
  if (document.fonts && document.fonts.addEventListener) document.fonts.addEventListener("loadingdone", () => layout(true));

  if (tasksRoot) {
    const bar = tasksRoot.querySelector(".mfm-tfilter");
    const tin = bar && bar.querySelector("input");
    const tcount = bar && bar.querySelector(".mfm-tcount");
    const table = tasksRoot.querySelector("table");
    const rows = [...tasksRoot.querySelectorAll("tbody tr:not(.mfm-cat)")].map((tr) => ({ tr, text: tr.textContent.toLowerCase() }));
    const empty = el("tbody", "mfm-empty", '<tr><td colspan="3">No task matches this filter.</td></tr>');
    empty.hidden = true;
    if (table) table.appendChild(empty);
    if (bar && tin) {
      bar.hidden = false;
      const run = () => {
        const ts = tin.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
        let n = 0;
        for (const r of rows) {
          const ok = ts.every((t) => r.text.includes(t));
          r.tr.hidden = !ok;
          if (ok) n++;
        }
        tasksRoot.querySelectorAll("tbody:not(.mfm-empty)").forEach((tb) => {
          const all = tb.querySelectorAll("tr:not(.mfm-cat)").length;
          const shown = tb.querySelectorAll("tr:not(.mfm-cat):not([hidden])").length;
          tb.hidden = !shown;
          const cnt = tb.querySelector(".mfm-cat-c");
          if (cnt) cnt.textContent = ts.length ? `${shown} of ${all}` : String(all);
        });
        empty.hidden = n > 0;
        const msg = ts.length ? `${n} of ${rows.length} tasks` : "";
        tcount.textContent = msg;
        say(msg || `All ${rows.length} tasks shown`, 450);
      };
      tasksRoot.resetFilter = () => {
        tin.value = "";
        run();
      };
      tin.addEventListener("input", run);
      tin.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && tin.value) {
          tasksRoot.resetFilter();
          e.preventDefault();
        }
      });
    }
    tasksRoot.addEventListener("click", (e) => {
      const a = e.target.closest(".mfm-chip[data-fid]");
      if (!a || e.button || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;
      e.preventDefault();
      select(a.dataset.fid, { focus: true, scroll: true, push: true });
    });
  }

  layout(true);
  setRove(null);
  const initial = hashTarget();
  const nav = performance.getEntriesByType ? performance.getEntriesByType("navigation")[0] : null;
  if (initial) {
    select(initial, { hash: false });
    if (nav && nav.type === "back_forward") return;
    const settle = () =>
      requestAnimationFrame(() => {
        const b = sts.get(initial);
        if (!b) return;
        if (document.activeElement === b) b.blur();
        bringIntoView(b, true);
      });
    if (document.readyState === "complete") settle();
    else window.addEventListener("load", settle, { once: true });
  }
})();
