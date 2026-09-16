/* ==========================================================================
   AgentX - demo frontend logic
   Vanilla JS: calls the FastAPI backend and renders results + SVG charts.
   ========================================================================== */

"use strict";

const $ = (id) => document.getElementById(id);

/** POST JSON to the backend and return parsed data. */
async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${err.slice(0, 300)}`);
  }
  return res.json();
}

/** Render an object as pretty JSON inside a <pre>. */
function renderJson(el, data) {
  el.textContent = JSON.stringify(data, null, 2);
}

/** Parse "(x,y),(x,y),..." into [[x, y], ...]. */
function parsePoints(text) {
  const matches = [...text.matchAll(/\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?/g)];
  const pts = matches.map((m) => [parseFloat(m[1]), parseFloat(m[2])]);
  if (pts.length < 2) throw new Error("Cannot parse coordinates, use (x,y),(x,y) format");
  return pts;
}

/** Normalize points to an SVG viewBox and return drawing helpers. */
function fit(points, W, H, pad = 36) {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = Math.max(maxX - minX, 1e-6);
  const spanY = Math.max(maxY - minY, 1e-6);
  const s = Math.min((W - 2 * pad) / spanX, (H - 2 * pad) / spanY);
  const ox = (W - spanX * s) / 2, oy = (H - spanY * s) / 2;
  return {
    W, H,
    x: (px) => ox + (px - minX) * s,
    y: (py) => oy + (maxY - py) * s, // flip Y for screen coords
  };
}

function svgOpen(W, H) {
  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
}

/* ==========================================================================
   01 - Text classification
   ========================================================================== */
async function runText() {
  const out = $("text-out");
  const btn = event.target;
  btn.disabled = true;
  try {
    const data = await api("/api/text/classify", { texts: [$("text-input").value] });
    const r = data.results[0];
    const scores = Object.entries(r.scores)
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`)
      .join("  |  ");
    renderJson(out, { label: r.label, confidence: r.confidence, scores });
  } catch (e) {
    out.textContent = "閿欒: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

/* ==========================================================================
   02 - NLP phase detection
   ========================================================================== */
async function runNlp() {
  const out = $("nlp-out");
  const btn = event.target;
  btn.disabled = true;
  try {
    const data = await api("/api/nlp/analyze", { text: $("nlp-input").value, language: "auto" });
    renderJson(out, data);
  } catch (e) {
    out.textContent = "閿欒: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

/* ==========================================================================
   03 - Probability calibration
   ========================================================================== */
async function runCalibrate() {
  const out = $("cal-out");
  const btn = event.target;
  btn.disabled = true;
  try {
    const data = await api("/api/calibrate", { method: $("cal-method").value, n_samples: 800 });
    renderJson(out, data);
  } catch (e) {
    out.textContent = "閿欒: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

/* ==========================================================================
   04 - Scheduling + Gantt
   ========================================================================== */
function parseJobs(text) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, i) => {
      const parts = line.split(",").map((s) => s.trim());
      if (parts.length < 2) throw new Error(`绗?${i + 1} 琛屾牸寮忛敊璇痐);
      const deps = parts[3] ? parts[3].split(/[;|]/).map((s) => s.trim()).filter(Boolean) : [];
      return {
        job_id: parts[0],
        duration: parseFloat(parts[1]),
        priority: parts[2] ? parseInt(parts[2], 10) : 0,
        dependencies: deps,
        weight: parts[4] ? parseFloat(parts[4]) : 1.0,
        resources: 1.0,
      };
    });
}

function drawGantt(entries, W = 620, H = 240) {
  const maxEnd = Math.max(...entries.map((e) => e.end), 1);
  const rowH = 34, padTop = 24, padLeft = 70, padRight = 20;
  // Assign a row per machine in first-seen order.
  const rowOf = {};
  entries.forEach((e) => {
    const key = `machine_${e.machine}`;
    if (!(key in rowOf)) rowOf[key] = Object.keys(rowOf).length;
  });
  const rows = Math.max(1, Object.keys(rowOf).length);
  const H2 = padTop + rows * rowH + 20;
  const x = (t) => padLeft + (t / maxEnd) * (W - padLeft - padRight);
  const palette = ["#2563eb", "#0ea5e9", "#8b5cf6", "#16a34a", "#f59e0b", "#ef4444"];

  let svg = svgOpen(W, H2);
  svg += `<line x1="${padLeft}" y1="${padTop - 10}" x2="${W - padRight}" y2="${padTop - 10}" class="axis"/>`;
  Object.entries(rowOf).forEach(([key, ri]) => {
    const y = padTop + ri * rowH;
    svg += `<text x="${padLeft - 8}" y="${y + 20}" text-anchor="end">${key}</text>`;
  });
  entries.forEach((e) => {
    const ri = rowOf[`machine_${e.machine}`];
    const y = padTop + ri * rowH;
    const color = palette[ri % palette.length];
    svg += `<rect x="${x(e.start)}" y="${y + 6}" width="${Math.max(x(e.end) - x(e.start), 6)}" height="22" rx="4" fill="${color}" opacity="0.85"/>`;
    svg += `<text x="${x(e.start) + 6}" y="${y + 22}" fill="#fff" font-size="11">${e.job_id}</text>`;
  });
  svg += `</svg>`;
  return svg;
}

async function runSchedule() {
  const out = $("sched-out");
  const viz = $("sched-gantt");
  const btn = event.target;
  btn.disabled = true;
  try {
    const jobs = parseJobs($("sched-input").value);
    const machines = parseInt($("sched-machines").value, 10) || 2;
    const data = await api("/api/schedule", { machines, jobs });
    viz.innerHTML = drawGantt(data.entries);
    renderJson(out, {
      makespan: data.makespan,
      machine_utilization: data.machine_utilization,
      total_work: data.total_work,
    });
  } catch (e) {
    out.textContent = "閿欒: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

/* ==========================================================================
   05 - TSP path optimization
   ========================================================================== */
function drawTsp(points, route, W = 620, H = 300) {
  const v = fit(points, W, H);
  const byId = new Map(points.map((p, i) => [i, p]));
  let svg = svgOpen(W, H);
  svg += `<line x1="20" y1="${H - 20}" x2="${W - 20}" y2="${H - 20}" class="axis"/>`;
  svg += `<line x1="20" y1="20" x2="20" y2="${H - 20}" class="axis"/>`;
  // Route polyline
  const ptsStr = route.map((id) => `${v.x(byId.get(id)[0])},${v.y(byId.get(id)[1])}`).join(" ");
  svg += `<polyline points="${ptsStr}" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linejoin="round"/>`;
  points.forEach((p, i) => {
    const cx = v.x(p[0]), cy = v.y(p[1]);
    const depot = i === route[0];
    svg += `<circle cx="${cx}" cy="${cy}" r="${depot ? 7 : 5}" fill="${depot ? "#ef4444" : "#0ea5e9"}" stroke="#fff" stroke-width="1.5"/>`;
    svg += `<text x="${cx + 8}" y="${cy - 6}">${i}</text>`;
  });
  svg += `</svg>`;
  return svg;
}

async function runTsp() {
  const out = $("tsp-out");
  const viz = $("tsp-viz");
  const btn = event.target;
  btn.disabled = true;
  try {
    const coords = parsePoints($("tsp-input").value);
    const data = await api("/api/optimize/tsp", { coords });
    viz.innerHTML = drawTsp(coords, data.route);
    renderJson(out, { route: data.route, distance: data.distance });
  } catch (e) {
    out.textContent = "閿欒: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

/* ==========================================================================
   06 - Convex hull
   ========================================================================== */
function drawHull(points, hull, W = 620, H = 300) {
  const v = fit(points, W, H);
  let svg = svgOpen(W, H);
  svg += `<line x1="20" y1="${H - 20}" x2="${W - 20}" y2="${H - 20}" class="axis"/>`;
  svg += `<line x1="20" y1="20" x2="20" y2="${H - 20}" class="axis"/>`;
  if (hull.length >= 3) {
    const ptsStr = [...hull.map((p) => `${v.x(p[0])},${v.y(p[1])}`), `${v.x(hull[0][0])},${v.y(hull[0][1])}`].join(" ");
    svg += `<polygon points="${ptsStr}" fill="#2563eb" opacity="0.18" stroke="#2563eb" stroke-width="2.5" stroke-linejoin="round"/>`;
  }
  points.forEach((p) => {
    svg += `<circle cx="${v.x(p[0])}" cy="${v.y(p[1])}" r="4.5" fill="#0ea5e9" stroke="#fff" stroke-width="1.5"/>`;
  });
  svg += `</svg>`;
  return svg;
}

async function runHull() {
  const out = $("hull-out");
  const viz = $("hull-viz");
  const btn = event.target;
  btn.disabled = true;
  try {
    const points = parsePoints($("hull-input").value);
    const data = await api("/api/geometry/hull", { points });
    viz.innerHTML = drawHull(points, data.hull);
    renderJson(out, { hull: data.hull, area: data.area, perimeter: data.perimeter });
  } catch (e) {
    out.textContent = "閿欒: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

/* ==========================================================================
   07 - Collision detection
   ========================================================================== */
function drawCollision(polyA, polyB, collide, W = 620, H = 300) {
  const all = [...polyA, ...polyB];
  const v = fit(all, W, H);
  const poly = (pts, color, opacity) => {
    const ptsStr = pts.map((p) => `${v.x(p[0])},${v.y(p[1])}`).join(" ");
    return `<polygon points="${ptsStr}" fill="${color}" opacity="${opacity}" stroke="${color}" stroke-width="2"/>`;
  };
  const color = collide ? "#ef4444" : "#16a34a";
  let svg = svgOpen(W, H);
  svg += poly(polyA, "#2563eb", 0.35);
  svg += poly(polyB, color, 0.35);
  svg += `</svg>`;
  return svg;
}

async function runCollision() {
  const out = $("col-out");
  const viz = $("col-viz");
  const off = parseFloat($("col-offset").value);
  // Two squares; B slides along X via the slider.
  const polyA = [[0, 0], [4, 0], [4, 4], [0, 4]];
  const polyB = [[off, 1], [off + 4, 1], [off + 4, 5], [off, 5]];
  try {
    const data = await api("/api/geometry/collision", { poly_a: polyA, poly_b: polyB });
    viz.innerHTML = drawCollision(polyA, polyB, data.collide);
    out.textContent = JSON.stringify({ collide: data.collide, mtv: data.mtv, offset_x: off }, null, 2);
  } catch (e) {
    out.textContent = "閿欒: " + e.message;
  }
}

/* ==========================================================================
   08 - ETL
   ========================================================================== */
async function runEtl() {
  const out = $("etl-out");
  const btn = event.target;
  btn.disabled = true;
  try {
    const rows = JSON.parse($("etl-input").value);
    const data = await api("/api/etl/clean", { rows, options: {} });
    renderJson(out, data);
  } catch (e) {
    out.textContent = "閿欒: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

/* ==========================================================================
   09 - Finance forecast
   ========================================================================== */
function drawForecast(points, W = 620, H = 260) {
  const values = points.map((p) => p.predicted);
  const xs = values.map((_, i) => i);
  const v = fit(values.map((val, i) => [i, val]), W, H);
  let svg = svgOpen(W, H);
  svg += `<line x1="30" y1="${H - 24}" x2="${W - 20}" y2="${H - 24}" class="axis"/>`;
  svg += `<line x1="30" y1="20" x2="30" y2="${H - 24}" class="axis"/>`;
  const ptsStr = values.map((val, i) => `${v.x(i)},${v.y(val)}`).join(" ");
  svg += `<polyline points="${ptsStr}" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linejoin="round"/>`;
  values.forEach((val, i) => {
    svg += `<circle cx="${v.x(i)}" cy="${v.y(val)}" r="4" fill="#0ea5e9"/>`;
    svg += `<text x="${v.x(i)}" y="${v.y(val) - 8}" text-anchor="middle">${val.toFixed(1)}</text>`;
    svg += `<text x="${v.x(i)}" y="${H - 8}" text-anchor="middle">${points[i].index}</text>`;
  });
  svg += `</svg>`;
  return svg;
}

async function runFinance() {
  const out = $("fin-out");
  const viz = $("fin-viz");
  const btn = event.target;
  btn.disabled = true;
  try {
    // Deterministic demo series (geometric random walk seeded).
    const seed = 42;
    const prices = [100];
    let state = seed;
    for (let i = 1; i < 90; i++) {
      state = (state * 1664525 + 1013904223) % 4294967296;
      const r = (state / 4294967296) - 0.5;
      prices.push(Math.max(prices[i - 1] * (1 + r * 0.04), 20));
    }
    const data = await api("/api/finance/forecast", {
      prices,
      model: $("fin-model").value,
      horizon: parseInt($("fin-horizon").value, 10) || 5,
      lags: 5,
    });
    viz.innerHTML = drawForecast(data.points);
    renderJson(out, { model: data.model, evaluation: data.evaluation, points: data.points });
  } catch (e) {
    out.textContent = "閿欒: " + e.message;
  } finally {
    btn.disabled = false;
  }
}
