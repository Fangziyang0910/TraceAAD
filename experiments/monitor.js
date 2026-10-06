"use strict";
const REFRESH_MS = 15000;
const STATUS = {finished: "已完成", running: "运行中", queued: "排队中", blocked: "受阻", unknown: "未确认"};
const OUTCOME = {
  valid: ["有效", "ok"], ok: ["有效", "ok"], expanded: ["有效", "ok"],
  improve: ["有效 · 提升", "ok"], regress: ["有效 · 退步", "ok"], plateau: ["有效 · 持平", "ok"],
  eval_failed: ["评测失败", "bad"], evaluation_failed: ["评测失败", "bad"],
  exec_error: ["执行错误", "bad"], invalid: ["候选无效", "bad"],
  duplicate: ["重复", "neutral"], runtime_error: ["运行错误", "bad"],
  timeout: ["超时", "warn"], invalid_output: ["输出无效", "bad"], invalid_result: ["结果无效", "bad"],
  invalid_source: ["源码无效", "bad"], known_failure: ["已知失败", "neutral"], delivery_failed: ["未交付", "warn"],
};
const $ = id => document.getElementById(id);
const tip = $("tip");

const S = {
  batch: "", metric: (() => { try { return localStorage.getItem("monitor.metric") || "value"; } catch { return "value"; } })(),
  view: "monitor", data: null, fetchedAt: 0, structure: "", charts: new Map(), run: null, detail: null,
  timer: null, inflight: false, request: null,
};
const etags = new Map(), bodies = new Map();

/* ---------- formatting ---------- */
const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[c]);
function fmt(v, d = 4) {
  if (v == null || !Number.isFinite(+v)) return "–";
  const n = +v, a = Math.abs(n);
  return a >= 1000 ? n.toFixed(1) : a >= 100 ? n.toFixed(2) : n.toFixed(d);
}
function axisFmt(v, span) {
  const d = span > 0 ? Math.max(0, Math.min(5, 2 - Math.floor(Math.log10(span)))) : 3;
  return (+v).toFixed(d);
}
function rel(ms) {
  const s = Math.max(0, Math.round((Date.now() - ms) / 1000));
  return s < 5 ? "刚刚" : s < 60 ? `${s} 秒前` : s < 3600 ? `${Math.floor(s / 60)} 分钟前` : `${Math.floor(s / 3600)} 小时前`;
}
function dur(sec) {
  if (sec == null || !Number.isFinite(sec)) return "–";
  if (sec < 60) return sec <= 0 ? "0 分" : "<1 分";
  const m = Math.ceil(sec / 60), d = Math.floor(m / 1440), h = Math.floor(m % 1440 / 60), r = m % 60;
  return d ? `${d} 天 ${h} 时` : h ? `${h} 时 ${r} 分` : `${r} 分`;
}
function clock(v) {
  const t = v ? new Date(v) : null;
  return t && !Number.isNaN(+t) ? t.toLocaleString("zh-CN", {month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit"}) : "–";
}
function speed(t) {
  const r = t?.rate_per_minute;
  return r == null ? "–" : `${r < .01 ? "<0.01" : r.toFixed(2)}/分`;
}
function eta(t) {
  if (!t) return "–";
  if (t.state === "estimated") return dur(t.eta_seconds);
  return {finished: "已完成", search_complete: "搜索结束", warming_up: "积累中", stale: "久未推进",
          inactive: "等待", partial: "部分可估", unavailable: "暂无估算"}[t.state] || "–";
}
const tone = s => (OUTCOME[s] || [s, s === "valid" ? "ok" : "neutral"])[1];
const outcomeLabel = s => (OUTCOME[s] || [s])[0];
const toneVar = {ok: "--ok", bad: "--bad", warn: "--warn", neutral: "--faint"};
const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const seriesColor = i => `var(--s${i % 8 + 1})`;

/* metric: fitness is always "higher is better"; value is the raw objective. */
const shown = (fitness, task) => fitness == null ? null
  : S.metric === "fitness" || task.direction === "max" ? fitness : -fitness;
function bestFitness(run) {
  const last = run.curve?.at(-1);
  return last?.fitness ?? run.best_fitness ?? null;
}

/* ---------- network ---------- */
async function getJSON(url, {signal} = {}) {
  const headers = {};
  if (etags.has(url)) headers["If-None-Match"] = etags.get(url);
  const response = await fetch(url, {headers, cache: "no-store", signal});
  if (response.status === 304 && bodies.has(url)) return {data: bodies.get(url), changed: false};
  if (!response.ok) throw new Error(`${response.status} ${response.statusText || "请求失败"}`);
  const data = await response.json();
  signal?.throwIfAborted();
  const tag = response.headers.get("ETag");
  if (tag) { etags.set(url, tag); bodies.set(url, data); }
  return {data, changed: true};
}
function hashParams() { return new URLSearchParams(location.hash.slice(1)); }
function writeHash() {
  const params = new URLSearchParams({b: S.batch});
  if (S.run) params.set("r", `${S.run.task}/${S.run.name}`);
  if (S.view === "compare") {
    params.set("v", "compare");
    if (CMP.ids.length) params.set("c", CMP.ids.join(","));
    if (CMP.ref && CMP.ref !== CMP.ids[0]) params.set("ref", CMP.ref);
  }
  history.replaceState(null, "", `#${params}`);
}
const stateURL = b => `/api/state?batch=${encodeURIComponent(b)}`;
const runURL = r => `/api/run?batch=${encodeURIComponent(r.batch)}&task=${encodeURIComponent(r.task)}&name=${encodeURIComponent(r.name)}`;

function setLive(mode, text) {
  $("live").className = `live ${mode}`;
  $("live-text").textContent = text;
}
function showError(message) {
  $("error").hidden = !message;
  $("error").textContent = message || "";
}

async function loadBatches() {
  const {data} = await getJSON("/api/batches");
  const select = $("batch"), previous = S.batch;
  select.replaceChildren(...data.batches.map(b => new Option(b.label || b.id, b.id)));
  const wanted = previous || hashParams().get("b") || data.default_batch;
  S.batch = data.batches.some(b => b.id === wanted) ? wanted : data.batches[0]?.id || "";
  select.value = S.batch;
  if (previous && previous !== S.batch) {
    closeRun(); S.data = null; S.structure = "";
    writeHash();
  }
}

async function refresh({force = false} = {}) {
  if (!S.batch) {
    try { await loadBatches(); }
    catch (error) { setLive("err", "连接失败"); showError(`读取失败：${error.message}`); return; }
    if (!S.batch) { setLive("", "暂无批次"); return; }
  }
  if (S.inflight && S.request?.batch === S.batch && !force) return;
  S.request?.controller.abort();
  const request = {batch: S.batch, controller: new AbortController()};
  S.request = request; S.inflight = true;
  const batch = request.batch;
  $("refresh").classList.add("spin");
  try {
    if (force) etags.delete(stateURL(batch));
    const {data, changed} = await getJSON(stateURL(batch), {signal: request.controller.signal});
    if (S.request !== request || batch !== S.batch) return;
    S.fetchedAt = Date.now();
    setLive("on", "实时 · 刚刚"); showError("");
    if (changed || !S.data) { S.data = data; render(data); }
    if (S.run) await loadDetail();
  } catch (error) {
    if (error.name !== "AbortError" && S.request === request && batch === S.batch) {
      setLive("err", "连接失败"); showError(`读取失败：${error.message}`);
    }
  } finally {
    if (S.request === request) {
      S.inflight = false; S.request = null;
      $("refresh").classList.remove("spin");
    }
  }
}

/* ---------- overview rendering (incremental) ---------- */
function render(data) {
  renderKPIs(data);
  const tasks = data.tasks || [];
  const structure = tasks.map(t => `${t.key}:${t.runs.map(r => r.name).join(",")}`).join("|");
  if (structure !== S.structure) buildTasks(tasks, structure);
  for (const task of tasks) updateTask(task);
}

function renderKPIs(data) {
  const s = data.summary || {};
  const runs = s.runs || 0;
  $("k-runs").innerHTML = `${s.finished ?? 0}<small>/ ${runs}</small>`;
  $("k-seg").innerHTML = ["finished", "running", "queued", "blocked", "unknown"]
    .map(k => s[k] ? `<i class="${k}" style="width:${s[k] / runs * 100}%" title="${STATUS[k]} ${s[k]}"></i>` : "").join("");
  $("k-runs-n").textContent = ["running", "queued", "blocked", "unknown"].filter(k => s[k]).map(k => `${s[k]} ${STATUS[k]}`).join(" · ") || "全部完成";
  const pct = s.budget ? s.budget_used / s.budget * 100 : 0;
  $("k-budget").innerHTML = `${pct >= 99.95 || pct === 0 ? pct.toFixed(0) : pct.toFixed(1)}<small>%</small>`;
  $("k-budget-bar").style.width = `${Math.min(100, pct)}%`;
  $("k-budget-n").textContent = `${(s.budget_used ?? 0).toLocaleString()} / ${(s.budget ?? 0).toLocaleString()}`;
  const t = s.timing;
  // Partial estimates still bound the batch from below: the slowest estimable run.
  const etas = (data.tasks || []).flatMap(task => task.runs)
    .filter(r => r.status === "running" && r.timing?.state === "estimated").map(r => r.timing.eta_seconds);
  $("k-eta").textContent = t?.state === "partial" && etas.length ? `≥ ${dur(Math.max(...etas))}` : eta(t);
  $("k-eta-n").textContent = t?.eta_at ? `约 ${clock(t.eta_at)} · 按最慢一路` : t?.pending_runs ? `${t.eta_runs}/${t.pending_runs} 路可估算` : "不含选择与测试";
  $("k-nodes").textContent = (s.valid_nodes ?? 0).toLocaleString();
  $("k-nodes-n").textContent = s.valid_rate != null ? `记录有效率 ${(s.valid_rate * 100).toFixed(1)}% · ${s.valid_candidate_count}/${s.candidate_count}` : "暂无候选记录";
  const alerts = (data.tasks || []).flatMap(task => task.runs).filter(r =>
    (r.status === "running" && r.timing?.state === "stale") || (r.budget && r.budget_used > r.budget)).length;
  document.title = `${alerts ? `⚠${alerts} · ` : ""}${s.finished ?? 0}/${runs} · ${pct.toFixed(0)}% · TraceAAD`;
}

function buildTasks(tasks, structure) {
  S.structure = structure;
  S.charts.clear();
  const root = $("tasks");
  if (!tasks.length) { root.innerHTML = '<div class="empty">该批次没有运行</div>'; return; }
  root.innerHTML = tasks.map(task => `
    <article class="task" data-task="${esc(task.key)}">
      <header class="task-head">
        <div><h2>${esc(task.label)}</h2><div class="dir">原始${esc(task.unit)} · ${task.direction === "min" ? "越低越好" : "越高越好"}</div></div>
        <div class="agg" data-agg></div>
      </header>
      <div class="chart-box"><canvas aria-label="${esc(task.label)} 各 rep 历史最优曲线"></canvas></div>
      <div class="chart-note" data-note></div>
      <div class="runs">
        <div class="run-head"><span>重复</span><span>状态</span><span>进度</span><span>搜索最优</span><span title="搜索结束后 top-5 在独立选择集上复评，选中程序的得分；cvrp/op 用 val_50，与搜索分不同尺度">选择分</span><span>后端</span></div>
        ${task.runs.map((run, i) => `
          <button class="run" type="button" data-run="${esc(run.name)}" style="--c:${seriesColor(i)}">
            <span class="rep"><span class="sw"></span>${run.repeat == null ? esc(run.name) : `Rep ${esc(run.repeat)}`}</span>
            <span class="pill" data-status></span>
            <span class="prog"><span class="track"><i data-bar></i></span><span class="pm"><span data-used></span><span data-eta></span></span></span>
            <span class="r-num" data-best></span>
            <span class="r-num dim" data-sel></span>
            <span class="r-be">${esc(run.backend || "–")}</span>
          </button>`).join("")}
      </div>
    </article>`).join("");
  for (const card of root.querySelectorAll(".task")) {
    const canvas = card.querySelector("canvas");
    const chart = {canvas, card, key: card.dataset.task, series: [], hover: null, sig: ""};
    S.charts.set(chart.key, chart);
    bindOverlayHover(chart);
    resizer.observe(canvas.parentElement);
  }
}

function updateTask(task) {
  const card = document.querySelector(`.task[data-task="${CSS.escape(task.key)}"]`);
  if (!card) return;
  task.runs.forEach(run => {
    const row = card.querySelector(`.run[data-run="${CSS.escape(run.name)}"]`);
    if (!row) return;
    const pct = run.budget ? Math.min(100, run.budget_used / run.budget * 100) : 0;
    const t = run.timing;
    const stale = run.status === "running" && t?.state === "stale";
    const pill = row.querySelector("[data-status]");
    pill.className = `pill ${stale ? "stale" : run.status}`;
    pill.textContent = stale ? "停滞" : STATUS[run.status] || run.status;
    pill.title = stale ? "长时间未见新候选：后端可能中断或卡在慢评价" : "";
    row.querySelector("[data-bar]").style.width = `${pct}%`;
    const over = run.budget && run.budget_used > run.budget;
    row.querySelector("[data-used]").innerHTML = over
      ? `<span class="over" title="消耗超过预算：可能有重复计数或续跑">${run.budget_used}/${run.budget}</span>` : `${run.budget_used}/${run.budget}`;
    row.querySelector("[data-eta]").textContent = run.status === "running"
      ? (t?.state === "estimated" ? `剩 ${dur(t.eta_seconds)}` : speed(t)) : "";
    row.querySelector("[data-best]").textContent = fmt(shown(bestFitness(run), task));
    const info = run.selection_info;
    const tied = info && info.finalists - info.failed > 1 && info.distinct === 1;
    row.querySelector("[data-sel]").innerHTML = esc(fmt(shown(run.selection_fitness, task)))
      + (tied ? `<span class="flag" title="${info.finalists - info.failed} 个 finalist 的选择分完全相同：选择阶段没有区分力">并列</span>` : "");
    row.title = `${run.name}\nseed ${run.seed ?? "–"} · 有效候选 ${run.valid_nodes} · ${speed(t)}`;
  });
  const bests = task.runs.map(bestFitness).filter(v => v != null).map(v => shown(v, task));
  const agg = card.querySelector("[data-agg]");
  if (bests.length) {
    const mean = bests.reduce((a, b) => a + b, 0) / bests.length;
    const sd = bests.length > 1 ? Math.sqrt(bests.reduce((a, b) => a + (b - mean) ** 2, 0) / (bests.length - 1)) : 0;
    const better = S.metric === "fitness" || task.direction === "max" ? Math.max(...bests) : Math.min(...bests);
    agg.innerHTML = `<b>${fmt(mean)} <span style="font-weight:500;color:var(--muted);font-size:12px">± ${fmt(sd, 3)}</span></b>搜索最优 均值 ± 标准差 · 最好 ${fmt(better)}`;
  } else agg.textContent = "暂无有效候选";
  const chart = S.charts.get(task.key);
  const sig = S.metric + JSON.stringify(task.runs.map(r => [r.name, r.status, r.budget, r.budget_used, r.x_label, r.curve]));
  if (chart && chart.sig !== sig) {
    chart.sig = sig;
    chart.task = task;
    chart.series = task.runs.map((run, i) => ({
      name: run.repeat == null ? run.name : `Rep ${run.repeat}`, color: i,
      live: run.status === "running",
      points: (run.curve || []).filter(p => Number.isFinite(p.fitness)).map(p => ({x: p.evaluation, f: p.fitness, p})),
    }));
    chart.maxX = Math.max(1, ...task.runs.map(r => Math.max(r.budget || 0, r.budget_used || 0, r.curve?.at(-1)?.evaluation || 0)));
    chart.xLabel = task.runs[0]?.x_label || "已记录序号";
    drawChart(chart);
  }
}

/* ---------- canvas charts ---------- */
const resizer = new ResizeObserver(entries => {
  for (const entry of entries) {
    const canvas = entry.target.querySelector("canvas");
    const chart = [...S.charts.values(), S.detailChart].find(c => c && c.canvas === canvas);
    if (chart) drawChart(chart);
  }
});

function stepAt(points, x) {
  let lo = 0, hi = points.length - 1, found = null;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (points[mid].x <= x) { found = points[mid]; lo = mid + 1; } else hi = mid - 1;
  }
  return found;
}

/* Robust y-range: ignore the early warm-up (first 8% of budget) so late-stage
   differences between reps stay visible; clipped early values are noted. */
function yDomain(series, maxX) {
  const all = series.flatMap(s => s.points);
  if (!all.length) return null;
  const cut = maxX * 0.08;
  const late = series.flatMap(s => {
    const start = stepAt(s.points, cut);
    return [...(start ? [start] : []), ...s.points.filter(p => p.x > cut)];
  });
  const vals = (late.length ? late : all).map(p => p.f);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.1 || Math.max(Math.abs(hi) * 0.01, 1e-6);
  lo -= pad; hi += pad;
  return {lo, hi, clipped: all.some(p => p.f < lo)};
}

function drawChart(chart) {
  const {canvas} = chart;
  const box = canvas.parentElement.getBoundingClientRect();
  if (!box.width) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const W = Math.round(box.width), H = Math.round(box.height);
  if (canvas.width !== W * dpr || canvas.height !== H * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const task = chart.task || {};
  const colors = {grid: cssVar("--grid"), text: cssVar("--faint"), ink: cssVar("--ink"), panel: cssVar("--panel")};
  const dom = yDomain(chart.series, chart.maxX);
  const note = chart.card?.querySelector("[data-note]");
  ctx.font = "11px Inter, system-ui, sans-serif";
  if (!dom) {
    ctx.fillStyle = colors.text; ctx.textAlign = "center";
    ctx.fillText("暂无有效候选", W / 2, H / 2);
    if (note) note.textContent = "";
    return;
  }
  const L = 58, R = 14, T = 10, B = 22;
  const x = v => L + v / chart.maxX * (W - L - R);
  const y = v => T + (dom.hi - v) / (dom.hi - dom.lo) * (H - T - B);
  chart.geom = {L, R, T, B, W, H, x, y, dom};
  // grid + axis labels (in the chosen display metric)
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 3; i++) {
    const f = dom.lo + (dom.hi - dom.lo) * i / 3, py = y(f);
    ctx.strokeStyle = colors.grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, Math.round(py) + .5); ctx.lineTo(W - R, Math.round(py) + .5); ctx.stroke();
    ctx.fillStyle = colors.text;
    ctx.fillText(axisFmt(shown(f, task), dom.hi - dom.lo), L - 7, py);
  }
  ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";
  ctx.fillText("0", L, H - 6);
  ctx.textAlign = "right"; ctx.fillText(String(chart.maxX), W - R, H - 6);
  ctx.textAlign = "center"; ctx.fillText(chart.xLabel || "已记录序号", (L + W - R) / 2, H - 6);
  // series
  ctx.save();
  ctx.beginPath(); ctx.rect(L, T - 2, W - L - R + 2, H - T - B + 4); ctx.clip();
  const single = chart.series.length === 1;
  chart.series.forEach(s => {
    if (!s.points.length) return;
    const color = cssVar(`--s${s.color % 8 + 1}`);
    const dim = chart.focus != null && chart.focus !== s.color;
    ctx.globalAlpha = dim ? 0.25 : 1;
    ctx.strokeStyle = color; ctx.lineWidth = single ? 2 : 1.8; ctx.lineJoin = "round";
    ctx.beginPath();
    s.points.forEach((p, i) => {
      if (i === 0) ctx.moveTo(x(p.x), y(p.f));
      else { ctx.lineTo(x(p.x), y(s.points[i - 1].f)); ctx.lineTo(x(p.x), y(p.f)); }
    });
    ctx.stroke();
    if (single) {
      const grad = ctx.createLinearGradient(0, T, 0, H - B);
      grad.addColorStop(0, color + "33"); grad.addColorStop(1, color + "00");
      ctx.lineTo(x(s.points.at(-1).x), H - B); ctx.lineTo(x(s.points[0].x), H - B); ctx.closePath();
      ctx.fillStyle = grad; ctx.fill();
      ctx.fillStyle = color;
      for (const p of s.points) if (p.p.kind === "breakthrough" || p.p.kind === "initial") {
        ctx.beginPath(); ctx.arc(x(p.x), y(p.f), chart.hoverPoint === p ? 5 : 3, 0, Math.PI * 2); ctx.fill();
      }
    }
    const last = s.points.at(-1);
    if (s.live) {
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(x(last.x), y(last.f), 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = colors.panel; ctx.lineWidth = 1.5; ctx.stroke();
    }
  });
  ctx.globalAlpha = 1;
  ctx.restore();
  // hover guide
  if (chart.hoverX != null) {
    ctx.strokeStyle = colors.text; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x(chart.hoverX) + .5, T); ctx.lineTo(x(chart.hoverX) + .5, H - B); ctx.stroke();
    ctx.setLineDash([]);
  }
  if (note) note.textContent = dom.clipped ? "纵轴已按前 8% 预算之后的数值缩放，更早的较差值被裁掉" : "";
}

function placeTip(event) {
  tip.hidden = false;
  const b = tip.getBoundingClientRect();
  let left = event.clientX + 14, top = event.clientY + 14;
  if (left + b.width > innerWidth - 8) left = event.clientX - b.width - 14;
  if (top + b.height > innerHeight - 8) top = event.clientY - b.height - 14;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}
function hideTip() { tip.hidden = true; }

function bindOverlayHover(chart) {
  let frame = 0;
  chart.canvas.addEventListener("pointermove", event => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      const g = chart.geom;
      if (!g) return;
      const rect = chart.canvas.getBoundingClientRect();
      const px = event.clientX - rect.left;
      if (px < g.L || px > g.W - g.R) { chart.hoverX = null; drawChart(chart); hideTip(); return; }
      const xv = Math.round((px - g.L) / (g.W - g.L - g.R) * chart.maxX);
      chart.hoverX = xv;
      drawChart(chart);
      const task = chart.task;
      const rows = chart.series.map(s => {
        const end = s.points.at(-1);
        const at = end && xv <= end.x ? stepAt(s.points, xv) : null;
        return `<div class="row"><span><i class="sw" style="--c:${seriesColor(s.color)}"></i>${esc(s.name)}</span><strong>${at ? fmt(shown(at.f, task)) : "–"}</strong></div>`;
      }).join("");
      tip.innerHTML = `<h4>${esc(chart.xLabel || "已记录序号")} ${xv}</h4>${rows}`;
      placeTip(event);
    });
  });
  chart.canvas.addEventListener("pointerleave", () => {
    cancelAnimationFrame(frame);
    chart.hoverX = null; drawChart(chart); hideTip();
  });
}

function bindDetailHover(chart) {
  let frame = 0;
  chart.canvas.addEventListener("pointermove", event => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      const g = chart.geom, s = chart.series[0];
      if (!g || !s) return;
      const rect = chart.canvas.getBoundingClientRect();
      const px = event.clientX - rect.left, py = event.clientY - rect.top;
      let best = null, dist = 18;
      for (const p of s.points) {
        if (p.p.kind === "progress") continue;
        const d = Math.hypot(g.x(p.x) - px, g.y(p.f) - py);
        if (d < dist) { dist = d; best = p; }
      }
      if (chart.hoverPoint !== best) { chart.hoverPoint = best; drawChart(chart); }
      if (!best) { hideTip(); return; }
      const p = best.p, task = chart.task;
      tip.innerHTML = `<h4>${p.kind === "initial" ? "首个有效候选" : "刷新历史最优"}</h4>`
        + `<div class="row"><span>${S.metric === "fitness" ? "Fitness" : `原始${esc(task.unit)}`}</span><strong>${fmt(shown(p.fitness, task), 6)}</strong></div>`
        + (p.gain != null ? `<div class="row"><span>Fitness 改进</span><strong>+${fmt(p.gain, 6)}</strong></div>` : "")
        + `<div class="row"><span>${esc(chart.xLabel || "已记录序号")}</span><strong>${p.evaluation}</strong></div>`
        + `<div class="row"><span>动作</span><strong>${esc(p.operator && p.operator !== "unknown" ? p.operator : "未记录")}</strong></div>`
        + (p.candidate != null ? `<div class="row"><span>候选 / 父代</span><strong>#${esc(p.candidate)}${p.parent_id != null ? ` ← #${esc(p.parent_id)}` : ""}</strong></div>` : "")
        + (p.idea ? `<p>${esc(p.idea)}</p>` : "");
      placeTip(event);
    });
  });
  chart.canvas.addEventListener("pointerleave", () => {
    cancelAnimationFrame(frame);
    chart.hoverPoint = null; drawChart(chart); hideTip();
  });
}

/* ---------- detail drawer ---------- */
function openRun(taskKey, name) {
  S.run = {batch: S.batch, task: taskKey, name};
  S.detail = null;
  const task = S.data?.tasks.find(t => t.key === taskKey);
  const index = task ? task.runs.findIndex(r => r.name === name) : 0;
  S.run.color = Math.max(0, index);
  $("d-title").innerHTML = `<span class="sw" style="--c:${seriesColor(S.run.color)}"></span>${esc(task?.label || taskKey)} · ${esc(task?.runs[index]?.repeat != null ? `Rep ${task.runs[index].repeat}` : name)}`;
  $("d-sub").textContent = "正在读取…";
  $("d-body").innerHTML = '<div class="skeleton" style="height:420px"></div>';
  $("drawer").classList.add("open"); $("scrim").classList.add("open");
  $("drawer").setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
  $("close").focus();
  writeHash();
  loadDetail();
}
function closeRun() {
  if (!S.run) return;
  S.detailRequest?.abort(); S.detailRequest = null;
  bodies.delete(runURL(S.run)); etags.delete(runURL(S.run));
  S.run = null; S.detail = null;
  if (S.detailChart) { resizer.unobserve(S.detailChart.canvas.parentElement); S.detailChart = null; }
  hideTip();
  $("drawer").classList.remove("open"); $("scrim").classList.remove("open");
  $("drawer").setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
  writeHash();
}
async function loadDetail() {
  const selected = S.run;
  if (!selected) return;
  S.detailRequest?.abort();
  const request = new AbortController();
  S.detailRequest = request;
  try {
    const {data, changed} = await getJSON(runURL(selected), {signal: request.signal});
    if (S.run !== selected || S.detailRequest !== request) return;
    if (changed || !S.detail) { S.detail = data; renderDetail(data); }
  } catch (error) {
    if (error.name === "AbortError" || S.run !== selected || S.detailRequest !== request) return;
    if (!S.detail) $("d-body").innerHTML = `<div class="empty">${esc(error.message)}</div>`;
    else $("d-sub").textContent = `刷新失败，保留上次数据：${error.message}`;
  }
}

function renderDetail(run) {
  const task = {...run.task_meta, key: run.task};
  const body = $("d-body");
  const scroll = body.scrollTop;
  const codeOpen = new Map([...body.querySelectorAll("[data-program]")].map(el => [el.dataset.program, el.querySelector("details.code")?.open]));
  const t = run.timing || {};
  const best = bestFitness(run);
  const outcomes = Object.entries(run.outcomes || {}).sort((a, b) => b[1] - a[1]);
  const total = run.candidate_count ?? outcomes.reduce((a, [, n]) => a + n, 0);
  const denominator = Math.max(1, total);
  const programs = [];
  const same = run.search_best?.code && run.search_best.code === run.selected_best?.code && run.search_best.fitness === run.selected_best.fitness;
  if (same) programs.push({role: "search", title: "搜索最优 / 最终选中程序", program: run.search_best});
  else {
    if (run.search_best) programs.push({role: "search", title: "搜索最优程序", program: run.search_best});
    if (run.selected_best) programs.push({role: "selected", title: "最终选中程序", program: run.selected_best});
  }
  const ops = Object.entries(run.operators || {}).sort((a, b) => b[1] - a[1]);
  const opMax = Math.max(1, ...ops.map(([, n]) => n));
  const breakthroughs = (run.curve || []).filter(p => p.kind === "breakthrough").length;
  $("d-sub").innerHTML = `<span class="pill ${esc(run.status)}">${STATUS[run.status] || esc(run.status)}</span> · ${esc(run.backend || "–")} · seed ${esc(run.seed ?? "–")} · ${esc(run.name)}`;
  const tile = (k, v) => `<div class="tile"><small>${k}</small><strong>${v}</strong></div>`;
  body.innerHTML = `
    <div class="tiles">
      ${tile("进度", `${run.budget_used} / ${run.budget}`)}
      ${tile("记录有效率", run.valid_rate != null ? `${(run.valid_rate * 100).toFixed(1)}% <span style="color:var(--muted);font-weight:500;font-size:12px">(${run.valid_candidate_count}/${total})</span>` : "–")}
      ${tile(`搜索最优 ${S.metric === "fitness" ? "Fitness" : esc(task.unit)}`, fmt(shown(best, task), 5))}
      ${tile("选择分", fmt(shown(run.selection_fitness, task), 5))}
      ${tile("平均速度", speed(t))}
      ${tile(t.basis === "active_time" ? "累计搜索耗时" : "墙钟耗时", dur(t.elapsed_seconds))}
      ${tile("预计剩余", eta(t))}
      ${tile("预计完成", t.eta_at ? clock(t.eta_at) : run.updated_at ? `完成于 ${clock(run.updated_at)}` : "–")}
    </div>
    <section class="sec"><h3>历史最优曲线<small>${breakthroughs} 次突破 · 悬停圆点查看候选</small></h3><div class="d-chart"><canvas></canvas></div></section>
    <section class="sec"><h3>候选结果<small>共 ${total} 条记录</small></h3>
      <div class="stack">${outcomes.map(([k, n]) => `<i style="width:${n / denominator * 100}%;background:var(${toneVar[tone(k)]});${tone(k) === "neutral" ? "opacity:.6" : ""}" title="${esc(outcomeLabel(k))} ${n}"></i>`).join("")}</div>
      <div class="legend">${outcomes.map(([k, n]) => `<span><i style="background:var(${toneVar[tone(k)]})"></i>${esc(outcomeLabel(k))} <b>${n}</b></span>`).join("")}</div>
    </section>
    <div class="two">
      <section class="sec"><h3>动作分布</h3><div class="bars">${ops.map(([k, n]) => `<div class="bar-row"><span>${esc(k)}</span><span class="track"><i style="width:${n / opMax * 100}%;--c:var(--accent)"></i></span><b>${n}</b></div>`).join("") || '<span class="empty">–</span>'}</div></section>
      <section class="sec"><h3>最近候选</h3>
        <table class="recent"><thead><tr><th>${esc(run.x_label || "序号")}</th><th>动作</th><th>结果</th><th class="nr">${S.metric === "fitness" ? "Fitness" : esc(task.unit)}</th></tr></thead>
        <tbody>${(run.recent || []).map(r => `<tr><td class="num">${esc(r.evaluation)}</td><td>${esc(r.operator === "unknown" ? "–" : r.operator)}</td><td><span class="st ${tone(r.status) === "ok" ? "valid" : tone(r.status)}">${esc(outcomeLabel(r.status))}</span></td><td class="nr">${fmt(shown(r.fitness, task))}</td></tr>`).join("")}</tbody></table>
      </section>
    </div>
    ${programs.map(({role, title, program}) => `<section class="sec" data-program="${role}"><h3>${title}<small>候选 #${esc(program.id ?? "–")} · 训练 ${S.metric === "fitness" ? "Fitness" : esc(task.unit)} ${fmt(shown(program.fitness, task), 5)}${program.operator ? ` · ${esc(program.operator)}` : ""}</small></h3>
      <p class="idea"></p>
      ${program.code ? `<details class="code" ${codeOpen.get(role) ? "open" : ""}><summary>查看代码</summary><pre></pre></details>` : '<p class="empty">历史记录未保存此候选的源码</p>'}</section>`).join("")}`;
  for (const {role, program} of programs) {
    const section = body.querySelector(`[data-program="${role}"]`);
    section.querySelector(".idea").textContent = program.idea || "（无 idea 记录）";
    const pre = section.querySelector("pre"), details = section.querySelector("details.code");
    if (!pre || !details) continue;
    const fill = () => { if (details.open && !pre.textContent) pre.textContent = program.code; };
    if (details.open) fill();
    details.addEventListener("toggle", fill);
  }
  const canvas = body.querySelector(".d-chart canvas");
  if (S.detailChart) resizer.unobserve(S.detailChart.canvas.parentElement);
  S.detailChart = {
    canvas, task, xLabel: run.x_label, maxX: Math.max(1, run.budget || 0, run.budget_used || 0, run.curve?.at(-1)?.evaluation || 0),
    series: [{name: run.name, color: S.run?.color ?? 0, live: run.status === "running",
      points: (run.curve || []).filter(p => Number.isFinite(p.fitness)).map(p => ({x: p.evaluation, f: p.fitness, p}))}],
  };
  bindDetailHover(S.detailChart);
  resizer.observe(canvas.parentElement);
  body.scrollTop = scroll;
}

/* ---------- cross-batch comparison ---------- */
const CMP = {cohorts: [], ids: [], ref: null, data: null, col: {}};
const GROUP = {train: "训练", selection: "选择", test: "测试 · 训练规模", gen: "泛化 · 其他规模"};
const isTrace = c => c.batch.startsWith("traceaad");

async function loadCohorts() {
  const {data} = await getJSON("/api/cohorts");
  CMP.cohorts = data.cohorts || [];
  const known = new Set(CMP.cohorts.map(c => c.id));
  CMP.ids = CMP.ids.filter(id => known.has(id));
  if (!CMP.ids.length) {
    const first = CMP.cohorts.find(c => c.batch === S.batch)?.id || CMP.cohorts[0]?.id;
    // Previous version: the highest TraceAAD version below the current batch with held-out results.
    const version = c => (c.batch.match(/_v(\d+(?:_\d+)*)/)?.[1] || "").split("_").map(Number);
    const newer = (a, b) => { for (let i = 0; i < Math.max(a.length, b.length); i++) { const d = (a[i] || 0) - (b[i] || 0); if (d) return d; } return 0; };
    const current = version(CMP.cohorts.find(c => c.id === first) || {batch: ""});
    const prior = CMP.cohorts.filter(c => c.id !== first && isTrace(c) && c.heldout_tasks.length >= 4
        && version(c).length && (!current.length || newer(version(c), current) < 0))
      .sort((a, b) => newer(version(b), version(a)))[0]?.id;
    CMP.ids = [first, prior, "eoh", "reevo"].filter((id, i, a) => id && known.has(id) && a.indexOf(id) === i);
  }
  if (!CMP.ids.includes(CMP.ref)) CMP.ref = CMP.ids[0] || null;
}

function renderChips() {
  const byId = new Map(CMP.cohorts.map(c => [c.id, c]));
  const chips = CMP.ids.map((id, i) => `<span class="chip ${id === CMP.ref ? "ref" : ""}">
      <span class="sw" style="--c:${seriesColor(i)}"></span><span class="name" title="${esc(id)}">${esc(byId.get(id)?.label || id)}</span>
      <button type="button" class="star" data-ref="${esc(id)}" aria-pressed="${id === CMP.ref}" title="设为参照">★</button>
      <button type="button" data-remove="${esc(id)}" title="移除" aria-label="移除 ${esc(id)}">×</button></span>`).join("");
  const rest = CMP.cohorts.filter(c => !CMP.ids.includes(c.id));
  const group = (label, items) => items.length ? `<optgroup label="${label}">${items.map(c =>
    `<option value="${esc(c.id)}">${esc(c.label)}${c.heldout_tasks.length ? ` · held-out ${c.heldout_tasks.length}/5` : " · 无 held-out"}</option>`).join("")}</optgroup>` : "";
  $("cmp-chips").innerHTML = chips + (CMP.ids.length < 12 ? `<select class="ctl chip-add" id="cmp-add" aria-label="添加对比对象">
    <option value="">＋ 添加对比对象…</option>${group("TraceAAD", rest.filter(isTrace))}${group("基线方法", rest.filter(c => !isTrace(c)))}</select>` : "");
}

async function loadCompare({force = false} = {}) {
  CMP.request?.abort();
  const request = new AbortController();
  CMP.request = request;
  renderChips();
  writeHash();
  if (!CMP.ids.length) { CMP.data = null; renderCompare(); return; }
  const url = `/api/compare?cohorts=${encodeURIComponent(CMP.ids.join(","))}`;
  if (force) etags.delete(url);
  try {
    const {data, changed} = await getJSON(url, {signal: request.signal});
    if (CMP.request !== request || url !== `/api/compare?cohorts=${encodeURIComponent(CMP.ids.join(","))}`) return;
    $("cmp-error").hidden = true;
    if (changed || CMP.data !== data) { CMP.data = data; renderCompare(); }
  } catch (error) {
    if (error.name === "AbortError" || CMP.request !== request) return;
    $("cmp-error").hidden = false;
    $("cmp-error").textContent = `对比数据读取失败：${error.message}`;
  }
}

const colValue = (run, col) => col.key === "train" ? run.train : col.key === "selection" ? run.selection : run.heldout?.[col.scale];
const scaleLabel = (task, key) => task === "online_bin_packing" ? key.replace("_", " · C") : `n=${key}`;
function columns(task) {
  const cols = [{key: "train", label: "训练", group: "train"}, {key: "selection", label: "选择", group: "selection"}];
  for (const s of task.scales) cols.push({key: `h:${s.key}`, scale: s.key, label: scaleLabel(task.key, s.key), group: s.test ? "test" : "gen"});
  return cols.filter(col => task.cohorts.some(c => c.runs.some(r => Number.isFinite(colValue(r, col)))));
}
function stat(values) {
  const v = values.filter(Number.isFinite);
  if (!v.length) return null;
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const sd = v.length > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1)) : null;
  return {n: v.length, mean, sd};
}
const relGap = (m, ref) => (m - ref) / Math.abs(ref) * 100;  // fitness space: positive = better
function delta(d) {
  if (d == null || !Number.isFinite(d)) return "";
  const cls = Math.abs(d) < 0.05 ? "same" : d > 0 ? "up" : "down";
  return `<em class="d ${cls}">${d > 0 ? "+" : ""}${d.toFixed(Math.abs(d) < 10 ? 2 : 1)}%</em>`;
}
function matrix(task, cols) {
  const M = {};
  for (const c of task.cohorts) {
    M[c.id] = {};
    for (const col of cols) M[c.id][col.key] = stat(c.runs.map(r => colValue(r, col)));
  }
  return M;
}

function renderCompare() {
  const data = CMP.data;
  if (!data || !data.tasks.length) {
    $("cmp-summary").innerHTML = `<div class="empty">${CMP.ids.length ? "所选对象没有可对比的结果" : "请添加对比对象"}</div>`;
    $("cmp-tasks").innerHTML = "";
    return;
  }
  const order = CMP.ids.filter(id => data.cohorts.some(c => c.id === id));
  const label = new Map(data.cohorts.map(c => [c.id, c.label]));
  const color = id => seriesColor(CMP.ids.indexOf(id));
  const who = id => `<div class="who"><span class="sw" style="--c:${color(id)}"></span><span title="${esc(id)}">${esc(label.get(id) || id)}</span>${id === CMP.ref ? '<span class="refmark">参照</span>' : ""}</div>`;

  // summary: average rank and mean relative gap per column group, across tasks
  const acc = Object.fromEntries(order.map(id => [id, Object.fromEntries(Object.keys(GROUP).map(g => [g, {ranks: [], rels: []}]))]));
  const perTask = data.tasks.map(task => {
    const cols = columns(task), M = matrix(task, cols);
    for (const col of cols) {
      const cells = order.filter(id => M[id]?.[col.key]).map(id => [id, M[id][col.key].mean]);
      if (cells.length >= 2) {
        const sorted = cells.map(([, m]) => m).sort((a, b) => b - a);
        for (const [id, m] of cells) {
          const first = sorted.indexOf(m), last = sorted.lastIndexOf(m);
          acc[id][col.group].ranks.push((first + last) / 2 + 1);
        }
      }
      const ref = M[CMP.ref]?.[col.key];
      if (ref) for (const [id, m] of cells) if (id !== CMP.ref) {
        const gap = relGap(m, ref.mean);
        if (Number.isFinite(gap)) acc[id][col.group].rels.push(gap);
      }
    }
    return {task, cols, M};
  });
  const groups = Object.keys(GROUP).filter(g => order.some(id => acc[id][g].ranks.length || acc[id][g].rels.length));
  const avg = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
  const bestRank = Object.fromEntries(groups.map(g => [g, Math.min(...order.map(id => avg(acc[id][g].ranks) ?? Infinity))]));
  $("cmp-summary").innerHTML = `
    <div class="panel-head"><h2>总览</h2><p>平均名次：各任务各列按均值排名后取平均（越小越好）；Δ：相对参照的平均相对差（按 fitness，正 = 更好）</p></div>
    <div class="tbl-wrap"><table class="cmp"><thead><tr><th>对象</th>${groups.map(g => `<th>${GROUP[g]}</th>`).join("")}</tr></thead><tbody>
    ${order.map(id => `<tr><td>${who(id)}</td>${groups.map(g => {
      const r = avg(acc[id][g].ranks), d = avg(acc[id][g].rels), n = acc[id][g].ranks.length;
      if (r == null && d == null) return '<td class="na">—</td>';
      return `<td class="${r != null && r === bestRank[g] ? "best" : ""}"><b>${r == null ? "—" : r.toFixed(2)}</b><small>${n} 项${id === CMP.ref ? " · 参照" : ""}</small>${id === CMP.ref ? "" : delta(d)}</td>`;
    }).join("")}</tr>`).join("")}
    </tbody></table></div>`;

  $("cmp-tasks").innerHTML = perTask.map(({task, cols, M}) => {
    if (!cols.length) {
      const rejected = task.cohorts.flatMap(c => c.runs).flatMap(r => Object.values(r.heldout_verification || {}))
        .filter(state => !["verified", "legacy"].includes(state)).length;
      return `<article class="panel"><div class="panel-head"><h2>${esc(task.label)}</h2></div><div class="empty">暂无可用成绩${rejected ? `；${rejected} 项测试结果未通过程序身份核验，已排除` : ""}</div></article>`;
    }
    const ids = order.filter(id => task.cohorts.some(c => c.id === id));
    const head1 = [], head2 = [];
    for (let i = 0; i < cols.length;) {
      const g = cols[i].group;
      let j = i; while (j < cols.length && cols[j].group === g) j++;
      if (g === "train" || g === "selection") head1.push(`<th rowspan="2">${GROUP[g]}</th>`);
      else {
        head1.push(`<th class="grp ${g}" colspan="${j - i}"><span>${GROUP[g]}</span></th>`);
        for (let k = i; k < j; k++) head2.push(`<th>${esc(cols[k].label)}</th>`);
      }
      i = j;
    }
    const best = Object.fromEntries(cols.map(col => {
      const means = ids.map(id => M[id][col.key]?.mean).filter(Number.isFinite);
      return [col.key, means.length >= 2 ? Math.max(...means) : null];
    }));
    const rows = ids.map(id => {
      const cohort = task.cohorts.find(c => c.id === id);
      return `<tr><td>${who(id)}</td>${cols.map(col => {
        const st = M[id][col.key];
        if (!st) return '<td class="na">—</td>';
        const ref = M[CMP.ref]?.[col.key];
        let note = "";
        if (col.key === "train" && cohort.runs.some(r => r.status === "running")) note = '<span class="flag" title="含未完成的运行：训练分仍会变化">运行中</span>';
        if (col.key === "selection") {
          const tied = cohort.runs.filter(r => r.finalists > 1 && r.ties === r.finalists).length;
          if (tied) note = `<span class="flag" title="${tied} 路的 top-5 选择分完全相同">${tied} 路并列</span>`;
        }
        return `<td class="${st.mean === best[col.key] ? "best" : ""}"><b>${fmt(shown(st.mean, task))}</b>${note}
          <small>${st.sd == null ? "" : `± ${fmt(st.sd, 3)} · `}n=${st.n}</small>${id !== CMP.ref && ref ? delta(relGap(st.mean, ref.mean)) : ""}</td>`;
      }).join("")}</tr>`;
    }).join("");
    const pick = cols.some(c => c.key === CMP.col[task.key]) ? CMP.col[task.key]
      : (cols.find(c => c.group === "test") || cols[0]).key;
    CMP.col[task.key] = pick;
    const sources = task.cohorts.filter(c => c.source).map(c => `${esc(label.get(c.id) || c.id)}：${esc(c.source)}`).join("；");
    const identityNotes = task.cohorts.map(cohort => {
      const checks = cohort.runs.flatMap(run => Object.values(run.heldout_verification || {}));
      const rejected = checks.filter(state => !["verified", "legacy"].includes(state)).length;
      const legacy = checks.filter(state => state === "legacy").length;
      const notes = [rejected ? `${rejected} 项测试结果与最终程序不一致或无法核验，已排除` : "",
        legacy ? `${legacy} 项历史测试结果未记录程序身份` : ""].filter(Boolean);
      return notes.length ? `${esc(label.get(cohort.id) || cohort.id)}：${notes.join("；")}` : "";
    }).filter(Boolean).join("；");
    return `<article class="panel cmp-task" data-task="${esc(task.key)}">
      <div class="panel-head"><h2>${esc(task.label)}</h2><p>原始${esc(task.unit)} · ${task.direction === "min" ? "越低越好" : "越高越好"}</p></div>
      <div class="tbl-wrap"><table class="cmp"><thead><tr><th rowspan="2">对象</th>${head1.join("")}</tr><tr>${head2.join("")}</tr></thead><tbody>${rows}</tbody></table></div>
      <div class="dots-head"><h3>各 rep 分布（向右 = 更好）</h3><div class="segctl small">${cols.map(col =>
        `<button type="button" data-col="${esc(col.key)}" aria-pressed="${col.key === pick}">${esc(col.label)}</button>`).join("")}</div></div>
      <svg class="dots"></svg>
      ${sources ? `<div class="src">held-out 来源 · ${sources}</div>` : ""}
      ${identityNotes ? `<div class="src">${identityNotes}</div>` : ""}
    </article>`;
  }).join("");
  drawAllDots();
}

function drawAllDots() {
  if (!CMP.data) return;
  for (const task of CMP.data.tasks) {
    const card = document.querySelector(`.cmp-task[data-task="${CSS.escape(task.key)}"]`);
    if (card) drawDots(card.querySelector("svg.dots"), task, CMP.col[task.key]);
  }
}

function drawDots(svg, task, colKey) {
  const col = columns(task).find(c => c.key === colKey);
  const ids = CMP.ids.filter(id => task.cohorts.some(c => c.id === id));
  const W = Math.max(320, Math.round(svg.parentElement.clientWidth - 32));
  const rowH = 26, top = 6, bottom = 24, L = Math.min(220, Math.round(W * 0.3)), R = 16;
  const H = top + ids.length * rowH + bottom;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("height", H);
  const series = ids.map(id => {
    const cohort = task.cohorts.find(c => c.id === id);
    const pts = col ? cohort.runs.map(r => ({v: colValue(r, col), rep: r.rep, name: r.name})).filter(p => Number.isFinite(p.v)) : [];
    return {id, pts, st: stat(pts.map(p => p.v)), label: CMP.cohorts.find(c => c.id === id)?.label || id};
  });
  const all = series.flatMap(s => s.pts.map(p => p.v));
  if (!all.length) { svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle">该列无数据</text>`; return; }
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.08 || Math.max(Math.abs(hi) * 0.01, 1e-6);
  lo -= pad; hi += pad;
  const x = v => L + (v - lo) / (hi - lo) * (W - L - R);
  const out = [];
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * i / 4, px = x(v);
    out.push(`<line class="axis" x1="${px}" x2="${px}" y1="${top}" y2="${H - bottom}"/>`,
      `<text x="${px}" y="${H - 8}" text-anchor="${i === 0 ? "start" : i === 4 ? "end" : "middle"}">${axisFmt(shown(v, task), hi - lo)}</text>`);
  }
  series.forEach((s, i) => {
    const cy = top + i * rowH + rowH / 2, c = cssVar(`--s${CMP.ids.indexOf(s.id) % 8 + 1}`);
    const text = s.label.length > 30 ? s.label.slice(0, 29) + "…" : s.label;
    out.push(`<text class="lab" x="0" y="${cy + 4}">${esc(text)}</text>`);
    if (s.st?.sd != null) out.push(`<line x1="${x(s.st.mean - s.st.sd)}" x2="${x(s.st.mean + s.st.sd)}" y1="${cy}" y2="${cy}" stroke="${c}" stroke-width="4" stroke-linecap="round" opacity=".28"/>`);
    for (const p of s.pts) out.push(`<circle cx="${x(p.v)}" cy="${cy}" r="4.5" fill="${c}" fill-opacity=".85"><title>${esc(s.label)} · Rep ${esc(p.rep ?? "?")}：${fmt(shown(p.v, task), 5)}</title></circle>`);
    if (s.st) out.push(`<line class="mean" x1="${x(s.st.mean)}" x2="${x(s.st.mean)}" y1="${cy - 8}" y2="${cy + 8}"><title>均值 ${fmt(shown(s.st.mean, task), 5)}</title></line>`);
  });
  svg.innerHTML = out.join("");
}

function setView(view) {
  S.view = view;
  $("view-monitor").hidden = view !== "monitor";
  $("view-compare").hidden = view !== "compare";
  document.querySelectorAll("[data-for=monitor]").forEach(el => el.hidden = view !== "monitor");
  $("live").hidden = view !== "monitor";
  document.querySelectorAll("[data-view]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.view === view)));
  if (view === "compare") {
    closeRun();
    (CMP.cohorts.length ? Promise.resolve() : loadCohorts()).then(() => loadCompare()).catch(error => {
      $("cmp-error").hidden = false; $("cmp-error").textContent = error.message;
    });
  } else {
    refresh();
    for (const chart of S.charts.values()) drawChart(chart);
  }
  writeHash();
}

/* ---------- events ---------- */
document.querySelectorAll("[data-view]").forEach(b => b.addEventListener("click", () => setView(b.dataset.view)));
$("cmp-chips").addEventListener("click", event => {
  const star = event.target.closest("[data-ref]"), remove = event.target.closest("[data-remove]");
  if (star) { CMP.ref = star.dataset.ref; renderChips(); writeHash(); renderCompare(); }
  if (remove) {
    CMP.ids = CMP.ids.filter(id => id !== remove.dataset.remove);
    if (!CMP.ids.includes(CMP.ref)) CMP.ref = CMP.ids[0] || null;
    loadCompare();
  }
});
$("cmp-chips").addEventListener("change", event => {
  if (event.target.id !== "cmp-add" || !event.target.value) return;
  CMP.ids.push(event.target.value);
  if (!CMP.ref) CMP.ref = CMP.ids[0];
  loadCompare();
});
$("cmp-tasks").addEventListener("click", event => {
  const button = event.target.closest("[data-col]");
  if (!button) return;
  const card = button.closest(".cmp-task"), key = card.dataset.task;
  CMP.col[key] = button.dataset.col;
  card.querySelectorAll("[data-col]").forEach(b => b.setAttribute("aria-pressed", String(b === button)));
  drawDots(card.querySelector("svg.dots"), CMP.data.tasks.find(t => t.key === key), CMP.col[key]);
});
let dotsFrame = 0;
window.addEventListener("resize", () => {
  if (S.view !== "compare") return;
  cancelAnimationFrame(dotsFrame);
  dotsFrame = requestAnimationFrame(drawAllDots);
});
$("tasks").addEventListener("click", event => {
  const row = event.target.closest(".run");
  if (row) openRun(row.closest(".task").dataset.task, row.dataset.run);
});
$("tasks").addEventListener("pointerover", event => {
  const row = event.target.closest(".run");
  const card = event.target.closest(".task");
  if (!card) return;
  const chart = S.charts.get(card.dataset.task);
  const focus = row ? [...card.querySelectorAll(".run")].indexOf(row) : null;
  if (chart && chart.focus !== focus) { chart.focus = focus; drawChart(chart); }
});
$("tasks").addEventListener("pointerleave", () => {
  for (const chart of S.charts.values()) if (chart.focus != null) { chart.focus = null; drawChart(chart); }
});
$("close").addEventListener("click", closeRun);
$("scrim").addEventListener("click", closeRun);
document.addEventListener("keydown", event => { if (event.key === "Escape") closeRun(); });
$("refresh").addEventListener("click", () => {
  if (S.view === "compare") loadCohorts().then(() => loadCompare({force: true}));
  else refresh({force: true});
});
$("batch").addEventListener("change", event => {
  closeRun();
  S.batch = event.target.value;
  writeHash();
  S.data = null; S.structure = "";
  $("tasks").innerHTML = '<div class="skeleton"></div><div class="skeleton"></div>';
  setLive("", "读取中");
  refresh();
});
function setMetric(metric) {
  S.metric = metric;
  try { localStorage.setItem("monitor.metric", metric); } catch {}
  document.querySelectorAll("[data-metric]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.metric === metric)));
  if (S.data) for (const task of S.data.tasks) updateTask(task);
  if (S.detail) renderDetail(S.detail);
  if (S.view === "compare" && CMP.data) renderCompare();
}
document.querySelectorAll("[data-metric]").forEach(b => b.addEventListener("click", () => setMetric(b.dataset.metric)));

const THEMES = ["auto", "light", "dark"];
function applyTheme(theme) {
  if (theme === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  $("theme").title = `主题：${{auto: "跟随系统", light: "浅色", dark: "深色"}[theme]}`;
  for (const chart of S.charts.values()) drawChart(chart);
  if (S.detailChart) drawChart(S.detailChart);
  if (S.view === "compare") drawAllDots();
}
let theme = (() => { try { return localStorage.getItem("monitor.theme") || "auto"; } catch { return "auto"; } })();
$("theme").addEventListener("click", () => {
  theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
  try { localStorage.setItem("monitor.theme", theme); } catch {}
  applyTheme(theme);
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme(theme));

/* ---------- polling: only while visible ---------- */
function startPolling() {
  stopPolling();
  S.timer = setInterval(() => {
    if (S.fetchedAt && !$("live").classList.contains("err")) $("live-text").textContent = `实时 · ${rel(S.fetchedAt)}`;
  }, 5000);
  let ticks = 0;
  // Comparison data changes slowly (finished runs, new held-out files): poll 4x less often.
  S.poll = setInterval(async () => {
    ticks++;
    if (ticks % 4 === 0 || !S.batch) {
      try { await loadBatches(); } catch {}
    }
    if (S.view === "compare") {
      if (ticks % 4 === 0) { try { await loadCohorts(); } catch {} loadCompare(); }
    }
    else refresh();
  }, REFRESH_MS);
}
function stopPolling() { clearInterval(S.timer); clearInterval(S.poll); }
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopPolling();
  else { S.view === "compare" ? loadCompare() : refresh(); startPolling(); }
});

(async function start() {
  applyTheme(theme);
  setMetric(S.metric === "fitness" ? "fitness" : "value");
  try {
    await loadBatches();
    const hash = hashParams();
    CMP.ids = (hash.get("c") || "").split(",").filter(Boolean);
    CMP.ref = hash.get("ref") || CMP.ids[0] || null;
    if (hash.get("v") === "compare") setView("compare");
    else {
      S.view = "monitor";
      await refresh();
      const linked = hash.get("r");
      if (linked && linked.includes("/")) openRun(linked.slice(0, linked.indexOf("/")), linked.slice(linked.indexOf("/") + 1));
    }
  } catch (error) {
    setLive("err", "初始化失败");
    showError(error.message);
  } finally {
    if (!document.hidden) startPolling();
  }
})();
