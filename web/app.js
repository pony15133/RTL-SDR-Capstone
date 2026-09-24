/* Station web interface. Plain JavaScript, no libraries; talks to dashboard.py's JSON API. */
"use strict";

const TOKEN = document.querySelector('meta[name="station-token"]').content;
const $ = (sel, root = document) => root.querySelector(sel);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const img = p => `/image?path=${encodeURIComponent(p)}`;
const fmtTime = s => s ? String(s).replace("T", " ").slice(0, 19) + " UTC" : "–";
const fmtShort = s => s ? String(s).replace("T", " ").slice(5, 16) : "–";
const mhz = hz => hz ? (hz / 1e6).toFixed(3) + " MHz" : "–";
const pct = v => v == null ? "–" : Math.round(100 * v) + "%";
const gb = n => n == null ? "–" : (n >= 1e9 ? (n / 1e9).toFixed(2) + " GB" : (n / 1e6).toFixed(1) + " MB");
const isMeteor = n => /METEOR/i.test(n || "");

const VERDICT = {
  detected: { label: "Satellite found", cls: "detected" },
  uncertain: { label: "Needs review", cls: "uncertain" },
  not_detected: { label: "Nothing found", cls: "not_detected" },
};
function verdictChip(c) {
  if (c.recording_status && c.recording_status !== "SUCCESS") return `<span class="chip failed">Recording ${esc(c.recording_status.toLowerCase())}</span>`;
  if (c.review_status === "signal") return `<span class="chip detected">Signal (reviewed)</span>`;
  if (c.review_status === "noise") return `<span class="chip not_detected">Noise (reviewed)</span>`;
  const v = VERDICT[c.detection_verdict];
  return v ? `<span class="chip ${v.cls}">${v.label}</span>` : `<span class="chip">–</span>`;
}
function confBar(v) {
  if (v == null) return `<span class="muted">–</span>`;
  const cls = v >= 0.7 ? "ok" : v >= 0.3 ? "warn" : "";
  return `<span class="conf"><span class="bar ${cls}"><i style="width:${Math.round(100 * v)}%"></i></span>${pct(v)}</span>`;
}
function countdown(iso) {
  const d = (new Date(iso) - Date.now()) / 1000;
  if (d <= 0) return "now";
  const h = Math.floor(d / 3600), m = Math.floor(d % 3600 / 60), s = Math.floor(d % 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${String(s).padStart(2, "0")}s` : `${s}s`;
}
function toast(msg, bad = false) {
  const t = $("#toast"); t.textContent = msg; t.className = "show" + (bad ? " bad" : "");
  clearTimeout(toast.timer); toast.timer = setTimeout(() => t.className = "", 3800);
}
async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json", "X-Station-Token": TOKEN }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(data.error || (data.problems || []).join(" ") || r.statusText), { data });
  return data;
}

/* ---------------------------------------------------------------- charts */
function skySvg(points, label = "") {
  const r = e => 100 * (90 - Math.max(0, e)) / 90;
  let g = "";
  for (const e of [0, 30, 60]) g += `<circle r="${r(e)}" fill="none" stroke="var(--line)"/><text x="3" y="${-r(e) + 10}">${e}°</text>`;
  g += `<line x1="-100" y1="0" x2="100" y2="0" stroke="var(--line)"/><line x1="0" y1="-100" x2="0" y2="100" stroke="var(--line)"/>`;
  g += `<text x="0" y="-104" text-anchor="middle">N</text><text x="106" y="3">E</text><text x="0" y="112" text-anchor="middle">S</text><text x="-114" y="3">W</text>`;
  const xy = p => { const a = p[0] * Math.PI / 180, rr = r(p[1]); return [rr * Math.sin(a), -rr * Math.cos(a)]; };
  const vis = (points || []).filter(p => p[1] >= 0).map(xy);
  if (vis.length > 1) {
    g += `<polyline fill="none" stroke="var(--accent)" stroke-width="2.5" points="${vis.map(p => p.join(",")).join(" ")}"/>`;
    g += `<circle cx="${vis[0][0]}" cy="${vis[0][1]}" r="4" fill="var(--ok)"/><circle cx="${vis.at(-1)[0]}" cy="${vis.at(-1)[1]}" r="4" fill="var(--bad)"/>`;
  } else {
    g += `<text x="0" y="4" text-anchor="middle">no track</text>`;
  }
  return `<svg class="sky" viewBox="-120 -118 240 236" role="img" aria-label="Sky track ${esc(label)}">${g}</svg>
    <div class="small muted row"><span><span style="color:var(--ok)">●</span> rises</span><span><span style="color:var(--bad)">●</span> sets</span><span>centre = overhead</span></div>`;
}
function lineSvg(xs, ys, { unit = "", w = 520, h = 160 } = {}) {
  if (xs.length < 2) return `<div class="muted small">No data</div>`;
  const pad = { l: 66, r: 10, t: 10, b: 22 };
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys, 0), y1 = Math.max(...ys, 0);
  const X = x => pad.l + (x - x0) / (x1 - x0 || 1) * (w - pad.l - pad.r);
  const Y = y => pad.t + (1 - (y - y0) / (y1 - y0 || 1)) * (h - pad.t - pad.b);
  let g = `<line x1="${pad.l}" x2="${w - pad.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--line)"/>`;
  for (const v of [y0, 0, y1]) g += `<text x="${pad.l - 6}" y="${Y(v) + 3}" text-anchor="end">${Math.round(v)}${unit}</text>`;
  g += `<text x="${pad.l}" y="${h - 4}">start</text><text x="${w - pad.r}" y="${h - 4}" text-anchor="end">end</text>`;
  g += `<polyline fill="none" stroke="var(--accent)" stroke-width="2" points="${xs.map((x, i) => X(x) + "," + Y(ys[i])).join(" ")}"/>`;
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img" aria-label="Doppler shift over the pass">${g}</svg>`;
}

/* ---------------------------------------------------------------- pages */
const pages = {};
let pollTimer = null, tickTimer = null;

pages.overview = async el => {
  const d = await api("/api/status");
  const L = d.live || {}, T = d.totals || {}, V = T.verdicts || {}, S = d.storage || {};
  const phase = L.phase || "unknown";
  const dotCls = phase === "recording" ? "rec" : ["stalled?", "not running", "unknown"].includes(phase) ? "warn" : "live";
  const cur = L.current || (L.upcoming || [])[0];
  const verb = L.current ? (phase === "recording" ? "Recording" : "Waiting for") : "Next pass";
  el.innerHTML = `
  <div class="pagehead"><div><h1>Overview</h1><p>${esc(L.station?.name || "Station")}${L.simulate ? " · simulation mode" : ""} · updated ${fmtTime(d.now_utc)}</p></div></div>
  ${(d.warnings || []).map(w => `<div class="alert ${w.level}">${esc(w.text)}${w.link ? `<a href="${w.link}">Open</a>` : ""}</div>`).join("")}
  <div class="grid">
    <section class="card s5">
      <h3>Right now</h3>
      <div class="hero"><span class="dot ${dotCls}"></span><div><div class="phase">${esc(phase)}</div>
        <div class="muted small">Recorder: ${esc(L.recorder_state || "–")}${L.heartbeat_age_s != null ? ` · heartbeat ${L.heartbeat_age_s}s ago` : ""}</div></div></div>
      ${cur ? `<div style="margin-top:16px"><div class="muted small">${verb}</div>
        <div class="row" style="justify-content:space-between"><div><b style="font-size:17px">${esc(cur.satellite)}</b>
        <div class="muted small">${mhz(cur.frequency_hz)} · max elevation ${cur.max_elevation_deg}° · ${cur.duration_min} min</div></div>
        <div class="right"><div class="big" data-countdown="${esc(cur.aos)}">${countdown(cur.aos)}</div><div class="muted small">until the pass starts</div></div></div></div>`
      : `<p class="muted" style="margin-top:14px">${esc(L.note || "No pass scheduled.")}</p>`}
    </section>
    <section class="card s7">
      <h3>Results</h3>
      <div class="tiles">
        <a class="tile ok" href="#/captures?verdict=detected"><b>${V.detected ?? 0}</b><span>Satellite found</span></a>
        <a class="tile warn" href="#/review"><b>${T.pending_review ?? 0}</b><span>Waiting for your review</span></a>
        <a class="tile" href="#/captures?verdict=not_detected"><b>${V.not_detected ?? 0}</b><span>Nothing found</span></a>
        <a class="tile ${T.failed ? "bad" : ""}" href="#/captures"><b>${T.failed ?? 0}</b><span>Recording failed</span></a>
        <a class="tile" href="#/images"><b>${T.decoded ?? 0}</b><span>METEOR images decoded</span></a>
        <a class="tile" href="#/captures"><b>${T.n ?? 0}</b><span>Recordings in total</span></a>
      </div>
    </section>
    <section class="card s7">
      <div class="row"><h3 style="margin:0">Latest captures</h3><span class="spacer"></span><a href="#/captures" class="small">All captures →</a></div>
      <div class="scroll" style="margin-top:8px">${captureTable((d.captures || []).slice(0, 6))}</div>
    </section>
    <section class="card s5">
      <h3>Latest waterfall</h3>
      ${d.latest_image ? `<img class="wf tall" src="${img(d.latest_image)}&t=${Date.now()}" alt="Latest Doppler-corrected waterfall"><p class="small muted" style="margin-top:6px">A satellite shows as a bright vertical line in the middle.</p>` : `<div class="empty"><b>No recordings yet</b>The first pass will appear here.</div>`}
    </section>
    <section class="card s4"><h3>${esc(d.sky?.label || "Sky track")}</h3>${skySvg(d.sky?.points)}</section>
    <section class="card s4">
      <h3>Storage</h3>
      <div class="row"><b>${S.free_gb} GB free</b><span class="muted">of ${S.total_gb} GB</span></div>
      <div class="bar ${S.free_gb < 10 ? "bad" : "ok"}" style="margin:8px 0 12px"><i style="width:${Math.max(2, 100 - 100 * S.free_gb / (S.total_gb || 1)).toFixed(0)}%"></i></div>
      <dl class="kv"><dt>Recordings</dt><dd>${S.recordings_gb} GB</dd><dt>Waiting for review</dt><dd>${S.uncertain_gb} GB</dd><dt>Rejected (archive)</dt><dd>${S.archived_gb} GB</dd></dl>
    </section>
    <section class="card s4">
      <h3>Detection models</h3>
      ${["waterfall", "iq"].map(k => { const m = d.models?.[k] || {}; return `<div style="margin-bottom:10px"><b>${k === "waterfall" ? "Waterfall model" : "IQ model"}</b>
        ${m.available ? `<span class="chip ok">ready</span><div class="small muted">${esc(m.version || "")}${m.test_f1 != null ? ` · test F1 ${m.test_f1.toFixed(2)}` : ""}${m.n_samples ? ` · ${m.n_samples} examples` : ""}${m.threshold != null ? ` · threshold ${m.threshold}` : ""}</div>` : `<span class="chip warn">not trained</span><div class="small muted">Run setup to train it.</div>`}</div>`; }).join("")}
    </section>
  </div>`;
  schedulePoll(10000);
};

function captureTable(rows) {
  if (!rows.length) return `<div class="empty"><b>No captures yet</b>They appear after the first recorded pass.</div>`;
  return `<table><thead><tr><th>#</th><th>When</th><th>Satellite</th><th>Result</th><th>Confidence</th><th>Raw IQ</th><th></th></tr></thead><tbody>
    ${rows.map(c => `<tr class="click" data-href="#/capture/${c.id}"><td>${c.id}</td><td>${fmtShort(c.actual_recording_start || c.timestamp_utc)}</td>
      <td>${esc(c.satellite_name || "–")}</td><td>${verdictChip(c)}</td><td>${confBar(c.decision_score ?? c.wf_ml_confidence_score ?? c.ml_confidence_score)}</td>
      <td class="small">${esc(c.iq_retention || "–")}</td><td>${String(c.decode_status || "").startsWith("decoded") ? '<span class="chip info">images</span>' : ""}</td></tr>`).join("")}
  </tbody></table>`;
}

pages.passes = async el => {
  const d = await api("/api/passes");
  const ups = d.upcoming || [];
  el.innerHTML = `<div class="pagehead"><div><h1>Upcoming passes</h1><p>Predicted from fresh orbit data (TLE). The station records each pass automatically, from 30 s before it rises to 30 s after it sets.</p></div></div>
  <div class="grid"><section class="card s8 flush"><div class="scroll">
    ${ups.length ? `<table><thead><tr><th>Satellite</th><th>Starts (UTC)</th><th>In</th><th class="num">Max elevation</th><th class="num">Length</th><th>Frequency</th><th></th></tr></thead><tbody>
    ${ups.map((p, i) => `<tr class="click ${i === 0 ? "sel" : ""}" data-i="${i}"><td><b>${esc(p.satellite)}</b></td><td>${fmtShort(p.aos)}</td><td data-countdown="${esc(p.aos)}">${countdown(p.aos)}</td>
      <td class="num"><span class="conf" style="justify-content:flex-end"><span class="bar ${p.max_elevation_deg >= 45 ? "ok" : p.max_elevation_deg >= 25 ? "" : "warn"}"><i style="width:${Math.round(100 * p.max_elevation_deg / 90)}%"></i></span>${p.max_elevation_deg}°</span></td>
      <td class="num">${p.duration_min} min</td><td>${mhz(p.frequency_hz)}</td><td class="small ${p.skipped_reason ? "warn" : "muted"}">${esc(p.skipped_reason || (p.max_elevation_deg >= 45 ? "great pass" : p.max_elevation_deg >= 25 ? "good pass" : "low pass"))}</td></tr>`).join("")}
    </tbody></table>` : `<div class="empty"><b>No schedule yet</b>Start run_station - the schedule appears here once passes are predicted.</div>`}
  </div></section>
  <section class="card s4"><h3 id="sky-label">${ups[0] ? esc(ups[0].satellite) : "Sky track"}</h3><div id="sky">${skySvg(ups[0]?.track)}</div>
    <p class="small muted">Higher passes (max elevation above ~30°) give much stronger signals.</p></section></div>`;
  el.querySelectorAll("tr[data-i]").forEach(tr => tr.onclick = () => {
    el.querySelectorAll("tr.sel").forEach(x => x.classList.remove("sel")); tr.classList.add("sel");
    const p = ups[+tr.dataset.i]; $("#sky-label").textContent = p.satellite; $("#sky").innerHTML = skySvg(p.track);
  });
  schedulePoll(60000);
};

pages.captures = async (el, params) => {
  const verdict = params.get("verdict") || "", sat = params.get("satellite") || "";
  let offset = 0;
  const load = async (append) => {
    const q = new URLSearchParams({ limit: 50, offset, ...(verdict && { verdict }), ...(sat && { satellite: sat }) });
    const d = await api("/api/captures?" + q);
    return d;
  };
  const d = await load();
  const filters = [["", "All"], ["detected", "Satellite found"], ["uncertain", "Needs review"], ["not_detected", "Nothing found"]];
  el.innerHTML = `<div class="pagehead"><div><h1>Captures</h1><p>Every recorded pass. Click one to see its waterfall, scores, track and decoded images.</p></div>
    <div class="row"><div class="segs">${filters.map(([v, l]) => `<button class="seg ${v === verdict ? "on" : ""}" data-v="${v}">${l}</button>`).join("")}</div>
    <select id="satf" aria-label="Satellite"><option value="">All satellites</option>${(d.satellites || []).map(s => `<option ${s === sat ? "selected" : ""}>${esc(s)}</option>`).join("")}</select></div></div>
    <section class="card flush"><div class="scroll" id="tbl">${captureTable(d.captures)}</div></section>
    <div class="row" style="margin-top:12px"><button id="more" class="${d.captures.length < 50 ? "hidden" : ""}">Load more</button></div>`;
  const go = (v, s) => location.hash = "#/captures?" + new URLSearchParams({ ...(v && { verdict: v }), ...(s && { satellite: s }) });
  el.querySelectorAll("[data-v]").forEach(b => b.onclick = () => go(b.dataset.v, sat));
  $("#satf").onchange = e => go(verdict, e.target.value);
  let rows = d.captures;
  $("#more").onclick = async () => { offset += 50; const n = await load(); rows = rows.concat(n.captures); $("#tbl").innerHTML = captureTable(rows); if (n.captures.length < 50) $("#more").classList.add("hidden"); };
};

pages.capture = async (el, params, id) => {
  const d = await api("/api/capture/" + id);
  const c = d.capture, track = d.track || [];
  const t0 = track.length ? new Date(track[0].timestamp_utc) : null;
  const dop = track.filter(p => p.doppler_hz != null);
  const reviewed = c.review_status === "signal" || c.review_status === "noise";
  el.innerHTML = `<div class="pagehead"><div><a href="#/captures" class="small">← Captures</a><h1>#${c.id} · ${esc(c.satellite_name || "Recording")}</h1>
      <p>${fmtTime(c.actual_recording_start || c.timestamp_utc)} · ${mhz(c.target_frequency_hz || c.frequency_hz)}</p></div>
    <div class="row">${verdictChip(c)}</div></div>
  <div class="grid">
    <section class="card s12">
      <div class="row"><div><b>${esc(c.retention_reason || c.notes || "")}</b><div class="small muted">Raw IQ: ${esc(c.iq_retention || "–")}${d.iq_size ? ` · ${gb(d.iq_size)}` : ""}${c.reviewed_by ? ` · reviewed by ${esc(c.reviewed_by)}` : ""}</div></div><span class="spacer"></span>
        <button class="good" data-label="1">${reviewed ? "Mark as" : "It's a"} signal</button><button class="bad" data-label="0">${reviewed ? "Mark as" : "It's"} noise</button>
        ${isMeteor(c.satellite_name) && d.iq_exists ? `<button id="decode" ${d.decoding || c.decode_status === "decoding" ? "disabled" : ""}>${d.decoding || c.decode_status === "decoding" ? "Decoding…" : "Decode METEOR images"}</button>` : ""}
        ${d.iq_exists ? `<a class="btn" href="/file?path=${encodeURIComponent(c.raw_iq_file_path)}">Download IQ</a>` : ""}</div>
      <p class="small muted" style="margin:8px 0 0">Your label always wins over the model, is saved with the capture and is added to the training data for the waterfall model.</p>
    </section>
    <section class="card s6"><h3>Doppler-corrected waterfall</h3>${c.waterfall_image_path ? `<img class="wf tall" src="${img(c.waterfall_image_path)}" alt="Waterfall">` : `<div class="empty">No waterfall</div>`}</section>
    <section class="card s6"><h3>Spectrogram</h3>${c.spectrogram_image_path ? `<img class="wf tall" src="${img(c.spectrogram_image_path)}" alt="Spectrogram">` : `<div class="empty">No spectrogram image</div>`}</section>
    <section class="card s4"><h3>Detector scores</h3>
      <dl class="kv"><dt>Waterfall model</dt><dd>${confBar(c.wf_ml_confidence_score)}</dd><dt>IQ model</dt><dd>${confBar(c.ml_confidence_score)}</dd>
      <dt>Rule detector</dt><dd>${c.rule_detection_result == null ? "–" : c.rule_detection_result ? "flagged" : "not flagged"} (${pct(c.rule_confidence_score)})</dd>
      <dt>Decision from</dt><dd>${esc(c.decision_source || "–")}</dd><dt>Verdict</dt><dd>${esc(c.detection_verdict || "–")}</dd>
      <dt>Model versions</dt><dd class="small">${esc([c.wf_model_version, c.model_version].filter(Boolean).join(", ") || "–")}</dd></dl></section>
    <section class="card s4"><h3>Recording</h3>
      <dl class="kv"><dt>Status</dt><dd>${esc(c.recording_status || "–")}</dd><dt>Tuned to</dt><dd>${mhz(c.frequency_hz)}</dd><dt>Downlink</dt><dd>${mhz(c.target_frequency_hz)}</dd>
      <dt>Sample rate</dt><dd>${c.sample_rate ? (c.sample_rate / 1e6).toFixed(3) + " Msps" : "–"}</dd><dt>Gain</dt><dd>${esc(c.gain ?? "auto")}</dd>
      <dt>Planned</dt><dd>${fmtShort(c.scheduled_aos)} → ${fmtShort(c.scheduled_los)}</dd><dt>Recorded</dt><dd>${fmtShort(c.actual_recording_start)} → ${fmtShort(c.actual_recording_stop)}</dd>
      <dt>Duration</dt><dd>${c.recording_duration_seconds ? Math.round(c.recording_duration_seconds) + " s" : "–"}</dd><dt>File size</dt><dd>${gb(c.output_file_size)}${c.expected_file_size ? ` (expected ${gb(c.expected_file_size)})` : ""}</dd>
      <dt>Doppler</dt><dd>${c.doppler_corrected ? `corrected, max ${Math.round(c.doppler_max_hz || 0)} Hz` : "not corrected"}</dd></dl></section>
    <section class="card s4"><h3>Pass track</h3>${skySvg(track.map(p => [p.azimuth_deg, p.elevation_deg]))}</section>
    <section class="card s6"><h3>Predicted Doppler shift</h3>${lineSvg(dop.map(p => (new Date(p.timestamp_utc) - t0) / 1000), dop.map(p => p.doppler_hz), { unit: " Hz" })}</section>
    <section class="card s6"><h3>METEOR images</h3>${d.decoded_images.length ? `<div class="gallery">${d.decoded_images.map(p => `<a href="${img(p)}" target="_blank" rel="noopener"><img loading="lazy" src="${img(p)}" alt=""><div class="cap">${esc(p.split(/[\\/]/).pop())}</div></a>`).join("")}</div>`
      : `<p class="muted">${esc(c.decode_status || (isMeteor(c.satellite_name) ? "Not decoded yet." : "Only METEOR passes carry LRPT images."))}</p>`}</section>
  </div>`;
  el.querySelectorAll("[data-label]").forEach(b => b.onclick = async () => {
    try { const r = await api(`/api/capture/${c.id}/label`, { label: +b.dataset.label, reviewer: reviewer() });
      toast(`Saved as ${b.dataset.label === "1" ? "signal" : "noise"} · raw IQ ${r.iq_action}`); render(); refreshBadge();
    } catch (e) { toast(e.message, true); }
  });
  const dec = $("#decode");
  if (dec) dec.onclick = async () => { try { const r = await api(`/api/capture/${c.id}/decode`, {}); toast(r.status); dec.disabled = true; dec.textContent = "Decoding…"; schedulePoll(8000); } catch (e) { toast(e.message, true); } };
  if (d.decoding || c.decode_status === "decoding") schedulePoll(8000);
};

function reviewer() {
  let n = ""; try { n = localStorage.getItem("reviewer") || ""; } catch (e) { /* storage blocked */ }
  if (!n) { n = (prompt("Your name (saved with your labels):") || "web").trim() || "web"; try { localStorage.setItem("reviewer", n); } catch (e) { /* ignore */ } }
  return n;
}

pages.review = async el => {
  const d = await api("/api/review");
  const q = d.pending || [];
  if (!q.length) {
    el.innerHTML = `<div class="pagehead"><div><h1>Review</h1><p>Captures the model wasn't sure about wait here, with their raw IQ kept.</p></div></div>
      <section class="card"><div class="empty"><b>Nothing to review</b>New uncertain captures will appear here automatically.</div></section>`;
    return;
  }
  let i = 0;
  const show = () => {
    const c = q[i];
    el.innerHTML = `<div class="pagehead"><div><h1>Review</h1><p>${q.length} capture(s) the model wasn't sure about. Is the satellite's trace visible?</p></div>
      <div class="small muted">Keys: <kbd>1</kbd> signal · <kbd>0</kbd> noise · <kbd>→</kbd> skip</div></div>
    <section class="card"><div class="review-card">
      <div>${c.waterfall_image_path ? `<img class="wf" src="${img(c.waterfall_image_path)}" alt="Waterfall to review">` : `<div class="empty">No waterfall image</div>`}</div>
      <div><h2>#${c.id} · ${esc(c.satellite_name || "")}</h2>
        <p class="muted">${fmtTime(c.actual_recording_start || c.timestamp_utc)} · ${mhz(c.target_frequency_hz || c.frequency_hz)}</p>
        <dl class="kv"><dt>Model confidence</dt><dd>${confBar(c.decision_score ?? c.wf_ml_confidence_score)}</dd><dt>Why</dt><dd>${esc(c.retention_reason || "")}</dd></dl>
        <div class="note">A real satellite after Doppler correction looks like a <b>bright line near the centre, running top to bottom</b> (sometimes faint or broken). Noise is an even speckle; interference is usually off-centre or perfectly constant.</div>
        <div class="review-actions"><button class="good" data-l="1">Signal <kbd>1</kbd></button><button class="bad" data-l="0">Noise <kbd>0</kbd></button><button data-skip>Skip <kbd>→</kbd></button><a class="btn" href="#/capture/${c.id}">Full details</a></div>
      </div></div>
      <div class="queue">${q.map((x, j) => `<a href="#" data-j="${j}" class="${j === i ? "cur" : ""}" title="#${x.id}">${x.waterfall_image_path ? `<img src="${img(x.waterfall_image_path)}" alt="">` : `#${x.id}`}</a>`).join("")}</div>
    </section>`;
    el.querySelectorAll("[data-l]").forEach(b => b.onclick = () => decide(+b.dataset.l));
    $("[data-skip]", el).onclick = () => { i = (i + 1) % q.length; show(); };
    el.querySelectorAll("[data-j]").forEach(a => a.onclick = e => { e.preventDefault(); i = +a.dataset.j; show(); });
  };
  const decide = async label => {
    const c = q[i];
    try {
      const r = await api(`/api/capture/${c.id}/label`, { label, reviewer: reviewer() });
      toast(`#${c.id} saved as ${label ? "signal" : "noise"} · raw IQ ${r.iq_action}`);
      q.splice(i, 1); refreshBadge();
      if (!q.length) return render();
      i = i % q.length; show();
    } catch (e) { toast(e.message, true); }
  };
  keyHandler = e => {
    if (e.target.matches("input, textarea, select")) return;
    if (e.key === "1") decide(1); else if (e.key === "0") decide(0); else if (e.key === "ArrowRight") { i = (i + 1) % q.length; show(); }
  };
  show();
};

pages.images = async el => {
  const d = await api("/api/images");
  const caps = d.captures || [];
  el.innerHTML = `<div class="pagehead"><div><h1>METEOR images</h1><p>Pictures decoded from kept METEOR-M passes with SatDump - proof the station received the satellite.</p></div></div>
    ${caps.length ? caps.map(c => `<section class="card" style="margin-bottom:16px"><div class="row"><h2 style="margin:0">#${c.id} · ${esc(c.satellite_name)}</h2><span class="muted small">${fmtTime(c.timestamp_utc)}</span><span class="spacer"></span><a href="#/capture/${c.id}" class="small">Capture →</a></div>
      <div class="gallery" style="margin-top:10px">${c.images.map(p => `<a href="${img(p)}" target="_blank" rel="noopener"><img loading="lazy" src="${img(p)}" alt=""><div class="cap">${esc(p.split(/[\\/]/).pop())}</div></a>`).join("")}</div></section>`).join("")
    : `<section class="card"><div class="empty"><b>No decoded images yet</b>Install SatDump (free, satdump.org) and keep METEOR-M2-3 / M2-4 in your satellite list. Kept METEOR passes are decoded automatically, or open a capture and press "Decode METEOR images".</div></section>`}`;
};

pages.settings = async el => {
  const d = await api("/api/config");
  const cfg = JSON.parse(JSON.stringify(d.config || {}));
  cfg.station = cfg.station || {}; cfg.recording = cfg.recording || {}; cfg.satellites = cfg.satellites || [];
  const R = cfg.recording, band = R.uncertain_band || [0.3, 0.7];
  const policies = [
    ["archive-negatives", "Archive", "Recordings with nothing found move to rejected/ - recoverable, deleted automatically only when the disk runs low."],
    ["delete-negatives", "Delete", "Recordings with nothing found are deleted to save space. The spectrogram, waterfall and database row are always kept."],
    ["keep-all", "Keep everything", "Never move or delete a recording (uses the most disk space)."],
  ];
  el.innerHTML = `<div class="pagehead"><div><h1>Settings</h1><p>Saved to <code>${esc(d.path || "(no file)")}</code>. Restart run_station after saving.</p></div>
    <button class="primary" id="save">Save settings</button></div>
  <div id="problems"></div>
  <div class="grid">
    <section class="card s6"><h2>Station</h2><div class="form-grid">
      <label class="field"><span>Name</span><input data-k="station.name" value="${esc(cfg.station.name || "")}"></label>
      <label class="field"><span>Latitude (°)</span><input data-k="station.lat_deg" type="number" step="0.0001" value="${esc(cfg.station.lat_deg ?? "")}"></label>
      <label class="field"><span>Longitude (°)</span><input data-k="station.lon_deg" type="number" step="0.0001" value="${esc(cfg.station.lon_deg ?? "")}"></label>
      <label class="field"><span>Height (m)</span><input data-k="station.alt_m" type="number" value="${esc(cfg.station.alt_m ?? 0)}"></label></div>
      <p class="small muted" style="margin-top:8px">Tip: in Google Maps, right-click your antenna's position and copy the coordinates.</p></section>
    <section class="card s6"><h2>Recording</h2><div class="form-grid">
      <label class="field"><span>Lowest pass to record (max elevation °)</span><input data-k="recording.min_elevation_deg" type="number" min="0" max="90" value="${esc(R.min_elevation_deg ?? 15)}"></label>
      <label class="field"><span>Sample rate</span><select data-k="recording.sample_rate" data-num>${[1024000, 1536000, 2048000, 2400000].map(v => `<option value="${v}" ${+R.sample_rate === v ? "selected" : ""}>${(v / 1e6).toFixed(3)} Msps</option>`).join("")}</select></label>
      <label class="field"><span>Start before / stop after pass (s)</span><div class="row" style="flex-wrap:nowrap"><input data-k="recording.pre_buffer" type="number" value="${esc(R.pre_buffer ?? 30)}"><input data-k="recording.post_buffer" type="number" value="${esc(R.post_buffer ?? 30)}"></div></label>
      <label class="field"><span>Recordings folder</span><input data-k="recording.output_dir" value="${esc(R.output_dir || "recordings")}"></label></div></section>
    <section class="card s6"><h2>When nothing is found</h2><div class="choice">
      ${policies.map(([v, l, t]) => `<label><input type="radio" name="ret" value="${v}" ${(R.retention || "archive-negatives") === v ? "checked" : ""}><span><b>${l}</b><span class="small muted">${t}</span></span></label>`).join("")}</div></section>
    <section class="card s6"><h2>How sure the model must be</h2>
      <p class="small muted">Captures with a confidence between the two numbers are <b>uncertain</b>: their raw IQ is always kept and they wait on the Review page, so a mistake by the model can never delete real data.</p>
      <div class="form-grid"><label class="field"><span>Below this = nothing found</span><input id="blo" type="number" min="0" max="1" step="0.05" value="${band[0]}"></label>
      <label class="field"><span>From this = satellite found</span><input id="bhi" type="number" min="0" max="1" step="0.05" value="${band[1]}"></label></div>
      <div class="bandview" id="bandview" style="margin-top:12px"></div>
      <label class="row small" style="margin-top:10px"><input type="checkbox" id="bandoff" ${R.uncertain_band === null ? "checked" : ""}> Turn off the uncertain level (yes/no only - not recommended)</label></section>
    <section class="card s12"><div class="row"><h2 style="margin:0">Satellites</h2><span class="spacer"></span><button id="addsat">+ Add satellite</button></div>
      <p class="small muted">Check a satellite is still transmitting on network.satnogs.org. METEOR-M2-3/M2-4 (137.9 MHz) are the best targets for an RTL-SDR.</p>
      <div class="scroll"><table><thead><tr><th>Name</th><th>NORAD id</th><th>Downlink (MHz)</th><th>Gain (dB)</th><th>Waterfall width (kHz)</th><th></th></tr></thead><tbody id="sats"></tbody></table></div></section>
    <section class="card s6"><h2>METEOR images</h2>
      <label class="row"><input type="checkbox" data-k="recording.decode" data-bool ${R.decode === false ? "" : "checked"}> Decode kept METEOR passes automatically (needs SatDump)</label>
      <label class="field" style="margin-top:10px"><span>SatDump program (leave empty to find it automatically)</span><input data-k="recording.satdump_path" value="${esc(R.satdump_path || "")}" placeholder="C:\\Program Files\\SatDump\\satdump.exe"></label></section>
    <section class="card s6"><h2>Running unattended</h2><div class="form-grid">
      <label class="field"><span>Always keep this much disk free (GB)</span><input data-k="recording.min_free_gb" type="number" min="0" step="0.5" value="${esc(R.min_free_gb ?? 2)}"></label>
      <label class="field"><span>Retries if the dongle is busy/unplugged</span><input data-k="recording.record_retries" type="number" min="0" max="5" value="${esc(R.record_retries ?? 2)}"></label></div>
      <label class="row small" style="margin-top:10px"><input type="checkbox" data-k="recording.prune_rejected" data-bool ${R.prune_rejected === false ? "" : "checked"}> When the disk is low, delete the oldest rejected recordings first</label></section>
  </div>`;
  const drawSats = () => {
    $("#sats").innerHTML = cfg.satellites.map((s, i) => `<tr><td><input data-s="${i}" data-f="name" value="${esc(s.name)}"></td><td><input data-s="${i}" data-f="norad_id" type="number" value="${esc(s.norad_id)}"></td>
      <td><input data-s="${i}" data-f="frequency_hz" data-mhz type="number" step="0.001" value="${s.frequency_hz ? (s.frequency_hz / 1e6).toFixed(3) : ""}"></td>
      <td><input data-s="${i}" data-f="gain" type="number" step="0.1" value="${esc(s.gain ?? 30)}"></td>
      <td><input data-s="${i}" data-f="waterfall_span_hz" data-khz type="number" value="${s.waterfall_span_hz ? s.waterfall_span_hz / 1000 : 48}"></td>
      <td><button data-del="${i}" aria-label="Remove">Remove</button></td></tr>`).join("");
    el.querySelectorAll("[data-del]").forEach(b => b.onclick = () => { readSats(); cfg.satellites.splice(+b.dataset.del, 1); drawSats(); });
  };
  const readSats = () => el.querySelectorAll("[data-s]").forEach(inp => {
    const s = cfg.satellites[+inp.dataset.s], f = inp.dataset.f; let v = inp.value;
    if (inp.hasAttribute("data-mhz")) v = Math.round(parseFloat(v) * 1e6);
    else if (inp.hasAttribute("data-khz")) v = Math.round(parseFloat(v) * 1000);
    else if (f === "norad_id") v = parseInt(v, 10);
    else if (f === "gain") v = parseFloat(v);
    s[f] = v;
  });
  const drawBand = () => {
    const lo = +$("#blo").value, hi = +$("#bhi").value, off = $("#bandoff").checked;
    $("#bandview").innerHTML = off ? `<div class="n" style="width:50%">nothing</div><div class="d" style="width:50%">found</div>`
      : `<div class="n" style="width:${lo * 100}%">${lo > 0.12 ? "nothing" : ""}</div><div class="u" style="width:${(hi - lo) * 100}%">${hi - lo > 0.12 ? "review" : ""}</div><div class="d" style="width:${(1 - hi) * 100}%">${1 - hi > 0.12 ? "found" : ""}</div>`;
  };
  drawSats(); drawBand();
  ["blo", "bhi", "bandoff"].forEach(id => $("#" + id).oninput = drawBand);
  $("#addsat").onclick = () => { readSats(); cfg.satellites.push({ name: "", norad_id: "", frequency_hz: 137900000, gain: 30, waterfall_span_hz: 48000 }); drawSats(); };
  $("#save").onclick = async () => {
    readSats();
    el.querySelectorAll("[data-k]").forEach(inp => {
      const [a, b] = inp.dataset.k.split(".");
      let v = inp.type === "checkbox" ? inp.checked : inp.value;
      if (inp.type === "number" || inp.hasAttribute("data-num")) v = v === "" ? null : Number(v);
      if (b === "satdump_path" && !v) v = null;
      cfg[a][b] = v;
    });
    cfg.recording.retention = el.querySelector('input[name="ret"]:checked').value;
    cfg.recording.uncertain_band = $("#bandoff").checked ? null : [+$("#blo").value, +$("#bhi").value];
    try { const r = await api("/api/config", { config: cfg }); $("#problems").innerHTML = ""; toast(r.message); }
    catch (e) { $("#problems").innerHTML = (e.data?.problems || [e.message]).map(p => `<div class="alert bad">${esc(p)}</div>`).join(""); toast("Not saved - see the problems above", true); }
  };
};

pages.health = async el => {
  const d = await api("/api/health");
  el.innerHTML = `<div class="pagehead"><div><h1>Health</h1><p>Everything the station needs, and what it has been doing.</p></div><button id="rel">Check again</button></div>
  <div class="grid">
    <section class="card s5"><h2>Checks</h2>${d.checks.map(c => `<div class="check"><span class="ic ${c.ok ? "ok" : "bad"}">${c.ok ? "✓" : "✗"}</span><div><b>${esc(c.name)}</b><div class="small muted">${esc(c.detail)}</div></div></div>`).join("")}</section>
    <section class="card s7 flush"><div style="padding:16px 16px 0"><h2>Station events</h2></div><div class="scroll" style="max-height:440px">
      ${d.status_log.length ? `<table><thead><tr><th>Time (UTC)</th><th>Part</th><th>Event</th><th>Details</th></tr></thead><tbody>${d.status_log.map(e => `<tr><td>${fmtShort(e.timestamp_utc)}</td><td>${esc(e.component)}</td><td><span class="chip ${/ERROR|FAIL|FULL/.test(e.state) ? "bad" : /RETRY|SKIP|PRUNED/.test(e.state) ? "warn" : ""}">${esc(e.state)}</span></td><td class="wrap small">${esc(e.message)}</td></tr>`).join("")}</tbody></table>` : `<div class="empty">No events yet</div>`}
    </div></section>
    <section class="card s12"><h2>Log file <span class="muted small">(logs/station.log, last 200 lines)</span></h2>${d.log_tail ? `<pre class="log">${esc(d.log_tail)}</pre>` : `<p class="muted">No log yet - it is written while run_station is running.</p>`}</section>
  </div>`;
  $("#rel").onclick = render;
};

/* ---------------------------------------------------------------- router */
let keyHandler = null;
document.addEventListener("keydown", e => keyHandler && keyHandler(e));

function schedulePoll(ms) { clearTimeout(pollTimer); pollTimer = setTimeout(() => { if (document.visibilityState === "visible") render(true); else schedulePoll(ms); }, ms); }

async function render(quiet) {
  clearTimeout(pollTimer); keyHandler = null;
  const [path, query] = (location.hash.slice(1) || "/").split("?");
  const parts = path.split("/").filter(Boolean);
  const name = parts[0] || "overview";
  const page = pages[name] ? name : "overview";
  document.querySelectorAll("nav.side a").forEach(a => a.classList.toggle("active", a.dataset.page === (page === "capture" ? "captures" : page)));
  const el = $("#page");
  if (!quiet) el.innerHTML = `<div class="empty"><b>Loading…</b></div>`;
  try {
    const scroll = window.scrollY;
    await pages[page](el, new URLSearchParams(query || ""), parts[1]);
    if (quiet) window.scrollTo(0, scroll); else el.focus({ preventScroll: true });
  } catch (e) {
    el.innerHTML = `<div class="alert bad">Couldn't load this page: ${esc(e.message)}. Is dashboard.py still running?</div>`;
  }
}

async function refreshBadge() {
  try {
    const d = await api("/api/review");
    const n = (d.pending || []).length, b = $("#review-count");
    b.textContent = n; b.classList.toggle("hidden", !n);
  } catch (e) { /* offline */ }
}
async function refreshBrand() {
  try { const d = await api("/api/config"); const st = d.config?.station || {};
    if (st.name) $("#station-name").textContent = st.name;
    if (st.lat_deg != null) $("#station-sub").textContent = `${(+st.lat_deg).toFixed(3)}°, ${(+st.lon_deg).toFixed(3)}°`;
  } catch (e) { /* ignore */ }
}

tickTimer = setInterval(() => {
  document.querySelectorAll("[data-countdown]").forEach(n => n.textContent = countdown(n.dataset.countdown));
  $("#clock").textContent = "UTC " + new Date().toISOString().slice(11, 19);
}, 1000);
window.addEventListener("hashchange", () => render());
$("#page").addEventListener("click", e => { const tr = e.target.closest("tr[data-href]"); if (tr && !e.target.closest("a, button")) location.hash = tr.dataset.href; });
render(); refreshBadge(); refreshBrand();
setInterval(refreshBadge, 30000);
