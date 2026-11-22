# AI-assisted: generated with Claude Code, reviewed and run by the author
"""Animated 3D map of Sri Lanka, drawn with deck.gl inside a Streamlit HTML component.

The island is built from the 9 provinces (geoBoundaries, CC BY 4.0), lifted into a thin
slab. Each city is a glowing column whose height and colour both show the chosen measure,
so the value never depends on colour alone. Browsers without WebGL get a lighter SVG
version of the same scene. The view slowly turns, the columns rise on
load, and both stop when the viewer asks for reduced motion or presses Pause.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = Path(__file__).resolve().parent / "assets"
DECK_URL = "https://unpkg.com/deck.gl@9.4.1/dist.min.js"

# one-hue ramps (dark end = low value, so small values sink into the dark surface)
RAMPS = {
    "rain": ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"],
    "temp": ["#5a2410", "#9a3d17", "#d95926", "#f0936a", "#ffd2bd"],
    "elev": ["#0b3d2e", "#126b4f", "#199e70", "#5fcca2", "#c4f1de"],
}
# light theme: the same hues stepped the other way (light = low), starting above the white surface
RAMPS_LIGHT = {
    "rain": ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
    "temp": ["#f7b494", "#eb6834", "#c4501f", "#9a3d17", "#5a2410"],
    "elev": ["#8fdcbd", "#1baf7a", "#14875e", "#0f6a4a", "#0b3d2e"],
}
MEASURES = {
    "rain": ("Annual rainfall", "mm / yr", 0),
    "temp": ("Mean temperature", "°C", 1),
    "elev": ("Elevation", "m", 0),
}
LABELLED = {"Colombo", "Kandy", "Jaffna", "Trincomalee", "Galle", "Hatton", "Hambantota",
            "Ratnapura", "Batticaloa", "Kalmunai", "Badulla", "Mannar", "Puttalam"}


def _hex(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def _ramp(key, t, ramps=RAMPS):
    stops = [_hex(c) for c in ramps[key]]
    t = float(np.clip(t, 0, 1)) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    c = stops[i] + (stops[i + 1] - stops[i]) * (t - i)
    return [int(round(v)) for v in c]


def city_table(daily: pd.DataFrame) -> pd.DataFrame:
    cs = daily.groupby("city").agg(
        lat=("latitude", "first"), lon=("longitude", "first"), elev=("elevation", "first"),
        rain=("precipitation_sum", lambda s: s.mean() * 365.25),
        temp=("temperature_2m_mean", "mean")).reset_index()
    # The data rounds coordinates to 0.1°, so several Colombo-area cities sit on one point.
    # Spread each stack around a small ring so every column can be seen and hovered.
    cs["plot_lon"], cs["plot_lat"] = cs.lon.astype(float), cs.lat.astype(float)
    for _, grp in cs.groupby(["lat", "lon"]):
        if len(grp) > 1:
            ang = np.linspace(0, 2 * np.pi, len(grp), endpoint=False)
            cs.loc[grp.index, "plot_lon"] += 0.075 * np.cos(ang)
            cs.loc[grp.index, "plot_lat"] += 0.075 * np.sin(ang)
    return cs


def _payload(cs: pd.DataFrame) -> dict:
    out, scales = [], {}
    for key in MEASURES:
        lo, hi = float(cs[key].min()), float(cs[key].max())
        scales[key] = {"min": lo, "max": hi, "ramp": RAMPS[key], "rampL": RAMPS_LIGHT[key]}
    for r in cs.itertuples():
        row = {"city": r.city, "pos": [round(r.plot_lon, 4), round(r.plot_lat, 4)],
               "label": r.city in LABELLED, "v": {}, "h": {}, "c": {}, "cl": {}}
        for key in MEASURES:
            lo, hi = scales[key]["min"], scales[key]["max"]
            t = (getattr(r, key) - lo) / (hi - lo) if hi > lo else 0.5
            row["v"][key] = round(float(getattr(r, key)), 1)
            row["h"][key] = round(0.12 + 0.88 * t, 4)   # never zero, so every city stays visible
            row["c"][key] = _ramp(key, t)
            row["cl"][key] = _ramp(key, t, RAMPS_LIGHT)
        out.append(row)
    return {"cities": out, "scales": scales,
            "measures": {k: {"name": v[0], "unit": v[1], "dp": v[2]} for k, v in MEASURES.items()}}


def island_html(daily: pd.DataFrame, height: int = 600) -> str:
    provinces = json.loads((ASSETS / "sri_lanka_provinces.json").read_text())
    data = _payload(city_table(daily))
    data["provinces"] = provinces
    blob = json.dumps(data, separators=(",", ":"))
    return TEMPLATE.replace("__DATA__", blob).replace("__DECK__", DECK_URL).replace("__H__", str(height))


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<script src="__DECK__"></script>
<style>
  :root{--ink:#e6ecf5;--ink2:#a9b6ca;--muted:#7f8ca3;--line:rgba(148,163,184,.22);--glass:rgba(10,16,32,.72);--accent:#38bdf8}
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:transparent;font-family:Inter,system-ui,sans-serif;color:var(--ink)}
  #wrap{position:relative;height:__H__px;border-radius:18px;overflow:hidden;
    background:radial-gradient(120% 90% at 50% 10%,#10213f 0%,#0a1226 45%,#060a16 100%);
    border:1px solid var(--line)}
  #wrap::before{content:"";position:absolute;inset:0;pointer-events:none;
    background-image:linear-gradient(rgba(56,189,248,.06) 1px,transparent 1px),linear-gradient(90deg,rgba(56,189,248,.06) 1px,transparent 1px);
    background-size:44px 44px;mask-image:radial-gradient(70% 70% at 50% 55%,#000 30%,transparent 80%)}
  #map{position:absolute;inset:0}
  .panel{position:absolute;background:var(--glass);border:1px solid var(--line);border-radius:14px;
    backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);padding:12px 14px}
  #ctrl{top:14px;left:14px;display:flex;flex-direction:column;gap:10px;max-width:270px}
  .eyebrow{font:600 11px/1 'Space Grotesk',sans-serif;letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
  .title{font:600 17px/1.25 'Space Grotesk',sans-serif;margin:0}
  .seg{display:flex;gap:4px;background:rgba(255,255,255,.04);border:1px solid var(--line);border-radius:10px;padding:3px}
  button{font:500 12.5px Inter,sans-serif;color:var(--ink2);background:transparent;border:0;border-radius:8px;
    padding:7px 10px;cursor:pointer;min-height:32px}
  button:hover{color:var(--ink);background:rgba(255,255,255,.06)}
  button[aria-pressed="true"]{color:#06101f;background:var(--accent)}
  button:focus-visible{outline:2px solid #fff;outline-offset:2px}
  .row{display:flex;gap:6px}
  .ghost{border:1px solid var(--line)}
  #legend{bottom:14px;right:14px;width:240px;padding:10px 12px}
  #legend .bar{height:10px;border-radius:5px;margin:8px 0 4px}
  #legend .ends{display:flex;justify-content:space-between;font-size:12px;color:var(--ink2)}
  #legend .note{font-size:11.5px;color:var(--muted);margin-top:6px;line-height:1.4}
  .tip{font:13px Inter,sans-serif;line-height:1.45}
  .tip b{font-family:'Space Grotesk',sans-serif;font-size:14px;color:#fff}
  html[data-theme="light"]{--ink:#0f172a;--ink2:#334155;--muted:#5b6678;--line:#d5e0ec;--glass:rgba(255,255,255,.82);--accent:#0a72c2}
  html[data-theme="light"] #wrap{background:radial-gradient(120% 90% at 50% 0%,#ffffff 0%,#eef6ff 50%,#e3effb 100%)}
  html[data-theme="light"] #wrap::before{background-image:linear-gradient(rgba(10,114,194,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(10,114,194,.07) 1px,transparent 1px)}
  html[data-theme="light"] .seg{background:#f1f6fb}
  html[data-theme="light"] button:hover{background:rgba(15,23,42,.05)}
  html[data-theme="light"] button[aria-pressed="true"]{color:#fff}
  html[data-theme="light"] .tip b{color:#0f172a}
  html[data-theme="light"] .panel{box-shadow:0 8px 24px rgba(15,23,42,.08)}
  button:focus-visible{outline-color:var(--accent)}
  @media (max-width:640px){#ctrl{max-width:calc(100% - 28px)} #legend{width:calc(100% - 28px)}}
</style></head>
<body>
<div id="wrap">
  <div id="map" role="img" aria-label="Rotating 3D map of Sri Lanka. Each of the 30 cities is a column whose height and colour show the selected measure. A table of the same values is below the map."></div>
  <div id="ctrl" class="panel">
    <div><div class="eyebrow">30 cities · 2010–2023</div><p class="title" id="ttl" aria-live="polite">Annual rainfall</p></div>
    <div class="seg" role="group" aria-label="Measure shown by the columns">
      <button data-m="rain" aria-pressed="true">Rainfall</button>
      <button data-m="temp" aria-pressed="false">Temperature</button>
      <button data-m="elev" aria-pressed="false">Elevation</button>
    </div>
    <div class="row">
      <button id="play" class="ghost" aria-pressed="true">Pause rotation</button>
      <button id="reset" class="ghost">Reset view</button>
    </div>
  </div>
  <div id="legend" class="panel" aria-hidden="true">
    <div class="eyebrow" id="lgname">Annual rainfall · mm / yr</div>
    <div class="bar" id="lgbar"></div>
    <div class="ends"><span id="lgmin"></span><span id="lgmax"></span></div>
  </div>
</div>
<script>
const D = __DATA__;
const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let measure = 'rain', spinning = !reduce, grow = reduce ? 1 : 0, t0 = performance.now(), redraw = () => {};
const HOME = {longitude: 80.72, latitude: 7.72, zoom: 7.05, pitch: 52, bearing: -18, maxPitch: 75, minZoom: 6, maxZoom: 10};
let view = Object.assign({}, HOME, reduce ? {} : {zoom: 6.4, pitch: 20, bearing: -60});
const PROV = {dark: ['#13233f','#162a4a','#122039','#182d50','#142644','#11203b','#172b4c','#152846','#13243f'],
              light: ['#ffffff','#f3f8fd','#eaf3fc','#f7fbff','#eef5fc','#e6f0fa','#f5f9fe','#edf4fb','#e9f2fb']};
const pidx = {}; let pi = 0;
D.provinces.forEach(p => { if(!(p.name in pidx)) pidx[p.name] = (pi++) % 9; });
function detectTheme(){
  try { const t = window.parent.document.documentElement.dataset.slTheme; if (t) return t; } catch (e) {}
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}
let theme = detectTheme(); document.documentElement.dataset.theme = theme;
const L = () => theme === 'light';
const pal = name => PROV[theme][pidx[name]];
const colOf = (c) => (L() ? c.cl : c.c)[measure];
setInterval(() => { const t = detectTheme(); if (t !== theme) { theme = t; document.documentElement.dataset.theme = t; setLegend(); redraw(); } }, 400);
const rgb = h => [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
const css = c => `rgb(${c[0]},${c[1]},${c[2]})`;
const fmt = (v, m) => v.toLocaleString('en-US', {minimumFractionDigits: D.measures[m].dp, maximumFractionDigits: D.measures[m].dp});
const ease = x => 1 - Math.pow(1 - x, 3);
const tipHtml = o => `<div class="tip"><b>${o.city}</b><br>Rainfall ${fmt(o.v.rain,'rain')} mm/yr<br>Mean temp ${fmt(o.v.temp,'temp')} °C<br>Elevation ${fmt(o.v.elev,'elev')} m</div>`;

function setLegend(){
  const s = D.scales[measure], m = D.measures[measure];
  document.getElementById('ttl').textContent = m.name;
  document.getElementById('lgname').textContent = `${m.name} · ${m.unit}`;
  document.getElementById('lgbar').style.background = `linear-gradient(90deg, ${(L() ? s.rampL : s.ramp).join(',')})`;
  document.getElementById('lgmin').textContent = `${fmt(s.min, measure)} ${m.unit}`;
  document.getElementById('lgmax').textContent = `${fmt(s.max, measure)} ${m.unit}`;
}
function setSpin(on){
  spinning = on;
  const b = document.getElementById('play');
  b.setAttribute('aria-pressed', String(on));
  b.textContent = on ? 'Pause rotation' : 'Play rotation';
}
document.querySelectorAll('.seg button').forEach(b => b.addEventListener('click', () => {
  measure = b.dataset.m;
  document.querySelectorAll('.seg button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  setLegend(); redraw();
}));
document.getElementById('play').addEventListener('click', () => setSpin(!spinning));
document.getElementById('reset').addEventListener('click', () => { view = Object.assign({}, HOME); redraw(); });
if (reduce) setSpin(false);
setLegend();

// intro camera move + slow turn, shared by both renderers
const start = Object.assign({}, view);
function tick(now){
  const k = Math.min(1, (now - t0) / 2200);
  if (!reduce && k < 1) {
    const e = ease(k);
    view = Object.assign({}, view, {zoom: start.zoom + (HOME.zoom - start.zoom)*e,
      pitch: start.pitch + (HOME.pitch - start.pitch)*e, bearing: start.bearing + (HOME.bearing - start.bearing)*e});
    grow = e;
  } else {
    grow = 1;
    if (spinning) view = Object.assign({}, view, {bearing: (view.bearing + 0.06) % 360});
  }
}

function gl_ok(){ try{ const c=document.createElement('canvas'); return !!(c.getContext('webgl2')||c.getContext('webgl')); }catch(e){ return false; } }
let mode = 'svg';
try { if (window.deck && gl_ok()) { startDeck(); mode = 'webgl'; } } catch (e) { console.warn('deck.gl failed, using SVG', e); document.getElementById('map').innerHTML = ''; }
if (mode === 'svg') startSvg();
document.getElementById('wrap').dataset.renderer = mode;

// ---------------------------------------------------------------- WebGL renderer (deck.gl)
function startDeck(){
  const {Deck, PolygonLayer, ColumnLayer, ScatterplotLayer, TextLayer, LightingEffect, AmbientLight, DirectionalLight} = deck;
  const lighting = new LightingEffect({
    a: new AmbientLight({color: [255,255,255], intensity: 1.0}),
    d1: new DirectionalLight({color: [180,220,255], intensity: 1.6, direction: [-3,-8,-6]}),
    d2: new DirectionalLight({color: [255,200,170], intensity: 0.6, direction: [4,6,-3]})
  });
  const tipStyle = () => L() ? {background: 'rgba(255,255,255,.97)', border: '1px solid #cfe0f1', borderRadius: '10px', padding: '9px 11px', color: '#0f172a', boxShadow: '0 8px 24px rgba(15,23,42,.12)'}
                            : {background: 'rgba(8,13,28,.92)', border: '1px solid rgba(56,189,248,.45)', borderRadius: '10px', padding: '9px 11px', color: '#e6ecf5'};
  function layers(now){
    const pulse = reduce ? 0.5 : (Math.sin(now/700) + 1) / 2;
    return [
      new PolygonLayer({id:'island', data: D.provinces, getPolygon: d => d.polygon, extruded: true,
        getElevation: 2600, filled: true, stroked: true,
        getFillColor: d => rgb(pal(d.name)).concat(255), getLineColor: L() ? [10,114,194,170] : [56,189,248,150], lineWidthMinPixels: 1,
        updateTriggers: {getFillColor: theme, getLineColor: theme},
        material: {ambient: .55, diffuse: .6, shininess: 24, specularColor: [90,140,200]}, pickable: true,
        autoHighlight: true, highlightColor: [56,189,248,60]}),
      new ScatterplotLayer({id:'halo', data: D.cities, getPosition: d => d.pos.concat(2700),
        getRadius: 5200 + 2600*pulse, radiusUnits: 'meters', stroked: true, filled: false,
        getLineColor: d => colOf(d).concat(Math.round(70 + 90*(1-pulse))), lineWidthMinPixels: 1.5,
        updateTriggers: {getRadius: now, getLineColor: [measure, now, theme]}, parameters: {depthTest: false}}),
      new ColumnLayer({id:'cols', data: D.cities, getPosition: d => d.pos, diskResolution: 24, radius: 3300,
        extruded: true, elevationScale: grow, getElevation: d => 2700 + 88000*d.h[measure],
        getFillColor: d => colOf(d).concat(245), material: {ambient: .7, diffuse: .55, shininess: 60},
        pickable: true, autoHighlight: true, highlightColor: [255,255,255,90],
        updateTriggers: {getElevation: measure, getFillColor: [measure, theme]},
        transitions: {getElevation: {duration: reduce ? 0 : 900}, getFillColor: {duration: reduce ? 0 : 600}}}),
      new TextLayer({id:'labels', data: D.cities.filter(d => d.label),
        getPosition: d => d.pos.concat((2700 + 88000*d.h[measure])*grow + 4000),
        getText: d => d.city, getSize: 12, getColor: L() ? [15,23,42,240] : [230,236,245,235], fontFamily: 'Inter, sans-serif',
        fontWeight: 500, getPixelOffset: [0,-6], billboard: true, outlineWidth: 3, outlineColor: L() ? [255,255,255,235] : [6,10,22,230],
        fontSettings: {sdf: true}, updateTriggers: {getPosition: [measure, grow], getColor: theme}, parameters: {depthTest: false}})
    ];
  }
  let rendered = false, dead = false;
  function fallback(){
    if (dead) return; dead = true;
    try { dk.finalize(); } catch (e) {}
    document.getElementById('map').innerHTML = '';
    document.getElementById('wrap').dataset.renderer = 'svg';
    startSvg();
  }
  setTimeout(() => { if (!rendered) fallback(); }, 4000);
  const dk = new Deck({
    parent: document.getElementById('map'), viewState: view, controller: {inertia: 400, scrollZoom: false},
    effects: [lighting], layers: layers(performance.now()),
    onAfterRender: () => { rendered = true; },
    onError: e => { console.warn('deck.gl error, switching to SVG', e); fallback(); },
    onViewStateChange: ({viewState, interactionState}) => {
      view = viewState;
      if (interactionState && (interactionState.isDragging || interactionState.isZooming || interactionState.isRotating)) setSpin(false);
      dk.setProps({viewState: view});
    },
    getTooltip: ({object, layer}) => !object ? null :
      {html: layer.id === 'island' ? `<div class="tip"><b>${object.name}</b></div>` : tipHtml(object), style: tipStyle()}
  });
  redraw = () => dk.setProps({viewState: view, layers: layers(performance.now())});
  function frame(now){ if (dead) return; tick(now); dk.setProps({viewState: view, layers: layers(now)}); if (!reduce) requestAnimationFrame(frame); }
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- SVG renderer (no WebGL needed)
function startSvg(){
  const wrap = document.getElementById('wrap'), host = document.getElementById('map');
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg'); svg.setAttribute('width', '100%'); svg.setAttribute('height', '100%');
  svg.style.cursor = 'grab'; svg.style.touchAction = 'none';
  host.appendChild(svg);
  const tip = document.createElement('div'); tip.className = 'svgtip'; tip.style.cssText = 'position:absolute;pointer-events:none;display:none;background:var(--glass);border:1px solid var(--line);border-radius:10px;padding:9px 11px;z-index:5;box-shadow:0 8px 24px rgba(0,0,0,.18);backdrop-filter:blur(8px)';
  wrap.appendChild(tip);
  const C = [80.72, 7.72], COSL = Math.cos(C[1]*Math.PI/180);
  function project(lon, lat, W, H){
    const k = Math.min(H * 0.32, W * 0.4) * Math.pow(2, view.zoom - 7.05);
    const x = (lon - C[0]) * COSL * k, y = -(lat - C[1]) * k;
    const b = -view.bearing * Math.PI/180, cb = Math.cos(b), sb = Math.sin(b);
    const xr = x*cb - y*sb, yr = x*sb + y*cb;
    const narrow = W < 640;
    return [W/2 + (narrow ? 0 : 40) + xr, H*(narrow ? 0.6 : 0.53) + yr * Math.cos(view.pitch*Math.PI/180)];
  }
  let hover = null;
  function draw(){
    const W = wrap.clientWidth, H = wrap.clientHeight, slab = 10, colH = H * 0.27;
    const lt = L(), edge = lt ? 'rgba(10,114,194,' : 'rgba(56,189,248,';
    let s = `<defs><linearGradient id="side" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${lt ? '#c9dcef' : '#0d1a33'}"/><stop offset="1" stop-color="${lt ? '#a9c3de' : '#060b18'}"/></linearGradient>
      <filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
    const paths = D.provinces.map(p => p.polygon.map(q => project(q[0], q[1], W, H)));
    paths.forEach(pts => { s += `<polygon points="${pts.map(q => q[0].toFixed(1)+','+(q[1]+slab).toFixed(1)).join(' ')}" fill="url(#side)" stroke="${edge}.25)" stroke-width="1"/>`; });
    paths.forEach((pts, i) => { s += `<polygon points="${pts.map(q => q[0].toFixed(1)+','+q[1].toFixed(1)).join(' ')}" fill="${pal(D.provinces[i].name)}" stroke="${edge}.55)" stroke-width="0.8"><title>${D.provinces[i].name}</title></polygon>`; });
    const cols = D.cities.map((c, i) => { const p = project(c.pos[0], c.pos[1], W, H); return {c, i, x: p[0], y: p[1]}; }).sort((a, b) => a.y - b.y);
    const pulse = reduce ? 0.5 : (Math.sin(performance.now()/700) + 1) / 2;
    cols.forEach(({c, i, x, y}) => {
      const base = colOf(c), h = (8 + colH * c.h[measure]) * grow, col = css(base), top = css(base.map(v => Math.min(255, v + 45)));
      const r = 4.5, ry = r * Math.cos(view.pitch*Math.PI/180);
      s += `<ellipse cx="${x}" cy="${y}" rx="${10 + 5*pulse}" ry="${(10 + 5*pulse)*ry/r}" fill="none" stroke="${col}" stroke-opacity="${0.35 + 0.4*(1-pulse)}"/>`;
      s += `<g data-i="${i}" style="cursor:pointer" ${hover === i ? 'filter="url(#glow)"' : ''}><rect x="${x-r}" y="${y-h}" width="${2*r}" height="${h}" fill="${col}"/>
        <rect x="${x-r}" y="${y-h}" width="${r*0.8}" height="${h}" fill="#fff" fill-opacity=".14"/>
        <ellipse cx="${x}" cy="${y}" rx="${r}" ry="${ry}" fill="${col}"/><ellipse cx="${x}" cy="${y-h}" rx="${r}" ry="${ry}" fill="${top}"/>
        <rect x="${x-9}" y="${y-h-6}" width="18" height="${h+12}" fill="transparent"/></g>`;
      if (c.label) s += `<text x="${x}" y="${y-h-10}" text-anchor="middle" font-family="Inter,sans-serif" font-size="12" font-weight="500" fill="${lt ? '#0f172a' : '#e6ecf5'}" stroke="${lt ? '#ffffff' : '#060a16'}" stroke-width="3" paint-order="stroke">${c.city}</text>`;
    });
    svg.innerHTML = s;
  }
  svg.addEventListener('pointermove', e => {
    const g = e.target.closest('g[data-i]');
    hover = g ? +g.dataset.i : null;
    if (g) { const r = wrap.getBoundingClientRect(); tip.innerHTML = tipHtml(D.cities[hover]); tip.style.display = 'block';
      tip.style.left = Math.min(e.clientX - r.left + 14, r.width - 190) + 'px'; tip.style.top = (e.clientY - r.top + 14) + 'px'; }
    else tip.style.display = 'none';
  });
  svg.addEventListener('pointerleave', () => { hover = null; tip.style.display = 'none'; });
  let drag = null;
  svg.addEventListener('pointerdown', e => { drag = {x: e.clientX, b: view.bearing}; svg.setPointerCapture(e.pointerId); svg.style.cursor = 'grabbing'; setSpin(false); });
  svg.addEventListener('pointermove', e => { if (drag) { view = Object.assign({}, view, {bearing: drag.b - (e.clientX - drag.x) * 0.4}); draw(); } });
  svg.addEventListener('pointerup', () => { drag = null; svg.style.cursor = 'grab'; });
  redraw = draw;
  let last = 0;
  function frame(now){ tick(now); if (now - last > 33) { draw(); last = now; } if (!reduce) requestAnimationFrame(frame); }
  if (reduce) draw(); else requestAnimationFrame(frame);
}
</script></body></html>
"""
