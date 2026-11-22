# AI-assisted: generated with Claude Code, reviewed and run by the author
"""Streamlit dashboard for the Sri Lanka weather ensemble project.

It only reads files that the notebook saved in artifacts/, so it starts quickly and
never retrains a model. Run with:  streamlit run app/app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from island3d import city_table, island_html

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIG = ART / "figures"

# AI-assisted: one palette for both themes (dataviz skill validator): adjacent CVD ΔE ≥ 8.4,
# normal-vision ΔE ≥ 19.8, and every hue ≥ 3:1 against both the dark (#0e1526) and white surface.
BLUE, ORANGE, AQUA, YELLOW = "#3987e5", "#d95926", "#199e70", "#c98500"
NEUTRAL = "#8b93a7"
FAMILY_COL = {"Baseline": NEUTRAL, "Bagging": BLUE, "Boosting": ORANGE, "Stacking": AQUA, "Hybrid": YELLOW}
SEASON_COL = {"SW": BLUE, "NE": ORANGE, "Inter": NEUTRAL}
TASK_LABEL = {"tmax": "Next-day max temperature (°C)",
              "rain_amount": "Next-day rainfall amount (log1p mm)",
              "rain_flag": "Will it rain tomorrow? (≥ 1 mm)"}
TASK_SHORT = {"tmax": "Max temp", "rain_amount": "Rain amount", "rain_flag": "Rain yes / no"}
MODEL_ORDER = ["Persistence", "Climatology", "Linear", "RandomForest", "BaggedTrees", "ExtraTrees",
               "HistGB", "XGBoost", "LightGBM", "CatBoost", "Stacking", "RF+ANN blend"]

st.set_page_config(page_title="Sri Lanka Weather Ensembles", page_icon="🌦️", layout="wide")


def is_light() -> bool:
    return st.context.theme.type == "light"


def ink() -> str:
    """Main text colour for the few chart marks that must stand out (the 'actual' line)."""
    return "#0f172a" if is_light() else "#e6ecf5"


# ------------------------------------------------------------------ style
# AI-assisted: colours live in CSS variables; a small script (below) copies the active Streamlit
# theme onto <html data-sl-theme>, so switching Light/Dark in the ⋮ menu restyles everything at once.
CSS = """
<style>
:root{
  --ink:#e6ecf5;--ink2:#b4c0d3;--muted:#8b98ad;--line:rgba(148,163,184,.18);
  --card:rgba(16,24,44,.55);--card-solid:#0e1526;--glass-edge:rgba(148,163,184,.22);--glass-hi:rgba(255,255,255,.06);
  --shadow:0 10px 30px rgba(0,0,0,.30);--blur:blur(18px) saturate(140%);
  --accent:#38bdf8;--accent2:#a78bfa;--accent-ink:#06101f;--chip:rgba(255,255,255,.04);
  --hero:linear-gradient(135deg,rgba(14,25,48,.9),rgba(10,14,28,.75));
  --hero-grid:rgba(56,189,248,.07);
  --title-grad:linear-gradient(90deg,#e6ecf5 0%,#7dd3fc 55%,#c4b5fd 100%);
  --page:radial-gradient(rgba(148,163,184,.07) 1px,transparent 1.3px) 0 0/24px 24px,
         radial-gradient(900px 500px at 8% -10%,rgba(56,189,248,.18),transparent 60%),
         radial-gradient(800px 500px at 100% 0%,rgba(167,139,250,.14),transparent 60%),
         radial-gradient(700px 600px at 50% 120%,rgba(25,158,112,.12),transparent 60%),#070b16;
}
html[data-sl-theme="light"]{
  --ink:#0f172a;--ink2:#334155;--muted:#5b6678;--line:#d9e3ee;
  --card:rgba(255,255,255,.55);--card-solid:#ffffff;--glass-edge:rgba(255,255,255,.85);--glass-hi:rgba(255,255,255,.95);
  --shadow:0 1px 2px rgba(15,23,42,.05),0 12px 32px rgba(30,64,120,.10);--blur:blur(20px) saturate(170%);
  --accent:#0a72c2;--accent2:#7c3aed;--accent-ink:#ffffff;--chip:#f1f6fb;
  --hero:linear-gradient(135deg,rgba(255,255,255,.78) 0%,rgba(255,255,255,.5) 60%,rgba(245,240,255,.55) 100%);
  --hero-grid:rgba(10,114,194,.07);
  --title-grad:linear-gradient(90deg,#0f172a 0%,#0a72c2 50%,#7c3aed 100%);
  --page:radial-gradient(rgba(15,23,42,.07) 1px,transparent 1.3px) 0 0/24px 24px,
         radial-gradient(620px 460px at 4% 2%,rgba(56,189,248,.38),transparent 65%),
         radial-gradient(560px 440px at 96% 8%,rgba(167,139,250,.34),transparent 65%),
         radial-gradient(640px 520px at 78% 72%,rgba(52,211,153,.26),transparent 65%),
         radial-gradient(560px 460px at 12% 88%,rgba(251,146,60,.20),transparent 65%),#f4f8fc;
}
.stApp{background:var(--page);background-attachment:fixed}
.block-container{padding-top:4.4rem;padding-bottom:2rem;max-width:1280px}
.stMarkdown p,.stMarkdown li{line-height:1.65}
h1,h2,h3,h4{letter-spacing:-.015em}
:focus-visible{outline:2px solid var(--accent) !important;outline-offset:2px}

/* cards */
/* glass: translucent fill + background blur + bright 1px edge + top highlight (glassmorphism) */
[data-testid="stMetric"],
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--card);
  box-shadow:var(--shadow),inset 0 1px 0 var(--glass-hi);
  backdrop-filter:var(--blur);-webkit-backdrop-filter:var(--blur)}
[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid var(--glass-edge) !important;border-radius:18px !important}
[data-testid="stMetric"]{border:1px solid var(--glass-edge);border-radius:16px;padding:16px 18px;position:relative;overflow:hidden;height:100%}
[data-testid="stMetric"]::after{content:"";position:absolute;left:0;top:0;height:3px;width:100%;
  background:linear-gradient(90deg,var(--accent),var(--accent2),transparent)}
[data-testid="stMetricLabel"] p{color:var(--muted);font-size:.74rem;letter-spacing:.09em;text-transform:uppercase;font-weight:600}
/* labels wrap instead of ending in "…" */
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *{white-space:normal !important;overflow:visible !important;text-overflow:clip !important}
[data-testid="stMetricValue"]{font-family:'Space Grotesk',sans-serif;font-weight:600;color:var(--ink);
  font-size:clamp(1.45rem,1.05rem + 1.4vw,2.05rem);line-height:1.15;white-space:normal;overflow-wrap:anywhere}
[data-testid="stImage"] img{border-radius:12px}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}
[data-testid="stCaptionContainer"]{color:var(--muted)}

/* hero */
.hero{position:relative;padding:clamp(18px,3.2vw,30px);border-radius:22px;border:1px solid var(--glass-edge);
  margin-bottom:18px;overflow:hidden;background:var(--hero);box-shadow:var(--shadow),inset 0 1px 0 var(--glass-hi);
  backdrop-filter:var(--blur);-webkit-backdrop-filter:var(--blur)}
.hero::before{content:"";position:absolute;inset:-1px;pointer-events:none;
  background-image:linear-gradient(var(--hero-grid) 1px,transparent 1px),linear-gradient(90deg,var(--hero-grid) 1px,transparent 1px);
  background-size:36px 36px;mask-image:linear-gradient(90deg,transparent,#000 65%)}
.hero > *{position:relative}
.hero .eyebrow{display:inline-flex;gap:8px;align-items:center;font:600 .72rem/1.2 'Space Grotesk',sans-serif;letter-spacing:.14em;
  text-transform:uppercase;color:var(--accent);padding:6px 11px;border:1px solid color-mix(in srgb,var(--accent) 40%,transparent);
  border-radius:999px;background:color-mix(in srgb,var(--accent) 9%,transparent)}
.hero .eyebrow i{width:7px;height:7px;flex:none;border-radius:50%;background:var(--accent);animation:ping 2.4s infinite}
.hero h1{font-family:'Space Grotesk',sans-serif;font-size:clamp(1.6rem,1.1rem + 2.3vw,2.65rem);line-height:1.12;
  margin:14px 0 10px;padding:0;color:var(--ink);font-weight:700;overflow-wrap:break-word}
.hero h1 span{background:var(--title-grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{color:var(--ink2);max-width:860px;margin:0;font-size:clamp(.95rem,.9rem + .25vw,1.05rem);line-height:1.65}
.hero p b{color:var(--ink)}
.hero .who{margin-top:12px;color:var(--muted);font-size:.84rem}
@keyframes ping{0%{box-shadow:0 0 0 0 color-mix(in srgb,var(--accent) 55%,transparent)}80%,100%{box-shadow:0 0 0 9px transparent}}

.kicker{font:600 .72rem/1.3 'Space Grotesk',sans-serif;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);margin:22px 0 6px}
.chart-title{font:600 1.02rem/1.35 'Space Grotesk',sans-serif;color:var(--ink);margin:2px 0 0}
.chart-sub{font-size:.84rem;color:var(--muted);margin:2px 0 4px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:4px 0}
.chip{font-size:.82rem;color:var(--ink2);padding:5px 10px;border:1px solid var(--glass-edge);border-radius:999px;background:var(--chip)}
.chip b{color:var(--ink);font-weight:600}

/* split timeline */
.tl{display:flex;height:40px;border-radius:10px;overflow:hidden;gap:2px;margin:6px 0}
.tl div{display:flex;align-items:center;justify-content:center;font:600 .8rem 'Space Grotesk',sans-serif;color:#06101f;white-space:nowrap;min-width:0;overflow:hidden}
.tl-legend{display:flex;justify-content:space-between;color:var(--muted);font-size:.78rem}
.foot{margin-top:40px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);font-size:.8rem;
  display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px 18px}

/* only Light and Dark are offered (Light is the default) */
[data-testid="stMainMenuItem-theme-System"]{display:none !important}
html.sl-auto [data-testid="stMainMenuPopover"]{opacity:0 !important;pointer-events:none !important}
/* the theme-sync script sits in an empty element; keep it out of the layout */
.stElementContainer:has(.sl-theme-sync){display:none}

/* phones */
@media (max-width:640px){
  .block-container{padding-left:1rem;padding-right:1rem;padding-top:3.6rem}
  .hero{border-radius:18px;margin-bottom:12px}
  .hero .who{font-size:.78rem}
  /* metric rows: two per line instead of one */
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]){flex-flow:row wrap !important;gap:.75rem !important}
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) > [data-testid="stColumn"]{
    flex:1 1 calc(50% - .4rem) !important;min-width:calc(50% - .4rem) !important;width:calc(50% - .4rem) !important}
  [data-testid="stMetric"]{padding:12px 13px}
  [data-testid="stMetricLabel"] p{font-size:.66rem;letter-spacing:.06em}
  [data-testid="stMetricValue"]{font-size:1.3rem}
  .kicker{margin-top:16px}
  .foot{flex-direction:column}
}
@media (prefers-reduced-motion: reduce){*{animation:none !important;transition:none !important}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

THEME_SYNC = """
<span class="sl-theme-sync"></span>
<script>
(() => {
  const root = document.documentElement;
  function sync(){
    const app = document.querySelector('.stApp'); if (!app) return;
    // read the theme's *text* colour: our CSS repaints the background, but never the text colour
    const m = getComputedStyle(app).color.match(/\\d+(\\.\\d+)?/g); if (!m) return;
    const [r, g, b] = m.map(Number), lum = (0.2126*r + 0.7152*g + 0.0722*b) / 255;
    const t = lum < 0.5 ? 'light' : 'dark';
    if (root.dataset.slTheme !== t) root.dataset.slTheme = t;
  }
  sync();
  if (!window.__slThemeSync) { window.__slThemeSync = setInterval(sync, 300); }

  // Light by default: Streamlit starts on "System", so on load we pick Light in its (hidden) menu
  // once. A visitor who later picks Dark keeps Dark, because the menu then no longer says System.
  if (window.__slDefault) return; window.__slDefault = true;
  const q = s => document.querySelector(s);
  const until = (f, ms = 6000) => new Promise(res => { const t0 = Date.now(); (function w(){ const v = f(); if (v || Date.now() - t0 > ms) res(v); else setTimeout(w, 60); })(); });
  (async () => {
    const btn = await until(() => q('[data-testid="stMainMenuButton"], [data-testid="stMainMenu"] button'));
    if (!btn) return;
    root.classList.add('sl-auto');
    btn.click();
    const sys = await until(() => q('[data-testid="stMainMenuItem-theme-System"]'), 2000);
    if (sys && sys.getAttribute('aria-checked') === 'true') q('[data-testid="stMainMenuItem-theme-Light"]').click();
    await new Promise(r => setTimeout(r, 80));
    if (q('[data-testid="stMainMenuPopover"]')) btn.click();
    document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
    setTimeout(() => root.classList.remove('sl-auto'), 150);
  })();
})();
</script>
"""
st.html(THEME_SYNC, unsafe_allow_javascript=True)


# ------------------------------------------------------------------ data loading
@st.cache_data
def load_metrics():
    return json.loads((ART / "metrics.json").read_text())


@st.cache_data
def load_preds():
    return pd.read_csv(ART / "predictions_test.csv", parse_dates=["time"])


@st.cache_data
def load_daily():
    return pd.read_parquet(ART / "daily_sample.parquet")


@st.cache_data
def load_csv(name, **kw):
    return pd.read_csv(ART / name, **kw)


@st.cache_data
def island(height):
    return island_html(load_daily(), height)


def metrics_table(task: str) -> pd.DataFrame:
    m = load_metrics()
    df = pd.DataFrame(m["tasks"][task]).T
    main = m["main_metric"][task]
    num = [c for c in df.columns if c != "family"]
    df[num] = df[num].astype(float)
    return df.sort_values(main, ascending=(main != "ROC_AUC"))


# ------------------------------------------------------------------ small UI pieces
def hero(eyebrow, title, accent, text, who=False):
    w = ('<div class="who">Gurusinghe W.A.L.M (224063X) · BSc (Hons) IT, University of Moratuwa</div>'
         if who else "")
    st.markdown(f'<div class="hero"><div class="eyebrow"><i></i>{eyebrow}</div>'
                f'<h1>{title} <span>{accent}</span></h1><p>{text}</p>{w}</div>', unsafe_allow_html=True)


def kicker(text):
    st.markdown(f'<div class="kicker">{text}</div>', unsafe_allow_html=True)


def image(name, caption=None):
    p = FIG / f"{name}.png"
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


def plot(fig, height=320, title=None, sub=None, unified=True):
    """Chart in a card. Titles are HTML (so they wrap on phones instead of colliding with the
    legend); colours of text and grid come from the active Streamlit theme."""
    fig.update_layout(
        height=height, title_text="", margin=dict(l=4, r=8, t=8, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12.5),
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, xanchor="left", title_text="",
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified" if unified else "closest", barcornerradius=4,
    )
    with st.container(border=True):
        if title:
            st.markdown(f'<div class="chart-title">{title}</div>' + (f'<div class="chart-sub">{sub}</div>' if sub else ""),
                        unsafe_allow_html=True)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})


def task_picker(options, key):
    t = st.segmented_control("Task", options, format_func=TASK_SHORT.get, default=options[0], key=key)
    return t or options[0]


def table(df, **kw):
    st.dataframe(df, width="stretch", **kw)


def footer():
    st.markdown('<div class="foot"><span>Bagging · Boosting · Stacking · Hybrids</span>'
                '<span>Data: Open-Meteo via Kaggle (rasulmah) · Map: geoBoundaries (CC BY 4.0)</span></div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ pages
def page_overview():
    hero("Next-day forecasts · 30 cities", "Ensemble learning on", "Sri Lankan weather",
         "Sri Lanka's year is shaped by two monsoons: the <b>south-west</b> (May–Sep) and the "
         "<b>north-east</b> (Dec–Feb). They decide when farmers plant, when rivers flood and when "
         "dengue rises. This dashboard compares <b>bagging, boosting, stacking and hybrid</b> models "
         "that forecast tomorrow's rain and temperature for 30 cities.", who=True)
    daily, m = load_daily(), load_metrics()
    c = st.columns(4)
    c[0].metric("Daily records", f"{len(daily):,}")
    c[1].metric("Cities", daily.city.nunique())
    c[2].metric("Period", f"{daily.time.min():%Y} – {daily.time.max():%Y}")
    best_t = metrics_table("tmax").iloc[0]
    c[3].metric("Best Tmax MAE (test)", f"{best_t.MAE:.3f} °C", help=best_t.name)

    left, right = st.columns([3, 2], gap="medium")
    with left:
        kicker("The island in 3D")
        components.html(island(600), height=604)
        st.caption("Column height and colour show the same value. Overlapping Colombo-area cities are "
                   "spread around a small ring. Drag to turn the island; Pause stops the rotation.")
        with st.expander("Map values as a table"):
            cs = city_table(daily)[["city", "rain", "temp", "elev"]].sort_values("rain", ascending=False)
            table(cs.rename(columns={"rain": "rain (mm/yr)", "temp": "mean temp (°C)", "elev": "elevation (m)"})
                  .style.format({"rain (mm/yr)": "{:,.0f}", "mean temp (°C)": "{:.1f}", "elevation (m)": "{:.0f}"}),
                  hide_index=True)
    with right:
        kicker("Dataset")
        with st.container(border=True):
            st.markdown(
                "- Kaggle *Sri Lanka Weather Dataset* (`rasulmah`), Open-Meteo reanalysis\n"
                "- 147,480 rows × 24 columns, 2010-01-01 → 2023-06-17\n"
                "- 30 cities from Jaffna to Hambantota, 0 m to 1,281 m elevation\n")
        kicker("Time-based split")
        sp = m["split"]
        with st.container(border=True):
            st.markdown(
                f'<div class="tl" role="img" aria-label="Train {sp["train"]}, validation {sp["validation"]}, test {sp["test"]}">'
                f'<div style="flex:11;background:{BLUE}">Train</div>'
                f'<div style="flex:1.4;background:{YELLOW}">Val</div>'
                f'<div style="flex:1.8;background:{AQUA}">Test</div></div>'
                '<div class="tl-legend"><span>2010</span><span>2021</span><span>2023</span></div>',
                unsafe_allow_html=True)
            st.markdown(f"- **Train:** {sp['train']}\n- **Validation:** {sp['validation']}\n- **Test:** {sp['test']}")
        st.info("All scores on this dashboard are on the test period, which no model saw during "
                "training, tuning, early stopping or blending.", icon=":material/verified:")
        kicker("Best test scores")
        with st.container(border=True):
            for task in TASK_LABEL:
                tb, main = metrics_table(task), m["main_metric"][task]
                st.markdown(f'<div class="chips"><span class="chip">{TASK_SHORT[task]}</span>'
                            f'<span class="chip"><b>{tb.index[0]}</b> · {main} {tb.iloc[0][main]:.4f}</span></div>',
                            unsafe_allow_html=True)
    footer()


def page_explore():
    hero("Daily weather · 2010–2023", "Explore", "the weather",
         "Pick a city and a date range to see its daily temperature, its rain coloured by monsoon "
         "season, and its average year.")
    daily = load_daily()
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        cities = sorted(daily.city.unique())
        city = c1.selectbox("City", cities, index=cities.index("Colombo"))
        dmin, dmax = daily.time.min().date(), daily.time.max().date()
        start, end = c2.slider("Date range", dmin, dmax, (pd.Timestamp("2021-01-01").date(), dmax))
    d = daily[(daily.city == city) & daily.time.between(pd.Timestamp(start), pd.Timestamp(end))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.time, y=d.temperature_2m_max, name="max", line=dict(color=ORANGE, width=1.6)))
    fig.add_trace(go.Scatter(x=d.time, y=d.temperature_2m_min, name="min", line=dict(color=BLUE, width=1.6),
                             fill="tonexty", fillcolor="rgba(217,89,38,0.10)"))
    fig.update_layout(yaxis_title="°C")
    plot(fig, 320, f"{city}: daily temperature", "Daily max and min, °C")

    fig = px.bar(d, x="time", y="precipitation_sum", color="season", color_discrete_map=SEASON_COL,
                 labels={"precipitation_sum": "mm", "time": ""})
    fig.update_traces(marker_line_width=0)
    plot(fig, 320, f"{city}: daily rainfall", "Coloured by monsoon season (SW May–Sep, NE Dec–Feb)")

    allc = daily[daily.city == city].assign(month=lambda x: x.time.dt.month)
    clim = allc.groupby(["month", allc.time.dt.year]).precipitation_sum.sum().groupby("month").mean().reset_index()
    fig = px.bar(clim, x="month", y="precipitation_sum", labels={"precipitation_sum": "mm / month"},
                 color_discrete_sequence=[BLUE])
    fig.update_xaxes(dtick=1)
    plot(fig, 290, f"{city}: average monthly rainfall", "Mean of each month's total, 2010–2023", unified=False)
    footer()


def page_arena():
    hero("Test-set leaderboard", "Model arena:", "bagging vs boosting vs stacking",
         "Every model is scored on 2022-01-01 → 2023-06-16, after training on 2010–2020 and tuning on 2021.")
    m = load_metrics()
    task = task_picker(list(TASK_LABEL), "arena_task")
    st.caption(TASK_LABEL[task])
    main = m["main_metric"][task]
    tb = metrics_table(task)
    higher = main == "ROC_AUC"

    best = tb[tb.family != "Baseline"].iloc[0]
    fam_best = tb.groupby("family")[main].agg("max" if higher else "min")
    cols = st.columns(len(fam_best))
    for c, (fam, v) in zip(cols, fam_best.reindex(["Baseline", "Bagging", "Boosting", "Stacking"]).items()):
        c.metric(f"Best {fam}", f"{v:.4f}", help=f"test {main}")

    fig = px.bar(tb.reset_index(names="model").iloc[::-1], x=main, y="model", color="family", orientation="h",
                 color_discrete_map=FAMILY_COL, text_auto=".4f")
    lo, hi = tb[main].min(), tb[main].max()
    fig.update_xaxes(range=[lo - (hi - lo) * .2, hi + (hi - lo) * .05] if higher else [0, hi * 1.02])
    # value labels sit inside the bar end in dark ink (≥ 5:1 on every family colour, both themes)
    fig.update_traces(textposition="inside", insidetextanchor="end", textfont=dict(color="#06101f", size=12),
                      cliponaxis=True)
    # best model on top, whatever the colour grouping
    fig.update_layout(yaxis_title="", yaxis_categoryorder="total ascending" if higher else "total descending")
    plot(fig, 470, f"Test {main}", f"{'Higher' if higher else 'Lower'} is better · colour = ensemble family",
         unified=False)

    kicker("Full comparison table (test set)")
    table(tb.style.format({c: "{:.4f}" for c in tb.columns if c not in ("family", "train_time_s")}
                          | {"train_time_s": "{:.1f}"}))
    st.caption(f"Best ensemble: **{best.name}** ({best.family}). Training time is wall-clock seconds on a "
               "16-core laptop; rain amount is scored on the log1p(mm) scale it was trained on.")
    footer()


def page_hybrids():
    hero("Two hybrid designs", "Hybrid", "models",
         "A statistical model and a random forest working together (RF + SARIMA), and a random forest "
         "paired with a neural network (RF + ANN).")
    m = load_metrics()
    kicker("RF + SARIMA · Colombo weekly mean temperature")
    st.markdown("SARIMA(1,0,1)(0,1,1)₅₂ with drift captures the season and the trend; a random forest learns "
                "what SARIMA leaves in its residuals. **Forecast = SARIMA + RF(residual).**")
    wk = load_csv("hybrid_sarima_weekly.csv", index_col=0, parse_dates=True)
    w = wk[wk.split != "train"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w.index, y=w.actual, name="actual", line=dict(color=ink(), width=2.4)))
    for name, c in [("SARIMA", ORANGE), ("RF", BLUE), ("RF+SARIMA", AQUA)]:
        fig.add_trace(go.Scatter(x=w.index, y=w[name], name=name, line=dict(color=c, width=1.8)))
    fig.add_vrect(x0=w.index.min(), x1=pd.Timestamp("2021-12-31"), fillcolor="#38bdf8", opacity=.07, line_width=0,
                  annotation_text="validation", annotation_position="top left")
    fig.update_layout(yaxis_title="°C")
    plot(fig, 360, "Weekly mean temperature, Colombo", "Shaded: validation year 2021 · after it: test")
    table(pd.DataFrame(m["hybrids"]["RF + SARIMA (Colombo weekly temp)"]).T.style.format("{:.4f}"))

    kicker("RF + ANN · next-day max temperature, all cities")
    lam = m.get("rf_ann_residual_lambda")
    st.markdown(f"Blend: ŷ = w·RF + (1−w)·ANN, with **w = {m['rf_ann_blend_weight']}** chosen on 2021. "
                "Residual: the ANN learns the RF's out-of-fold errors, and the forecast is RF + λ·ANN(residual)"
                + (f", with **λ = {lam}** chosen on 2021." if lam is not None else "."))
    table(pd.DataFrame(m["hybrids"]["RF + ANN (next-day Tmax)"]).T.style.format("{:.4f}"))
    p = load_preds()
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        cities = sorted(p.city.unique())
        city = c1.selectbox("City", cities, index=cities.index("Kandy"))
        lo, hi = p.time.min().date(), p.time.max().date()
        start, end = c2.slider("Days to show", lo, hi, ((p.time.max() - pd.Timedelta(days=120)).date(), hi))
    d = p[(p.city == city) & p.time.between(pd.Timestamp(start), pd.Timestamp(end))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.time, y=d["actual__tmax"], name="actual", line=dict(color=ink(), width=2.2)))
    # 3 validated hues + a dotted neutral for the weak ANN (orange and yellow together fail CVD checks)
    for col, name, c, dash in [("tmax__RandomForest", "RF", BLUE, "solid"), ("tmax__ANN", "ANN", NEUTRAL, "dot"),
                               ("tmax__RF+ANN blend", "RF+ANN blend", AQUA, "solid"),
                               ("tmax__RF+ANN residual", "RF+ANN residual", ORANGE, "solid")]:
        fig.add_trace(go.Scatter(x=d.time, y=d[col], name=name, line=dict(color=c, width=1.5, dash=dash)))
    fig.update_layout(yaxis_title="°C")
    plot(fig, 360, f"{city}: next-day max temperature", "Test period · °C")
    footer()


def page_explain():
    hero("Feature importance · SHAP · diagnostics", "Why the models", "behave as they do",
         "What each model leans on, why boosting overfits without early stopping, why stacking adds little, "
         "and why a random split would have flattered every score.")
    task = task_picker(["tmax", "rain_flag"], "explain_task")
    imp = load_csv("feature_importance_rf.csv")
    imp = imp[imp.task == task].sort_values("permutation", ascending=False).head(15)
    long = imp.melt(id_vars="feature", value_vars=["impurity", "permutation"], var_name="type", value_name="share")
    fig = px.bar(long.iloc[::-1], x="share", y="feature", color="type", barmode="group", orientation="h",
                 color_discrete_map={"impurity": NEUTRAL, "permutation": BLUE})
    fig.update_layout(yaxis_title="")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        plot(fig, 520, "RandomForest feature importance", "Impurity vs permutation, top 15", unified=False)
    with c2:
        with st.container(border=True):
            st.markdown('<div class="chart-title">SHAP summary</div><div class="chart-sub">Best boosting model</div>',
                        unsafe_allow_html=True)
            image(f"shap_{task}")

    kicker("Short answers")
    a = st.columns(3)
    notes = [
        ("Bagging lowers variance", "Averaging many deep trees cancels their individual noise; random feature "
         "subsets (RF, ExtraTrees) make trees disagree more, so errors cancel better."),
        ("Boosting lowers bias but can overfit", "Each tree fixes what the others still get wrong; without early "
         "stopping the validation error turns back up (below)."),
        ("Stacking needs diverse errors", "The tree ensembles make highly correlated mistakes, so a meta-learner "
         "has little to add."),
        ("Trees can't extrapolate", "Leaves are averages of past targets, so trend needs SARIMA."),
        ("Rain is harder than temperature", "It is intermittent, heavy-tailed and convective; daily grid averages "
         "miss the local storm triggers."),
    ]
    for i, (h, t) in enumerate(notes):
        with a[i % 3].container(border=True):
            st.markdown(f"**{h}.** {t}")
    c1, c2 = st.columns(2, gap="medium")
    with c1.container(border=True):
        st.markdown('<div class="chart-title">Boosting without early stopping</div>', unsafe_allow_html=True)
        image("boosting_overfit_curve")
    with c2.container(border=True):
        st.markdown('<div class="chart-title">Correlation of base-model test errors</div>', unsafe_allow_html=True)
        image("stacking_error_correlation")

    kicker("Leakage check: random vs time split")
    demo = pd.DataFrame(load_metrics()["leakage_demo"]).set_index(["task", "split"])
    table(demo.style.format("{:.4f}"))
    st.caption("Nearby cities share Open-Meteo grid cells (Colombo = Mabole exactly), so a random split "
               "lets the model peek at a twin of each test day. Final scores always use the time split.")
    st.warning("Limits: modelled reanalysis data, not Department of Meteorology station readings; "
               "nearby cities share grid cells; the test period covers only 18 months.", icon=":material/info:")
    footer()


def page_predict():
    hero("One day, every model", "Predict:", "next-day forecast vs what happened",
         "Choose a city and a day in the test period. Each model forecasts the next day from weather up to "
         "that day, and you can compare it with what really happened.")
    p = load_preds()
    with st.container(border=True):
        c1, c2 = st.columns(2)
        cities = sorted(p.city.unique())
        city = c1.selectbox("City", cities, index=cities.index("Colombo"))
        dates = p.time.dt.date.unique()
        day = c2.date_input("Forecast issued on (test period)", value=pd.Timestamp("2022-10-15").date(),
                            min_value=dates.min(), max_value=dates.max())
    row = p[(p.city == city) & (p.time.dt.date == day)]
    if row.empty:
        st.warning("No data for that day.")
        return
    r = row.iloc[0]
    tomorrow = (pd.Timestamp(day) + pd.Timedelta(days=1)).date()
    st.markdown(f"Using weather up to **{day}** in **{city}** to forecast **{tomorrow}**. "
                f"Today it rained {r.precipitation_sum:.1f} mm, with a max of {r.temperature_2m_max:.1f} °C.")

    a = st.columns(3)
    a[0].metric("Actual rain tomorrow", f"{r.actual_rain_mm:.1f} mm")
    a[1].metric("Actual rain day?", "Yes ☔" if r["actual__rain_flag"] == 1 else "No ☀️")
    a[2].metric("Actual max temp tomorrow", f"{r['actual__tmax']:.1f} °C")

    rows = []
    for model in MODEL_ORDER:
        rows.append({
            "model": model,
            "P(rain ≥ 1 mm)": r.get(f"rain_flag__{model}", np.nan),
            "rain (mm)": np.expm1(r.get(f"rain_amount__{model}", np.nan)),
            "max temp (°C)": r.get(f"tmax__{model}", np.nan),
            "temp error (°C)": r.get(f"tmax__{model}", np.nan) - r["actual__tmax"],
        })
    tbl = pd.DataFrame(rows).set_index("model")

    from_family = load_metrics()["tasks"]["tmax"]
    fam = {k: (from_family.get(k, {}).get("family") or "Hybrid") for k in tbl.index}
    t = tbl["max temp (°C)"].dropna()
    fig = go.Figure()
    for f in ["Baseline", "Bagging", "Boosting", "Stacking", "Hybrid"]:
        tf = t[[fam[k] == f for k in t.index]]
        fig.add_trace(go.Scatter(x=tf.values, y=tf.index, mode="markers", name=f,
                                 marker=dict(size=12, color=FAMILY_COL[f], line=dict(color="rgba(0,0,0,0)", width=0)),
                                 hovertemplate="%{y}: %{x:.2f} °C<extra></extra>"))
    fig.add_vline(x=r["actual__tmax"], line=dict(color=ink(), width=2, dash="dot"),
                  annotation_text=f"actual {r['actual__tmax']:.1f} °C", annotation_position="top")
    fig.update_layout(xaxis_title="°C", yaxis=dict(autorange="reversed", categoryorder="array",
                                                   categoryarray=list(t.index)))
    plot(fig, 420, "Max temperature forecasts by model", "Dotted line: what really happened",
         unified=False)

    table(tbl.style.format({"P(rain ≥ 1 mm)": "{:.0%}", "rain (mm)": "{:.1f}", "max temp (°C)": "{:.2f}",
                            "temp error (°C)": "{:+.2f}"})
          .background_gradient(subset=["P(rain ≥ 1 mm)"], cmap="Blues", vmin=0, vmax=1))
    st.caption("Persistence for the rain flag is a hard yes/no (today's rain), so it shows 0% or 100%.")
    footer()


# ------------------------------------------------------------------ navigation
PAGES = [
    st.Page(page_overview, title="Overview", icon=":material/public:", url_path="overview", default=True),
    st.Page(page_explore, title="Explore", icon=":material/travel_explore:", url_path="explore"),
    st.Page(page_arena, title="Model Arena", icon=":material/leaderboard:", url_path="arena"),
    st.Page(page_hybrids, title="Hybrids", icon=":material/merge_type:", url_path="hybrids"),
    st.Page(page_explain, title="Explain", icon=":material/psychology:", url_path="explain"),
    st.Page(page_predict, title="Predict", icon=":material/online_prediction:", url_path="predict"),
]
st.navigation(PAGES, position="top").run()
