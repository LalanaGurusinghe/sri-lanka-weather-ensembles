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

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIG = ART / "figures"

# AI-assisted: shared colours so every page uses the same family palette
FAMILY_COL = {"Baseline": "#9aa0a6", "Bagging": "#2a9d8f", "Boosting": "#e76f51",
              "Stacking": "#264653", "Hybrid": "#e9c46a"}
SEASON_COL = {"SW": "#2a9d8f", "NE": "#e76f51", "Inter": "#b0b5ba"}
TASK_LABEL = {"tmax": "Next-day max temperature (°C)",
              "rain_amount": "Next-day rainfall amount (log1p mm)",
              "rain_flag": "Will it rain tomorrow? (≥ 1 mm)"}
MODEL_ORDER = ["Persistence", "Climatology", "Linear", "RandomForest", "BaggedTrees", "ExtraTrees",
               "HistGB", "XGBoost", "LightGBM", "CatBoost", "Stacking", "RF+ANN blend"]

st.set_page_config(page_title="Sri Lanka Weather Ensembles", page_icon="🌦️", layout="wide")


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


def metrics_table(task: str) -> pd.DataFrame:
    m = load_metrics()
    df = pd.DataFrame(m["tasks"][task]).T
    main = m["main_metric"][task]
    num = [c for c in df.columns if c != "family"]
    df[num] = df[num].astype(float)
    return df.sort_values(main, ascending=(main != "ROC_AUC"))


def image(name, caption=None):
    p = FIG / f"{name}.png"
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


# ------------------------------------------------------------------ pages
def page_overview():
    st.title("🌦️ Ensemble Learning on Sri Lankan Weather")
    st.caption("Gurusinghe W.A.L.M (224063X) · BSc (Hons) IT, University of Moratuwa")
    st.markdown(
        "Sri Lanka's year is shaped by two monsoons: the **south-west** (May–Sep) and the "
        "**north-east** (Dec–Feb). They decide when farmers plant, when rivers flood and when "
        "dengue rises. This dashboard compares **bagging, boosting, stacking and hybrid** models "
        "that forecast tomorrow's rain and temperature for 30 cities.")
    daily, m = load_daily(), load_metrics()
    c = st.columns(4)
    c[0].metric("Daily records", f"{len(daily):,}")
    c[1].metric("Cities", daily.city.nunique())
    c[2].metric("Period", f"{daily.time.min():%Y} – {daily.time.max():%Y}")
    best_t = metrics_table("tmax").iloc[0]
    c[3].metric("Best Tmax MAE (test)", f"{best_t.MAE:.3f} °C", help=best_t.name)

    cs = daily.groupby("city").agg(lat=("latitude", "first"), lon=("longitude", "first"),
                                   elevation=("elevation", "first"),
                                   annual_rain_mm=("precipitation_sum", lambda s: s.mean() * 365.25),
                                   mean_temp=("temperature_2m_mean", "mean")).reset_index()
    left, right = st.columns([3, 2])
    with left:
        # AI-assisted: SVG geo map (coastline only), so it needs no WebGL and no map tiles
        cs["marker"] = 8 + cs.elevation / 60
        fig = px.scatter_geo(cs, lat="lat", lon="lon", color="annual_rain_mm", size="marker",
                             size_max=20, hover_name="city", height=560,
                             hover_data={"mean_temp": ":.1f", "annual_rain_mm": ":.0f", "elevation": True,
                                         "marker": False, "lat": False, "lon": False},
                             color_continuous_scale="YlGnBu")
        fig.update_traces(marker_line_color="#264653", marker_line_width=1)
        fig.update_geos(lataxis_range=[5.8, 10.0], lonaxis_range=[79.4, 82.1], showland=True,
                        landcolor="#f2efe6", showocean=True, oceancolor="#dceef5", showcountries=False,
                        coastlinecolor="#8a8a8a", resolution=50, projection_type="mercator")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar_title="mm / yr")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Dataset")
        st.markdown(
            "- Kaggle *Sri Lanka Weather Dataset* (`rasulmah`), Open-Meteo reanalysis\n"
            "- 147,480 rows × 24 columns, 2010-01-01 → 2023-06-17\n"
            "- 30 cities from Jaffna to Hambantota, 0 m to 1,281 m elevation\n")
        st.subheader("Time-based split")
        sp = m["split"]
        st.markdown(f"- **Train:** {sp['train']}\n- **Validation:** {sp['validation']}\n- **Test:** {sp['test']}")
        st.info("All scores on this dashboard are on the test period, which no model saw during "
                "training, tuning, early stopping or blending.")


def page_explore():
    st.title("🔎 Explore the weather")
    daily = load_daily()
    c1, c2 = st.columns([1, 2])
    city = c1.selectbox("City", sorted(daily.city.unique()), index=sorted(daily.city.unique()).index("Colombo"))
    dmin, dmax = daily.time.min().date(), daily.time.max().date()
    start, end = c2.slider("Date range", dmin, dmax, (pd.Timestamp("2021-01-01").date(), dmax))
    d = daily[(daily.city == city) & daily.time.between(pd.Timestamp(start), pd.Timestamp(end))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.time, y=d.temperature_2m_max, name="max", line=dict(color="#e76f51", width=1)))
    fig.add_trace(go.Scatter(x=d.time, y=d.temperature_2m_min, name="min", line=dict(color="#2a9d8f", width=1),
                             fill="tonexty", fillcolor="rgba(231,111,81,0.08)"))
    fig.update_layout(title=f"{city}: daily temperature (°C)", height=330, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    fig = px.bar(d, x="time", y="precipitation_sum", color="season", color_discrete_map=SEASON_COL,
                 labels={"precipitation_sum": "mm", "time": ""}, height=330,
                 title=f"{city}: daily rainfall, coloured by monsoon season")
    fig.update_traces(marker_line_width=0)
    fig.update_layout(margin=dict(t=40, b=10), bargap=0)
    st.plotly_chart(fig, width="stretch")

    allc = daily[daily.city == city].assign(month=lambda x: x.time.dt.month)
    clim = allc.groupby(["month", allc.time.dt.year]).precipitation_sum.sum().groupby("month").mean().reset_index()
    fig = px.bar(clim, x="month", y="precipitation_sum", labels={"precipitation_sum": "mm / month"},
                 title=f"{city}: average monthly rainfall, 2010–2023", height=300,
                 color_discrete_sequence=["#264653"])
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, width="stretch")


def page_arena():
    st.title("🏟️ Model Arena: Bagging vs Boosting vs Stacking")
    m = load_metrics()
    task = st.radio("Task", list(TASK_LABEL), format_func=TASK_LABEL.get, horizontal=True)
    main = m["main_metric"][task]
    tb = metrics_table(task)
    higher = main == "ROC_AUC"

    best = tb[tb.family != "Baseline"].iloc[0]
    fam_best = tb.groupby("family")[main].agg("max" if higher else "min")
    cols = st.columns(len(fam_best))
    for c, (fam, v) in zip(cols, fam_best.reindex(["Baseline", "Bagging", "Boosting", "Stacking"]).items()):
        c.metric(f"Best {fam}", f"{v:.4f}", help=f"test {main}")

    fig = px.bar(tb.reset_index(names="model").iloc[::-1], x=main, y="model", color="family", orientation="h",
                 color_discrete_map=FAMILY_COL, height=460, title=f"Test {main} ({'higher' if higher else 'lower'} is better)",
                 text_auto=".4f")
    lo, hi = tb[main].min(), tb[main].max()
    fig.update_xaxes(range=[lo - (hi - lo) * .2, hi + (hi - lo) * .1] if higher else [0, hi * 1.08])
    # best model on top, whatever the colour grouping
    fig.update_layout(margin=dict(t=40, b=10), yaxis_title="",
                      yaxis_categoryorder="total ascending" if higher else "total descending")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Full comparison table (test set)")
    st.dataframe(tb.style.format({c: "{:.4f}" for c in tb.columns if c not in ("family", "train_time_s")}
                                 | {"train_time_s": "{:.1f}"}), width="stretch")
    st.caption(f"Best ensemble: **{best.name}** ({best.family}). Training time is wall-clock seconds on a "
               "16-core laptop; rain amount is scored on the log1p(mm) scale it was trained on.")


def page_hybrids():
    st.title("🧬 Hybrid models")
    m = load_metrics()
    st.subheader("RF + SARIMA: Colombo weekly mean temperature")
    st.markdown("SARIMA(1,0,1)(0,1,1)₅₂ with drift captures the season and the trend; a random forest learns "
                "what SARIMA leaves in its residuals. **Forecast = SARIMA + RF(residual).**")
    wk = load_csv("hybrid_sarima_weekly.csv", index_col=0, parse_dates=True)
    w = wk[wk.split != "train"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w.index, y=w.actual, name="actual", line=dict(color="black", width=2.5)))
    for name, c in [("SARIMA", "#264653"), ("RF", "#2a9d8f"), ("RF+SARIMA", "#e76f51")]:
        fig.add_trace(go.Scatter(x=w.index, y=w[name], name=name, line=dict(color=c, width=1.4)))
    fig.add_vrect(x0=w.index.min(), x1=pd.Timestamp("2021-12-31"), fillcolor="grey", opacity=.08, line_width=0,
                  annotation_text="validation", annotation_position="top left")
    fig.update_layout(height=380, yaxis_title="°C", margin=dict(t=20, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(m["hybrids"]["RF + SARIMA (Colombo weekly temp)"]).T.style.format("{:.4f}"),
                 width="stretch")

    st.divider()
    st.subheader("RF + ANN: next-day max temperature, all cities")
    lam = m.get("rf_ann_residual_lambda")
    st.markdown(f"Blend: ŷ = w·RF + (1−w)·ANN, with **w = {m['rf_ann_blend_weight']}** chosen on 2021. "
                "Residual: the ANN learns the RF's out-of-fold errors, and the forecast is RF + λ·ANN(residual)"
                + (f", with **λ = {lam}** chosen on 2021." if lam is not None else "."))
    st.dataframe(pd.DataFrame(m["hybrids"]["RF + ANN (next-day Tmax)"]).T.style.format("{:.4f}"),
                 width="stretch")
    p = load_preds()
    c1, c2 = st.columns([1, 2])
    city = c1.selectbox("City", sorted(p.city.unique()), index=sorted(p.city.unique()).index("Kandy"))
    lo, hi = p.time.min().date(), p.time.max().date()
    start, end = c2.slider("Days to show", lo, hi, ((p.time.max() - pd.Timedelta(days=120)).date(), hi))
    d = p[(p.city == city) & p.time.between(pd.Timestamp(start), pd.Timestamp(end))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.time, y=d["actual__tmax"], name="actual", line=dict(color="black", width=2)))
    for col, name, c in [("tmax__RandomForest", "RF", "#2a9d8f"), ("tmax__ANN", "ANN", "#9aa0a6"),
                         ("tmax__RF+ANN blend", "RF+ANN blend", "#e76f51"),
                         ("tmax__RF+ANN residual", "RF+ANN residual", "#e9c46a")]:
        fig.add_trace(go.Scatter(x=d.time, y=d[col], name=name, line=dict(color=c, width=1.1)))
    fig.update_layout(height=380, yaxis_title="°C", title=f"{city}: test period", margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")


def page_explain():
    st.title("🧠 Why the models behave as they do")
    task = st.radio("Task", ["tmax", "rain_flag"], format_func=TASK_LABEL.get, horizontal=True)
    imp = load_csv("feature_importance_rf.csv")
    imp = imp[imp.task == task].sort_values("permutation", ascending=False).head(15)
    long = imp.melt(id_vars="feature", value_vars=["impurity", "permutation"], var_name="type", value_name="share")
    fig = px.bar(long.iloc[::-1], x="share", y="feature", color="type", barmode="group", orientation="h",
                 color_discrete_map={"impurity": "#9aa0a6", "permutation": "#2a9d8f"}, height=520,
                 title="RandomForest: impurity vs permutation importance (top 15)")
    fig.update_layout(margin=dict(t=40, b=10), yaxis_title="")
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig, width="stretch")
    with c2:
        image(f"shap_{task}", "SHAP summary for the best boosting model")

    st.subheader("Short answers")
    st.markdown(
        "- **Bagging lowers variance.** Averaging many deep trees cancels their individual noise; "
        "random feature subsets (RF, ExtraTrees) make trees disagree more, so errors cancel better.\n"
        "- **Boosting lowers bias but can overfit.** Each tree fixes what the others still get wrong; "
        "without early stopping the validation error turns back up (below).\n"
        "- **Stacking needs diverse errors.** The tree ensembles make highly correlated mistakes, so a "
        "meta-learner has little to add.\n"
        "- **Trees can't extrapolate.** Leaves are averages of past targets, so trend needs SARIMA.\n"
        "- **Rain is harder than temperature.** It is intermittent, heavy-tailed and convective; "
        "daily grid averages miss the local storm triggers.")
    c1, c2 = st.columns(2)
    with c1:
        image("boosting_overfit_curve", "Boosting without early stopping")
    with c2:
        image("stacking_error_correlation", "Correlation of base-model test errors")

    st.subheader("Leakage check: random vs time split")
    demo = pd.DataFrame(load_metrics()["leakage_demo"]).set_index(["task", "split"])
    st.dataframe(demo.style.format("{:.4f}"), width="stretch")
    st.caption("Nearby cities share Open-Meteo grid cells (Colombo = Mabole exactly), so a random split "
               "lets the model peek at a twin of each test day. Final scores always use the time split.")
    st.warning("Limits: modelled reanalysis data, not Department of Meteorology station readings; "
               "nearby cities share grid cells; the test period covers only 18 months.")


def page_predict():
    st.title("🔮 Predict: next-day forecast vs what happened")
    p = load_preds()
    c1, c2 = st.columns(2)
    city = c1.selectbox("City", sorted(p.city.unique()), index=sorted(p.city.unique()).index("Colombo"))
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
    st.dataframe(tbl.style.format({"P(rain ≥ 1 mm)": "{:.0%}", "rain (mm)": "{:.1f}", "max temp (°C)": "{:.2f}",
                                   "temp error (°C)": "{:+.2f}"})
                 .background_gradient(subset=["P(rain ≥ 1 mm)"], cmap="Blues", vmin=0, vmax=1),
                 width="stretch")
    st.caption("Persistence for the rain flag is a hard yes/no (today's rain), so it shows 0% or 100%.")


# ------------------------------------------------------------------ navigation
PAGES = {"Overview": page_overview, "Explore": page_explore, "Model Arena": page_arena,
         "Hybrids": page_hybrids, "Explain": page_explain, "Predict": page_predict}
with st.sidebar:
    st.header("Sri Lanka Weather")
    choice = st.radio("Page", list(PAGES))
    st.divider()
    st.caption("Bagging · Boosting · Stacking · Hybrids\n\nData: Open-Meteo via Kaggle (rasulmah)")
PAGES[choice]()
