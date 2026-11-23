# AI-assisted: generated with Claude Code, reviewed and run by the author
"""What the dashboard's 'Ask' page knows, plus the scenario verdicts.

Everything here is built from artifacts/metrics.json, so the numbers the chatbot quotes always
match the rest of the dashboard. Two pieces:

* verdicts(metrics): which model wins in which scenario (used by the Verdict page and the bot)
* build_kb(metrics): short question/answer cards, searched with a small BM25 ranker, so the bot
  works with no API key; the same cards are the grounding text when Claude answers.
"""
from __future__ import annotations

import math
import re
from collections import Counter

TASKS = ["tmax", "rain_amount", "rain_flag"]
TASK_NAME = {"tmax": "next-day max temperature", "rain_amount": "next-day rain amount",
             "rain_flag": "will it rain tomorrow (≥ 1 mm)"}
UNIT = {"tmax": "MAE °C", "rain_amount": "MAE (log1p mm)", "rain_flag": "ROC-AUC"}
BASELINES = {"Persistence", "Climatology"}
NICE = {"RandomForest": "Random Forest", "ExtraTrees": "Extra Trees", "BaggedTrees": "Bagged Trees",
        "HistGB": "HistGradientBoosting", "Linear": "Linear (Ridge / Logistic)",
        "RF": "Random Forest", "Blend (w_RF=0.8)": "RF + ANN blend (80 / 20)"}


def nice(m: str) -> str:
    return NICE.get(m, m)


def _main(metrics, task):
    return metrics["main_metric"][task]


def _better(task, metrics):
    """True when a higher score is better for this task's main metric."""
    return _main(metrics, task) == "ROC_AUC"


def ranked(metrics, task, metric=None, exclude=()):
    """[(model, score)] best first."""
    metric = metric or _main(metrics, task)
    rows = [(m, v[metric]) for m, v in metrics["tasks"][task].items() if m not in exclude and metric in v]
    hi = metric in ("ROC_AUC", "PR_AUC", "Accuracy", "F1", "R2")
    return sorted(rows, key=lambda r: r[1], reverse=hi)


def fmt(task, v):
    return f"{v:.4f}" if task == "rain_flag" else (f"{v:.3f} °C" if task == "tmax" else f"{v:.4f}")


# ------------------------------------------------------------------ verdicts
def verdicts(metrics) -> dict:
    """Winner per task and per practical scenario, all computed from metrics.json."""
    t = metrics["tasks"]
    out = {"per_task": [], "scenarios": []}

    for task in TASKS:
        allr = ranked(metrics, task)
        singles = [r for r in allr if r[0] not in BASELINES | {"Stacking", "Linear"}]
        persist = t[task]["Persistence"][_main(metrics, task)]
        best_m, best_v = allr[0]
        if _better(task, metrics):
            gain = f"+{(best_v - persist):.3f} AUC over persistence"
        else:
            gain = f"{(persist - best_v) / persist:.0%} lower error than persistence"
        out["per_task"].append({
            "task": task, "winner": best_m, "score": best_v, "runner_up": allr[1][0],
            "runner_score": allr[1][1], "best_single": singles[0][0], "single_score": singles[0][1],
            "gain": gain, "family": t[task][best_m]["family"],
        })

    # average rank of each single model across the three tasks (1 = best)
    pool = [m for m in t["tmax"] if m not in BASELINES | {"Stacking"}]
    ranks = {m: [] for m in pool}
    for task in TASKS:
        order = [m for m, _ in ranked(metrics, task) if m in pool]
        for i, m in enumerate(order, 1):
            ranks[m].append(i)
    avg = sorted(((m, sum(r) / len(r)) for m, r in ranks.items()), key=lambda r: r[1])
    out["avg_rank"] = avg
    out["rank_table"] = {task: {m: i for i, (m, _) in enumerate(
        [r for r in ranked(metrics, task) if r[0] not in BASELINES], 1)} for task in TASKS}

    # fastest model within 1% of the best single score, per task
    fast = []
    for task in TASKS:
        best = ranked(metrics, task, exclude=BASELINES | {"Stacking"})[0][1]
        hi = _better(task, metrics)
        ok = [(m, t[task][m]["train_time_s"]) for m, v in ranked(metrics, task, exclude=BASELINES | {"Stacking"})
              if (v >= best * 0.99 if hi else v <= best * 1.01)]
        fast.append((task, min(ok, key=lambda r: r[1])))

    acc = ranked(metrics, "rain_flag", "Accuracy")[0]
    f1 = ranked(metrics, "rain_flag", "F1")[0]
    lin_gap = [(task, t[task]["Linear"][_main(metrics, task)], ranked(metrics, task)[0][1]) for task in TASKS]
    hy = metrics["hybrids"]
    sar = hy["RF + SARIMA (Colombo weekly temp)"]
    ann = hy["RF + ANN (next-day Tmax)"]
    ann_best = min(ann.items(), key=lambda kv: kv[1]["MAE"])
    sar_best = min(sar.items(), key=lambda kv: kv[1]["MAE"])
    bag = {task: ranked(metrics, task, exclude=set(t[task]) - {"RandomForest", "BaggedTrees", "ExtraTrees"})[0]
           for task in TASKS}
    boo = {task: ranked(metrics, task, exclude=set(t[task]) - {"HistGB", "XGBoost", "LightGBM", "CatBoost"})[0]
           for task in TASKS}

    S = out["scenarios"]
    S.append(dict(icon=":material/emoji_events:", title="Highest accuracy, whatever it costs",
                  winner="Stacking",
                  why="Won all three tasks. It blends Random Forest, LightGBM, CatBoost and a linear model "
                      "through a Ridge / Logistic meta-learner, so different mistakes partly cancel.",
                  cost=f"Slowest to train (about {t['rain_flag']['Stacking']['train_time_s']:.0f} s for rain yes/no) "
                       "and four models to keep."))
    S.append(dict(icon=":material/thermostat:", title="One model for tomorrow's max temperature",
                  winner=out["per_task"][0]["best_single"],
                  why=f"Best single model on Tmax ({fmt('tmax', out['per_task'][0]['single_score'])} MAE). "
                      "Boosting fixes bias step by step, which suits a smooth, seasonal signal.",
                  cost="The four boosting libraries finish within about 0.002 °C of each other."))
    S.append(dict(icon=":material/water_drop:", title="One model for how much it will rain",
                  winner=out["per_task"][1]["best_single"],
                  why=f"Best single model on rain amount ({out['per_task'][1]['single_score']:.4f} MAE). "
                      "It uses the city name as a real category, and rain depends on how each place "
                      "faces the monsoons.",
                  cost="Rain amount stays hard for everyone: R² is only about 0.48."))
    S.append(dict(icon=":material/umbrella:", title="Rank days by chance of rain (ROC-AUC)",
                  winner=out["per_task"][2]["best_single"],
                  why=f"Best ensemble single model for rain yes/no (AUC {out['per_task'][2]['single_score']:.4f}). "
                      f"Logistic regression is just as good ({t['rain_flag']['Linear']['ROC_AUC']:.4f}), "
                      "because the signal is mostly additive.",
                  cost="The gain from any ensemble here is tiny; good lag features matter more."))
    S.append(dict(icon=":material/check_circle:", title="Plain yes/no labels (accuracy and F1)",
                  winner=acc[0],
                  why=(f"Highest accuracy ({acc[1]:.1%}) and highest F1 ({f1[1]:.3f}) at the 0.5 threshold. "
                       if acc[0] == f1[0] else
                       f"Highest accuracy ({acc[1]:.1%}); best F1 is {nice(f1[0])} ({f1[1]:.3f}). ")
                      + "Simple and easy to explain.",
                  cost="Stacking still ranks days better (higher ROC-AUC)."))
    S.append(dict(icon=":material/all_inclusive:", title="One single model for all three tasks",
                  winner=avg[0][0],
                  why=f"Best average rank across the three tasks ({avg[0][1]:.1f}; 1 = best), "
                      f"ahead of {nice(avg[1][0])} ({avg[1][1]:.1f}) and {nice(avg[2][0])} ({avg[2][1]:.1f}).",
                  cost="Never the single best, but never far behind and quick to train."))
    fast_txt = "; ".join(f"{TASK_NAME[task].split(' (')[0]}: {nice(m)} ({s:.1f} s)" for task, (m, s) in fast)
    S.append(dict(icon=":material/bolt:", title="Fast training, with accuracy within 1% of the best",
                  winner=nice(fast[0][1][0]),
                  why=f"Fastest model within 1% of the best single model for each task: {fast_txt}.",
                  cost="Timings are from one laptop run, so treat them as rough."))
    S.append(dict(icon=":material/park:", title="Best bagging model",
                  winner=bag["tmax"][0],
                  why=f"Extra Trees led bagging on both regressions; Random Forest on rain yes/no "
                      f"({nice(bag['rain_flag'][0])}). Both beat plain Bagged Trees everywhere, because random "
                      "features make the trees less alike.",
                  cost="Bagging lowers variance but can't fix bias, so boosting beat it on both regressions."))
    S.append(dict(icon=":material/trending_up:", title="Weekly temperature trend for one city",
                  winner=sar_best[0],
                  why=f"On the 2022–23 test weeks plain RF scored {sar['RF']['MAE']:.3f} MAE, the "
                      f"RF + SARIMA hybrid {sar['RF+SARIMA']['MAE']:.3f} and SARIMA alone "
                      f"{sar['SARIMA']['MAE']:.3f}.",
                  cost="Trees can't go beyond the training range. If the climate warms past it, "
                       "RF + SARIMA is the safer choice."))
    S.append(dict(icon=":material/hub:", title="Adding a neural network",
                  winner=ann_best[0],
                  why=f"80% RF + 20% ANN scored {ann_best[1]['MAE']:.3f} MAE, better than RF "
                      f"({ann['RF']['MAE']:.3f}) and ANN ({ann['ANN']['MAE']:.3f}) alone, because their "
                      "mistakes are only partly alike.",
                  cost=f"The residual hybrid only kept λ = {metrics['rf_ann_residual_lambda']} of the "
                       "ANN's correction: what a strong RF leaves is mostly noise."))
    out["lin_gap"] = lin_gap
    return out


# ------------------------------------------------------------------ knowledge cards
def _rows(metrics, task, k=4):
    return ", ".join(f"{nice(m)} {fmt(task, v)}" for m, v in ranked(metrics, task)[:k])


def build_kb(metrics) -> list[dict]:
    v = verdicts(metrics)
    t = metrics["tasks"]
    pt = {p["task"]: p for p in v["per_task"]}
    demo = {(d["task"], d["split"]): d for d in metrics["leakage_demo"]}
    bi = metrics["best_iterations"]
    hy = metrics["hybrids"]
    K = []

    def card(title, keys, answer, page=None):
        K.append({"title": title, "keys": keys, "answer": answer.strip(), "page": page})

    # ---- project and data
    card("What is this project about?", "project overview goal aim purpose summary about dashboard",
         """This project forecasts **tomorrow's weather for 30 Sri Lankan cities** from today's data and
compares four ensemble ideas: **bagging, boosting, stacking and hybrids**. There are three tasks:
next-day **max temperature**, next-day **rain amount**, and **will it rain tomorrow (≥ 1 mm)**.
The monsoons drive farming seasons, floods and dengue, so these forecasts matter. Author:
Gurusinghe W.A.L.M (224066J), BSc (Hons) IT, University of Moratuwa.""", "Overview")
    card("What is the dataset?", "dataset data kaggle open-meteo rows columns size source cities period years",
         """The Kaggle **Sri Lanka Weather Dataset** by `rasulmah` has daily **Open-Meteo** reanalysis values (modelled weather on
a grid, not station readings). It has **147,480 rows × 24 columns**: **30 cities × 4,916 days**, from **2010-01-01 to
2023-06-17**. Columns cover max, min and mean temperature, "feels like" temperature, rain amount and hours,
sunshine, wind speed, gusts and direction, evaporation (ET0), sunrise and sunset, the weather code,
latitude, longitude and elevation. There are no missing values and no duplicate rows.""", "Explore")
    card("Which cities are in the data?", "cities city list colombo kandy jaffna hatton locations map",
         """30 cities: a large **western cluster** (Colombo, Mabole, Mount Lavinia, Sri Jayewardenepura Kotte,
Moratuwa, Kesbewa, Maharagama, Kolonnawa, Athurugiriya, Oruwala, Gampaha, Negombo, Kalutara, Bentota),
the **south** (Galle, Weligama, Matara, Hambantota), the **north and north-west** (Jaffna, Mannar, Puttalam),
the **east** (Trincomalee, Kalmunai) and **inland and the hills** (Kurunegala, Pothuhera, Matale, Kandy,
Ratnapura, Badulla, Hatton). Elevation runs from 0 m to 1,281 m (Hatton).""", "Overview")
    card("What does the weather data show?", "eda monsoon rain season pattern temperature elevation wettest driest hottest",
         """* The **south-west monsoon** (May–Sep) soaks the west and the hills. The **north-east** (Dec–Feb) brings rain to the
  north and east. **Oct–Nov** (inter-monsoon) is the wettest time island-wide.
* **Hatton** (1,281 m) is about 7 °C cooler than the coast; **Trincomalee** is the hottest (≈ 30.8 °C average max).
* Rain is **heavy-tailed**: the median day is 2.6 mm, but the wettest day hit 338.8 mm. 67% of days have ≥ 1 mm.
* The driest city is **Hambantota** (≈ 1,070 mm a year); the wettest is **Ratnapura** (≈ 2,840 mm a year).""", "Explore")
    card("Why do nearby cities look identical?", "grid cell identical twin mabole mount lavinia shared duplicate",
         """Open-Meteo interpolates from a grid, so cities a few km apart can land on the **same grid cell**, and
temperature and rain use different grids. **Colombo has Mabole's exact temperatures** and **Mount Lavinia's exact
rainfall**. This is why a random train/test split cheats: a test day's twin can sit in training.""", "Explore")

    # ---- method
    card("What are the three prediction tasks?", "tasks targets predict target tmax rain_amount rain_flag regression classification",
         """For each city, using data up to **today (day t)**, predict **tomorrow (day t+1)**:
1. **Max temp** (`tmax`): regression, scored by **MAE in °C** (lower is better).
2. **Rain amount** (`rain_amount`): regression on **log(1 + mm)**, scored by MAE (lower is better).
3. **Rain yes/no** (`rain_flag`): classification of tomorrow ≥ 1 mm, scored by **ROC-AUC** (higher is better).""")
    card("Why predict log(1 + rain)?", "log1p log transform rain skew tail why",
         """Rain is heavy-tailed: most days are light, and a few storms drop 100+ mm. On raw millimetres, squared error
would chase those few storms. **log(1 + mm)** squeezes the tail (and handles 0 mm), so the model learns typical
days better. A separate rain yes/no classifier covers the "does it rain at all" question.""")
    card("What features were built?", "features feature engineering lags lag rolling mean calendar 68",
         """**68 features**, all built **per city** and only from day t or earlier:
* 16 of **today's values** (temperature, rain, rain hours, sunshine, wind, ET0, day length…)
* **lags** of 1, 2, 3 and 7 days for 8 variables
* **7- and 30-day rolling means** for 5 variables
* **calendar**: year, month, day-of-year as sin/cos, and SW / NE monsoon flags
* **location**: latitude, longitude, elevation and a city code (CatBoost gets the city name)

Wind direction and day-of-year are turned into sine and cosine so 359° sits next to 1°.""")
    card("How was data leakage prevented?", "leakage leak prevent rules future cheat honest",
         """* Every feature is from **day t or earlier**; every target is **day t+1**.
* Same-day descriptions of rain (rain hours, weather code) are only used as **today's or lagged** values, never tomorrow's.
* Rolling windows **end on day t**, and everything is built inside each city's own series.
* The split is **by date, never random**.""")
    card("What is the train / validation / test split?", "split time split train validation test dates rows",
         f"""A strict **time split**: **train** {metrics['split']['train']} (119,670 rows), **validation** {metrics['split']['validation']}
(10,950 rows, used for early stopping and blend weights), and **test** {metrics['split']['test']} (15,960 rows,
never touched until the final scores).""")
    card("Why not a random split? (leakage demo)", "random split leakage demo shuffle inflate why not random",
         f"""The same random forest scored **MAE {demo[('tmax','random split')]['MAE']:.3f}** on Tmax with a random split, but
**{demo[('tmax','time split')]['MAE']:.3f}** with the time split. On rain amount it was {demo[('rain_amount','random split')]['MAE']:.3f} vs
{demo[('rain_amount','time split')]['MAE']:.3f}. The random split *looks* better only because a test day's twin (the same grid
cell on the same date) sits in training. The time split is the honest one.""", "Model Arena")
    card("What are the baselines?", "baseline persistence climatology linear ridge logistic",
         f"""* **Persistence**: tomorrow = today.
* **Climatology**: the training-period average for that city and month.
* **Linear**: Ridge regression or Logistic Regression, with the city one-hot encoded.

Every ensemble has to beat them. Persistence Tmax MAE is {t['tmax']['Persistence']['MAE']:.3f} °C, and the best model gets {pt['tmax']['score']:.3f} °C.""")
    card("What do MAE, RMSE, R² and ROC-AUC mean?", "metric metrics mae rmse r2 roc auc f1 accuracy pr score",
         """* **MAE**: the average size of the mistake (e.g. off by 0.59 °C).
* **RMSE**: like MAE, but big mistakes count extra.
* **R²**: the share of the ups and downs explained (1 = perfect, 0 = no better than the average).
* **ROC-AUC**: pick a random rainy day and a random dry day; the chance the model scores the rainy day higher (0.5 = coin toss).
* **Accuracy / F1 / PR-AUC**: classification scores at a threshold; F1 balances catching rain days against false alarms.""")

    # ---- theory
    card("What is ensemble learning?", "ensemble learning what is idea wisdom crowd combine models",
         """**Ensemble learning combines many models so that together they predict better than any one alone.** Like
averaging ten friends' guesses: their mistakes partly cancel. It works when the members are reasonably good
**and wrong in different ways**. The three main families are **bagging** (parallel, then average: lowers
variance), **boosting** (one after another, each fixing the last one's errors: lowers bias) and **stacking**
(a meta-model learns how to combine different models).""")
    card("What are bias and variance?", "bias variance tradeoff overfit underfit",
         """* **Bias**: a model is consistently wrong because it's too simple (e.g. "29 °C every day").
* **Variance**: a model is jumpy and changes a lot with slightly different training data (e.g. a very deep tree).

**Bagging mainly reduces variance; boosting mainly reduces bias.**""")
    card("What is a decision tree?", "decision tree tree split leaf node base learner",
         """A flowchart of yes/no questions learned from the data (e.g. "Did it rain > 3 mm today?" → "Is it SW monsoon?").
Each leaf predicts the average of the training rows that reach it. Trees capture interactions and need no
scaling, but a deep tree **memorises noise** (high variance). Trees also **can't extrapolate**: they never
predict above the highest training value.""")
    card("What is bagging?", "bagging bootstrap aggregating bootstrap sample average vote parallel",
         """**Bagging = Bootstrap AGGregatING.** Make many bootstrap samples (random rows *with replacement*, the same size
as the data; each leaves out about 37% of the rows, called out-of-bag). Train one deep tree per sample, then
**average** the predictions (regression) or **vote** (classification). Each tree is noisy, but their noise
cancels, so **variance drops**. Adding trees never makes bagging overfit; it just stops helping.
Here: Bagged Trees, Random Forest and Extra Trees.""", "Model Arena")
    card("What is a Random Forest?", "random forest rf feature subset max_features decorrelate",
         f"""A random forest is **bagging plus a random subset of features at every split**. Without it, every tree grabs the
strongest feature (today's Tmax) first and the trees look alike. With it, trees are forced to use other
features, so they are **less correlated**, and different mistakes cancel better. Here, 200 trees, tuned with a
6-candidate random search using `TimeSeriesSplit(3)`. Tmax RF MAE: {t['tmax']['RandomForest']['MAE']:.3f} °C.""")
    card("What are Extra Trees?", "extra trees extremely randomized randomised threshold",
         f"""**Extremely Randomised Trees** go further than a random forest: they also pick the **split threshold at random**.
That makes the trees even less alike and faster to train. Here Extra Trees led bagging on both regressions
(Tmax {t['tmax']['ExtraTrees']['MAE']:.3f} °C).""")
    card("What is boosting?", "boosting sequential weak learner residual errors bias",
         """**Boosting builds models one after another; each new (usually shallow) tree focuses on the mistakes the previous
trees still make.** The final prediction is a weighted sum of all trees. It mainly **reduces bias**. Unlike
bagging, it **can overfit** if you keep adding trees, so we use early stopping. Here: HistGB, XGBoost,
LightGBM and CatBoost.""", "Model Arena")
    card("What is AdaBoost?", "adaboost adaptive boosting weights stump",
         """**AdaBoost** (1997) gives every training row a weight. Each round trains a **stump** (a one-split tree); rows it gets
**wrong get heavier** and right ones lighter, so the next stump focuses on the hard rows. Each stump also gets
a vote weight α = ½·ln((1−error)/error). The final answer is a weighted vote. Example: with error 0.3, α ≈ 0.42,
and the 3 wrong rows out of 10 end up holding half the weight. AdaBoost isn't one of the models here
(gradient boosting replaced it), but it's the classic way to explain boosting.""")
    card("What is gradient boosting?", "gradient boosting residual learning rate shrinkage gbm",
         """Each new tree is trained on the **residuals** (actual − prediction) still left, i.e. the negative gradient of
the loss. Start from the average, add a small step (**learning rate**, e.g. 0.05) of each tree, and repeat.
Example: actual 32, start 30 → residual +2 → tree predicts +2 → new prediction 30 + 0.5·2 = 31 → residual +1 → …
All four boosting models here are gradient boosting.""")
    card("What is early stopping and why is it needed?", "early stopping overfitting validation rounds iterations",
         f"""Boosting keeps lowering *training* error, but after a while it fits noise and *validation* error rises.
**Early stopping** watches the 2021 validation error and stops when it hasn't improved for 50–100 rounds.
Trees kept: Tmax XGBoost {bi['tmax']['XGBoost']}, LightGBM {bi['tmax']['LightGBM']}, CatBoost {bi['tmax']['CatBoost']};
rain amount {bi['rain_amount']['XGBoost']}/{bi['rain_amount']['LightGBM']}/{bi['rain_amount']['CatBoost']}; rain yes/no
{bi['rain_flag']['XGBoost']}/{bi['rain_flag']['LightGBM']}/{bi['rain_flag']['CatBoost']}, out of 2,000–3,000 allowed.
Rain amount stopped earliest because it has the least real signal.""")
    card("What is XGBoost?", "xgboost xgb extreme gradient boosting regularisation",
         f"""**XGBoost** grows trees **level by level**, adds **regularisation** (penalties on leaf values and leaf count),
uses second-order gradients, and learns where to send missing values. Here: learning rate 0.05, depth 6, 80%
rows and columns per tree, early stopping. It had the **best average rank across all three tasks**
(Tmax {t['tmax']['XGBoost']['MAE']:.4f}, rain {t['rain_amount']['XGBoost']['MAE']:.4f}, AUC {t['rain_flag']['XGBoost']['ROC_AUC']:.4f}).""")
    card("What is LightGBM?", "lightgbm lgbm light gradient boosting leaf-wise histogram",
         f"""**LightGBM** (Microsoft) grows trees **leaf by leaf**, always splitting the leaf that helps most, so trees are deep
and lopsided. **Histogram binning** makes it very fast, with extra tricks (GOSS, EFB). It can overfit more
easily. Here: 63 leaves, learning rate 0.05, early stopping. Tmax {t['tmax']['LightGBM']['MAE']:.4f} °C.""")
    card("What is CatBoost?", "catboost categorical ordered target statistics symmetric oblivious",
         f"""**CatBoost** (Yandex) grows **symmetric (oblivious) trees**, with the same question across a whole level, which is a
strong regulariser. It handles **categories natively** with ordered target statistics (each row's category is
encoded only from earlier rows, so there's no leakage), and uses **ordered boosting**. Here it got the **real city
name** and was the **best single model on rain amount** ({t['rain_amount']['CatBoost']['MAE']:.4f}).""")
    card("What is HistGradientBoosting?", "histgradientboosting histgb sklearn hist",
         f"""scikit-learn's own LightGBM-style (histogram) booster. Here: a fixed 400 rounds, learning rate 0.05, no early
stopping, as a plain reference. It turned out to be the **best single model for Tmax** ({t['tmax']['HistGB']['MAE']:.4f} °C).""")
    card("XGBoost vs LightGBM vs CatBoost: what's the difference?", "difference compare xgboost lightgbm catboost versus vs",
         """* **Tree growth**: XGBoost level-wise; LightGBM leaf-wise; CatBoost symmetric.
* **Speed**: LightGBM is usually fastest to train; CatBoost is fastest to predict.
* **Categories**: CatBoost is best (native); the others need encoding.
* **Overfitting risk**: LightGBM is the highest (deep leaves); CatBoost the lowest.

Here all four finished **within a few hundredths** of each other: the real gap was boosting vs bagging, not one library vs another.""")
    card("What is stacking?", "stacking stacked generalisation meta learner level base models",
         f"""**Stacking** trains a **meta-learner** on the predictions of several *different* base models. Here: **Random Forest,
LightGBM, CatBoost and Linear** → **Ridge** (regressions) / **Logistic Regression** (rain yes/no). The
meta-learner learns how much to trust each model. **It won all three tasks** (Tmax {pt['tmax']['score']:.4f} °C,
rain {pt['rain_amount']['score']:.4f}, AUC {pt['rain_flag']['score']:.4f}).""", "Model Arena")
    card("What are out-of-fold (OOF) predictions?", "out of fold oof kfold meta learner honest predictions stacking",
         """If the meta-learner learned from predictions on rows the base models had trained on, it would trust whichever
model memorised most. So the training years are split into 5 blocks (`KFold(5, shuffle=False)`, about 2 years
each). Each block is predicted by models trained on the other four, so **every row gets an honest prediction**.
The meta-learner trains on those. It was built by hand because sklearn's `StackingRegressor` won't take
`TimeSeriesSplit`.""")
    card("Why did stacking only win by a small margin?", "stacking small margin why correlated errors diversity",
         """Stacking helps only when base models make **different** mistakes. RF, LightGBM and CatBoost are all tree ensembles
on the same features, so their errors are highly correlated (see the heatmap on the Explain page). The
**linear model** is the odd one out: weaker alone but wrong in different ways, so the meta-learner gives it
real weight.""", "Explain")
    card("Bagging vs boosting vs stacking", "bagging vs boosting stacking difference compare versus",
         """| | Bagging | Boosting | Stacking |
|---|---|---|---|
| Built | in parallel | one after another | different models + meta-model |
| Combined by | average / vote | weighted sum | trained meta-learner |
| Reduces | variance | bias | both, a bit |
| Base model | deep tree | shallow tree | anything |
| Overfits by adding models? | no | yes (use early stopping) | yes, if not out-of-fold |""")

    # ---- hybrids
    s, a = hy["RF + SARIMA (Colombo weekly temp)"], hy["RF + ANN (next-day Tmax)"]
    card("What is the RF + SARIMA hybrid?", "sarima rf hybrid time series seasonal trend residual colombo weekly",
         f"""For **Colombo's weekly mean temperature**: **SARIMA** (order (1,0,1), seasonal (0,1,1,52), with a constant) learns
the yearly season and trend, then a **random forest learns SARIMA's leftover errors** from last week's rain,
sunshine and wind. Final = SARIMA + RF residual. Test MAE: SARIMA {s['SARIMA']['MAE']:.3f}, **hybrid
{s['RF+SARIMA']['MAE']:.3f}**, plain RF {s['RF']['MAE']:.3f}. The hybrid beat SARIMA. Plain RF won here only
because 2022–23 stayed inside the training range, and trees can't extrapolate a warming trend, while SARIMA can.""", "Hybrids")
    card("What is the RF + ANN hybrid?", "ann neural network mlp blend residual hybrid rf",
         f"""An **ANN** (`MLPRegressor`, 128 → 64 neurons) is combined with the RF for next-day Tmax in two ways:
1. **Blend**: w·RF + (1−w)·ANN with w = {metrics['rf_ann_blend_weight']} chosen on 2021 → **MAE {a['Blend (w_RF=0.8)']['MAE']:.3f}** (the best).
2. **Residual**: the ANN learns the RF's out-of-fold errors, final = RF + λ·ANN with λ = {metrics['rf_ann_residual_lambda']} → MAE {a['RF + ANN residual (lambda=0.15)']['MAE']:.3f}.

RF alone {a['RF']['MAE']:.3f}, ANN alone {a['ANN']['MAE']:.3f}. The blend wins because the two models' mistakes are only partly alike.""", "Hybrids")
    card("Why can't trees extrapolate?", "extrapolate extrapolation trend beyond range warming trees limit",
         """Each leaf predicts an **average of training targets**, so a tree or forest can never predict above the hottest
(or below the coolest) value it has seen. A model with a trend term (like SARIMA) can carry the level forward.
That's the reason for the RF + SARIMA design.""", "Hybrids")

    # ---- results
    card("Which model won overall?", "winner best model won overall results which model best",
         f"""**Stacking won all three tasks** on the 2022–23 test set:
* Max temp: **{pt['tmax']['score']:.4f} °C MAE** (runner-up {nice(pt['tmax']['runner_up'])} {pt['tmax']['runner_score']:.4f}); {pt['tmax']['gain']}.
* Rain amount: **{pt['rain_amount']['score']:.4f} MAE** (runner-up {nice(pt['rain_amount']['runner_up'])} {pt['rain_amount']['runner_score']:.4f}).
* Rain yes/no: **{pt['rain_flag']['score']:.4f} ROC-AUC** (runner-up {nice(pt['rain_flag']['runner_up'])} {pt['rain_flag']['runner_score']:.4f}).

Best single models: {nice(pt['tmax']['best_single'])} (Tmax), {nice(pt['rain_amount']['best_single'])} (rain amount),
{nice(pt['rain_flag']['best_single'])} (rain yes/no). Every scenario is on the Verdict page.""", "Verdict")
    for sc in v["scenarios"]:
        card(f"Scenario: {sc['title']}",
             "scenario best when which choose recommend " + sc["title"].lower(),
             f"**{nice(sc['winner'])}**. {sc['why']} *Trade-off:* {sc['cost']}", "Verdict")
    card("Results for max temperature", "tmax max temperature results scores table",
         f"Test MAE (°C), best first: {_rows(metrics, 'tmax', 6)}. R² ≈ {t['tmax']['Stacking']['R2']:.2f}.", "Model Arena")
    card("Results for rain amount", "rain amount results scores table mae",
         f"Test MAE (log1p mm), best first: {_rows(metrics, 'rain_amount', 6)}. R² is only ≈ {t['rain_amount']['Stacking']['R2']:.2f}: rain is hard.", "Model Arena")
    card("Results for will it rain", "rain flag yes no results roc auc accuracy f1 classification",
         f"Test ROC-AUC, best first: {_rows(metrics, 'rain_flag', 6)}. Stacking accuracy {t['rain_flag']['Stacking']['Accuracy']:.1%}, "
         f"F1 {t['rain_flag']['Stacking']['F1']:.3f}. The linear model is almost as good, because the signal is mostly additive.", "Model Arena")
    card("Why is rain harder to predict than temperature?", "rain hard harder difficult why temperature easier",
         """Tomorrow's max temperature ≈ today's + a small seasonal nudge, so even persistence is strong (R² ≈ 0.85). Rain is
patchy and storm-driven: an afternoon thunderstorm depends on local conditions that daily grid averages can't
see. The models learn the **season** well (good ROC-AUC for "will it rain?"), but the exact **amount** stays
noisy (R² ≈ 0.48).""")
    card("Why did boosting beat bagging?", "boosting beat bagging why better",
         """Bagging averages deep trees, which lowers variance but keeps their **step-like bias**. Boosting adds trees that fix
what is still wrong, so it captures smooth effects ("a little warmer each day through April"). That's why
boosting led both regressions. For rain yes/no, every ensemble was close to the linear model.""")
    card("Which features matter most?", "feature importance important features shap permutation top",
         """* **Max temp**: today's Tmax, the 7-day average Tmax, today's mean temperature, Tmax 1–2 days ago, rain hours.
* **Rain yes/no**: today's rain amount and rain hours, the weather code, yesterday's rain hours, ET0, time of year.

Measured with **permutation importance** (shuffle a column and see how much the score drops; fairer than the RF's
built-in importance) and **SHAP** (direction and size of each feature's push). In short: *today is the best
guide to tomorrow, and the last week tells you the season.*""", "Explain")
    card("What is SHAP?", "shap shapley explain explanation values",
         """**SHAP** splits each prediction into contributions from each feature (based on Shapley values from game theory).
The Explain page shows which features push forecasts up or down, and by how much, for the best boosting model.""", "Explain")
    card("What are the limits of this study?", "limits limitations weakness caveat",
         """* **Reanalysis data**, not real station readings, so local extremes are smoothed.
* **Nearby cities share grid cells**, so there are fewer than 30 truly independent series.
* The **test period is only 18 months**, so one unusual monsoon could change the rankings.""")

    # ---- dashboard
    card("How do I use this dashboard?", "dashboard pages navigate how use help tour menu",
         """Pages (top bar; ☰ on phones):
* **Overview**: key numbers, the **3D Sri Lanka map** (switch rain / temperature / elevation, pause the rotation) and the time split.
* **Explore**: the data by city and season.
* **Model Arena**: all models compared for each task.
* **Hybrids**: RF + SARIMA and RF + ANN.
* **Explain**: feature importance, SHAP and error correlation.
* **Predict**: pick a city and a test day to see every model's forecast next to what happened.
* **Verdict**: which model wins in which scenario.
* **Ask**: this chat.

Switch between the light and dark themes from the ⋮ menu.""")
    card("What does the 3D map show?", "3d map island deck.gl columns pause rotate overview",
         """On the Overview page each city is a column on a 3D Sri Lanka: its **height and colour** show annual rainfall, average
max temperature or elevation (use the buttons). Drag to rotate; use **Pause** to stop the animation. It uses
deck.gl (WebGL), with an SVG version for browsers where WebGL is off.""", "Overview")
    card("What does the Explore page show?", "explore page data city season charts",
         """**Explore** lets you pick cities and look at the raw weather: daily and monthly rainfall, temperature over
time, and how rain differs by **monsoon season** (SW, NE, inter-monsoon). It's the place to see the monsoon
pattern and the hill-country vs coast temperature gap.""", "Explore")
    card("What does the Model Arena page show?", "model arena page leaderboard compare models chart",
         """**Model Arena** ranks every model for the task you pick (max temp, rain amount, rain yes/no), coloured by family
(baseline, bagging, boosting, stacking), with the full metrics table, training times and the
random-vs-time-split leakage check.""", "Model Arena")
    card("What does the Hybrids page show?", "hybrids page sarima ann blend chart",
         """**Hybrids** plots the RF + SARIMA weekly forecasts for Colombo against SARIMA and plain RF, and shows the
RF + ANN blend weight and residual results.""", "Hybrids")
    card("What does the Explain page show?", "explain page feature importance shap correlation",
         """**Explain** shows which features drive the forecasts: random-forest impurity and permutation importance,
SHAP summaries for the best boosting model, and the error-correlation heatmap that explains why stacking only
wins by a little.""", "Explain")
    card("What does the Verdict page show?", "verdict page summary scenario winner which model when",
         """**Verdict** is the final summary: the winner for each task, a **scenario guide** (most accurate, best single
model per task, fastest, simplest, one model for everything, trends, neural network) and a rank table of every
model across the three tasks.""", "Verdict")
    card("How does the Predict page work?", "predict page demo city day forecast actual",
         """Pick a **city** and a **day from the 2022–23 test period**. The page shows every model's forecast for the next day's
rain chance, rain amount and max temperature, next to what actually happened. The values come from saved
test predictions; nothing is retrained.""", "Predict")
    card("Does the dashboard retrain models?", "retrain train live deploy artifacts how works streamlit",
         """No. The notebook trains everything and saves `artifacts/` (metrics, test predictions, figures). The Streamlit app
only reads those files, so it starts fast. It's deployed on Streamlit Community Cloud from GitHub and redeploys
on every push.""")
    card("How was AI used in this project?", "ai use claude declaration generated who did what",
         """Claude Code helped write the code (notebook cells, `src/`, the dashboard). Every AI-assisted file is labelled. The
author chose the dataset and questions, reviewed the checks, interpreted the results, wrote the findings and
recorded the video. See the AI-use section at the end of the notebook.""")
    return K


# ------------------------------------------------------------------ search (BM25)
SYN = {"rf": "random forest", "xgb": "xgboost", "lgbm": "lightgbm", "lgb": "lightgbm", "cb": "catboost",
       "gbm": "gradient boosting", "nn": "neural network ann", "mlp": "ann neural network",
       "temp": "temperature tmax", "tmax": "max temperature", "auc": "roc auc", "oof": "out of fold",
       "eda": "data pattern", "best": "best winner won", "win": "winner won best", "wins": "winner won best",
       "rainfall": "rain", "won": "winner won best", "winning": "winner won best", "precipitation": "rain", "hot": "temperature", "fast": "fast speed time training",
       "quickest": "fast speed", "leak": "leakage"}
STOP = set("a an the is are was were be to of and or in on for with what why how which does do did i me my "
           "this that it its as at by from you your can could would should about tell explain please who "
           "work works show shows mean means use used get".split())


def _tok(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9²]+", text.lower())
    out = []
    for w in words:
        out.extend(SYN.get(w, w).split())
    return [w for w in out if w not in STOP]


class Index:
    def __init__(self, kb):
        self.kb = kb
        self.docs = [_tok(c["title"] + " " + c["title"] + " " + c["keys"] + " " + c["keys"] + " " + c["answer"])
                     for c in kb]
        self.titles = [set(_tok(c["title"])) for c in kb]
        self.df = Counter(w for d in self.docs for w in set(d))
        self.avg = sum(map(len, self.docs)) / len(self.docs)

    def search(self, q: str, k: int = 3, min_score: float = 2.5):
        qt = _tok(q)
        n = len(self.docs)
        scores = []
        for i, d in enumerate(self.docs):
            tf = Counter(d)
            s = 0.0
            for w in set(qt):
                if w not in tf:
                    continue
                idf = math.log(1 + (n - self.df[w] + .5) / (self.df[w] + .5))
                s += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (.25 + .75 * len(d) / self.avg))
            if s:
                hit = set(qt) & self.titles[i]
                s += 1.5 * len(hit) + 3 * len(hit) / max(len(self.titles[i]), 1)
            scores.append((s, i))
        scores.sort(key=lambda r: (-r[0], r[1]))
        return [(s, self.kb[i]) for s, i in scores[:k] if s >= min_score]


def grounding_text(kb) -> str:
    """All cards as one reference document for Claude's system prompt."""
    return "\n\n".join(f"### {c['title']}\n{c['answer']}" for c in kb)
